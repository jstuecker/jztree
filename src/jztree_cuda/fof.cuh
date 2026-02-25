#ifndef FOF_H
#define FOF_H

#include <cub/cub.cuh>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

#include "xla/ffi/api/ffi.h"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

namespace ffi = xla::ffi;

__device__ __forceinline__ int find_root(const int* __restrict__ igroup, int x) {
    while (true) {
        int p = igroup[x];
        if (p == x) return x;
        x = p;
    }
}

__device__ __forceinline__ void link_roots(int* __restrict__ igroup, int a, int b) {
    while (true) {
        int ra = find_root(igroup, a);
        int rb = find_root(igroup, b);
        if (ra == rb) return;

        int hi = (ra > rb) ? ra : rb;
        int lo = (ra > rb) ? rb : ra;

        // Hook hi -> lo, but only if hi is still a root.
        // If it changed, retry.
        int old = atomicCAS(&igroup[hi], hi, lo);
        if (old == hi) return; // success
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                  Kernels for node-node linking                                 */
/* ---------------------------------------------------------------------------------------------- */

__global__ void NodeToChildLabel(
    const int* __restrict__ parent_igroup,
    const bool* __restrict__ parent_is_local,
    const int* __restrict__ parent_lvl,
    const int* __restrict__ parent_spl,
    int* __restrict__ node_igroup,
    const int size_parent,
    const float r2link
) {
    int node = blockIdx.x;
    if(node >= size_parent || !parent_is_local[node])
        return;
    int node_root = parent_igroup[node];
    if(node_root >= size_parent || !parent_is_local[node_root])
        return;
    float L2 = LvlToExt<3,float>(parent_lvl[node]).norm2();

    int inode_start = parent_spl[node], inode_end = parent_spl[node + 1];
    for(int inode = inode_start + threadIdx.x; inode < inode_end; inode += blockDim.x) {
        if((node_root == node) && (L2 > r2link)) { // node points to self, it was not linked anywhere
            node_igroup[inode] = inode;
        }
        else { 
            // If our node is linked, then we inherit the label of the first child of its root
            node_igroup[inode] = parent_spl[node_root];
        }
    }
}

template<int pass>
__global__ void NodeFof_Link_Count_Insert(
    const int* __restrict__ parent_ilist_splits,
    const int* __restrict__ parent_ilist,
    const int* __restrict__ spl,
    const Node<3,float>* __restrict__ nodes,
    const int* __restrict__ node_ilist_spl, // Input (pass 2)
    int* __restrict__ node_igroup, // Output (pass 0) and Input (pass 0-2)
    int* __restrict__ interaction_count, // Output (pass 1)
    int* __restrict__ node_ilist, // Output (pass 2)
    // Parameters:
    float r2link,
    float boxsize,
    int size_node_ilist
) {
    // Does a node based FoF, by evaluating guaranteed links and building an interaction list for
    // possible links at the child level
    //
    // pass 0: Link together nodes based on the interaction list
    //   (step 0b: Contract links in another kernel)
    // pass 1: Count the number of uncertain interactions
    //   (step 1b: Prefix sum)
    // pass 2: Insert the uncertain interactions into a list

    extern __shared__ unsigned char smem[];

    int parentQ = blockIdx.x;

    int inodeQ_start = spl[parentQ], inodeQ_end = spl[parentQ + 1];
    for(int iqoff = inodeQ_start; iqoff < inodeQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last node to avoid adding many conditionals
        int inodeQ = min(iqoff + threadIdx.x, inodeQ_end - 1); 
        bool valid = iqoff + threadIdx.x < inodeQ_end;
        NodeWithExt nodeQ = NodeLvlToHalfExt<3,float>(nodes[inodeQ]);
        
        PrefetchList<int> pf_ilist(parent_ilist, parent_ilist_splits[parentQ], parent_ilist_splits[parentQ + 1]);

        int nodeQ_igroup = node_igroup[inodeQ];

        int ncount = 0;

        int ilist_offset;
        if(pass == 2)
            ilist_offset = node_ilist_spl[inodeQ];

        while(!pf_ilist.finished()) {
            int parentT = pf_ilist.next();

            if(parentT < parentQ)
                continue; // each interaction needs to be evaluated only once

            int inodeT_start = spl[parentT], inodeT_end = spl[parentT + 1];

            NodeWithExt<3,float>* nodeT = reinterpret_cast<NodeWithExt<3,float>*>(smem);
            int* igroupT = reinterpret_cast<int*>(nodeT + blockDim.x);

            for(int itoff=inodeT_start; itoff < inodeT_end; itoff += blockDim.x) {
                int inodeT = itoff + threadIdx.x;

                if(inodeT < inodeT_end) {
                    nodeT[threadIdx.x] = NodeLvlToHalfExt<3,float>(nodes[inodeT]);
                    igroupT[threadIdx.x] = node_igroup[inodeT];
                }
                __syncthreads();
                
                for(int j = 0; j < min(inodeT_end - itoff, blockDim.x); j++) {
                    if(!valid)
                        break;
                    inodeT = itoff+j;

                    if((nodeQ_igroup == igroupT[j]) && (inodeQ != inodeT)) { 
                        // already linked -> skip
                        // Note: on pass 0 failing this this check doesn't guarantee that we are not 
                        //       linked, since node_igroup might have changed in the mean-time
                        //       this is fine, since we will check again when linking!
                        continue; 
                    }

                    // For not-yet-linked nodes there are three scenarios:
                    // (1) The other node is completely inside the linking length -> link (pass 0)
                    // (2) The other node is partially inside the linking length -> add to ilist (pass 1+2)
                    // (3) The other node is outside the linking length -> do noting

                    // Upper and lower bound to the distance between any two particles in A and B:
                    NodeWithExt<3,float> lT = nodeT[j];
                    float r2max = maxdist2<3,float>(lT.center, nodeQ.center, nodeQ.extent+lT.extent, boxsize);
                    float r2min = mindist2<3,float>(lT.center, nodeQ.center, nodeQ.extent+lT.extent, boxsize);

                    float L2 =  lT.extent.norm2() * 4.f;
                    
                    // Check whether we are guaranteed to be linked
                    if(max(r2max, L2) <= r2link) {
                        // The max(..., L2) handles a very rare scenario where r2max < L2 of the
                        // target node and only r2max < r2link. (This can happen if lQ is very small 
                        // and close). In this scenario nodeT would be linked with itself, but only 
                        // thanks to the existence of lQ and the self-linkedness of lT would not 
                        // be properly detected in NodeToChildLabel. To deal with this we simply add
                        // our node also to the interaction list and resolve it at the child-level 

                        if(pass == 0)
                            link_roots(node_igroup, inodeQ, inodeT);
                    }
                    else if((pass >= 1) && (r2min <= r2link)) {
                        if((pass == 2) && (ilist_offset + ncount < size_node_ilist))
                            node_ilist[ilist_offset + ncount] = itoff+j;
                        ncount += 1;
                    }
                }
                __syncthreads();
            }
        }

        // Output our counts
        if((pass == 1) && valid)
            interaction_count[inodeQ] = ncount;
    }
}

__global__ void KernelContractLinks(
    int* __restrict__ igroup,
    int num,
    int max_iter=100000
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num)
        return;

    int igr = igroup[idx];
    for(int i=0; i<max_iter; i++) {
        int igr_new = igroup[igr];
        if(igr_new == igr) break;
        igr = igr_new;
    }

    igroup[idx] = igr;
}

ffi::Error FofNode2Node(
    cudaStream_t stream,
    const int* __restrict__ parent_ilist_spl,
    const int* __restrict__ parent_ilist,
    const int* __restrict__ parent_spl,
    const Node<3,float>* __restrict__ nodes,
    const int* __restrict__ node_igroup_in,
    int* __restrict__ node_igroup,
    int* __restrict__ node_ilist_spl,
    int* __restrict__ node_ilist,
    // Parameters:
    const float r2link,
    const float boxsize,
    const int size_parent,
    const int size_node,
    const size_t size_node_ilist,
    const int block_size
) {
    size_t smem_alloc_bytes = block_size * (sizeof(NodeWithExt<3,float>) + sizeof(int));

    cudaMemsetAsync(node_ilist_spl, 0, sizeof(int)*(size_node+1), stream);
    cudaMemsetAsync(node_igroup, 0, sizeof(int)*(size_node), stream);
    cudaMemsetAsync(node_ilist, 0, sizeof(int)*size_node_ilist, stream);

    int* interaction_count = node_ilist_spl + 1; // use the node_ilist_spl as temporary storage

    cudaMemcpyAsync(node_igroup, node_igroup_in, size_node*sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // pass 0: link
    NodeFof_Link_Count_Insert<0><<< size_parent, block_size, smem_alloc_bytes, stream >>>(
        parent_ilist_spl, parent_ilist, parent_spl, nodes, nullptr,
        node_igroup, nullptr, nullptr,
        r2link, boxsize, size_node_ilist
    );

    // contract links so that each node points to its root
    int contract_blocks = (size_node + block_size - 1) / block_size;
    KernelContractLinks<<< contract_blocks, block_size, 0, stream >>>(node_igroup, size_node);

    // pass 1: count interactions
    NodeFof_Link_Count_Insert<1><<< size_parent, block_size, smem_alloc_bytes, stream >>>(
        parent_ilist_spl, parent_ilist, parent_spl, nodes, nullptr,
        node_igroup, interaction_count, nullptr,
        r2link, boxsize, size_node_ilist
    );

    // Get the prefix sum with CUB
    // We can use the interaction list array as a temporary stoarge
    // This should easily fit in general, but better check that it actually does:
    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(
        nullptr, tmp_bytes, node_ilist_spl + 1, node_ilist_spl + 1,  size_node, stream
    ); // determine the needed allocation size for CUB:

    if (tmp_bytes > size_node_ilist * sizeof(int)) {
        return ffi::Error(ffi::ErrorCode::kOutOfRange,
            "Scan allocation too small!  Needed: " +  std::to_string(tmp_bytes) + " bytes." + 
            "Have:" + std::to_string(size_node_ilist * sizeof(int)) + " bytes. ");
    }
    cub::DeviceScan::InclusiveSum(
        node_ilist, tmp_bytes, interaction_count, node_ilist_spl + 1, size_node, stream
    );

    // pass 2: insert interactions
    NodeFof_Link_Count_Insert<2><<< size_parent, block_size, smem_alloc_bytes, stream >>>(
        parent_ilist_spl, parent_ilist, parent_spl, nodes, node_ilist_spl,
        node_igroup, nullptr, node_ilist,
        r2link, boxsize, size_node_ilist
    );
    
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

/* ---------------------------------------------------------------------------------------------- */
/*                              Kernels for particle-particle linking                             */
/* ---------------------------------------------------------------------------------------------- */


__global__ void FofLeaf2LeafLink(
    const int* __restrict__ ilist_spl,
    const int* __restrict__ ilist,
    const int* __restrict__ spl,
    const Vec<3,float>* __restrict__ pos,
    int* __restrict__ part_igroup,
    // Parameters:
    float r2link,
    float boxsize
) {
    extern __shared__ unsigned char smem[];

    int leafQ = blockIdx.x;

    int ipartQ_start = spl[leafQ], ipartQ_end = spl[leafQ + 1];
    for(int iqoff = ipartQ_start; iqoff < ipartQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last leaf to avoid adding many conditionals
        int ipartQ = min(iqoff + threadIdx.x, ipartQ_end - 1); 
        bool valid = iqoff + threadIdx.x < ipartQ_end;
        
        PrefetchList<int> pf_ilist(ilist, ilist_spl[leafQ], ilist_spl[leafQ + 1]);
        
        Vec<3,float> xQ = pos[ipartQ];
        int igroupQ = part_igroup[ipartQ];

        while(!pf_ilist.finished()) {
            int nodeT = pf_ilist.next();

            if(nodeT < leafQ)
                continue; // each interaction needs to be evaluated only once

            int ipartT_start = spl[nodeT], ipartT_end = spl[nodeT + 1];

            PosId<3,float>* tileT = reinterpret_cast<PosId<3,float>*>(smem);

            for(int itoff=ipartT_start; itoff < ipartT_end; itoff += blockDim.x) {
                int ipartT = itoff + threadIdx.x;

                if(ipartT < ipartT_end) {
                    tileT[threadIdx.x] = {pos[ipartT], part_igroup[ipartT]};
                }
                __syncthreads();
                
                for(int j = 0; j < min(ipartT_end - itoff, blockDim.x); j++) {
                    if(!valid)
                        break;

                    float r2 = distance_squared<3,float>(xQ, tileT[j].pos, boxsize);

                    if((igroupQ != tileT[j].id) && (r2 <= r2link))
                        link_roots(part_igroup, ipartQ, itoff+j);
                }
                __syncthreads();
            }
        }
    }
}

ffi::Error FofLeaf2Leaf(
    cudaStream_t stream,
    const int* __restrict__ ilist_spl,
    const int* __restrict__ ilist,
    const int* __restrict__ spl,
    const Vec<3,float>* __restrict__ pos,
    const int* __restrict__ part_igroup_in,
    int* __restrict__ part_igroup,
    // Parameters:
    float r2link,
    float boxsize,
    int size_leaves,
    int size_part,
    int block_size
) {
    cudaMemcpyAsync(part_igroup, part_igroup_in, size_part*sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // Now do particle-particle linking
    size_t smem = block_size * sizeof(PosId<3,float>);
    FofLeaf2LeafLink<<< size_leaves, block_size, smem, stream >>> (
        ilist_spl, ilist, spl, pos, part_igroup,
        r2link, boxsize
    );

    // contract links so that each particle points to its root
    int contract_blocks = (size_part + block_size - 1) / block_size;
    KernelContractLinks<<< contract_blocks, block_size, 0, stream >>>(part_igroup, size_part);
    
    return ffi::Error::Success();
}

__global__ void KernelInsertLinks(
    const int* __restrict__ igroupLinkA,
    const int* __restrict__ igroupLinkB,
    const int* __restrict__ num_links,
    int* __restrict__ igroup
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= num_links[0])
        return;

    link_roots(igroup, igroupLinkA[idx], igroupLinkB[idx]);
}


ffi::Error InsertLinks(
    cudaStream_t stream,
    const int* __restrict__ igroup_in,    
    const int* __restrict__ igroupLinkA,
    const int* __restrict__ igroupLinkB,
    const int* num_links,
    int* __restrict__ igroup,
    const int size_links,
    const int size_groups,
    const int block_size
) {
    cudaMemcpyAsync(igroup, igroup_in, size_groups*sizeof(int), cudaMemcpyDeviceToDevice, stream);
    
    KernelInsertLinks<<< div_ceil(size_links, block_size), block_size, 0, stream>>>(
        igroupLinkA, igroupLinkB, num_links, igroup
    );

    KernelContractLinks<<< div_ceil(size_groups, block_size), block_size, 0, stream >>>(
        igroup, size_groups
    );
    
    return ffi::Error::Success();
}

#endif // FOF_H