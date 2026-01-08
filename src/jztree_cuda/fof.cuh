#ifndef FOF_H
#define FOF_H

#include <cub/cub.cuh>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/knn_math.cuh"
#include "common/iterators.cuh"

#include "xla/ffi/api/ffi.h"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

namespace ffi = xla::ffi;

struct __align__(16) PosAndIgroup {
    float3 pos;
    int igroup;
};

__device__ __forceinline__ int find_root(const int* __restrict__ igroup, int x) {
    while (true) {
        int p = abs(igroup[x]);
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
    const int* __restrict__ node_igroup,
    const int* __restrict__ isplit,
    int* __restrict__ leaf_igroup
) {
    int node = blockIdx.x;
    int node_root = node_igroup[node];

    int ileaf_start = isplit[node], ileaf_end = isplit[node + 1];
    for(int ileaf = ileaf_start + threadIdx.x; ileaf < ileaf_end; ileaf += blockDim.x) {
        if(node_root == node) { // node points to self, it was not linked anywhere
            leaf_igroup[ileaf] = ileaf;
        }
        else { 
            // If our node is linked, then we inherit the label of the first child of its root
            // Self-linked nodes are indicated through a negative self-pointer and this will 
            // be triggered for them, too!
            leaf_igroup[ileaf] = isplit[abs(node_root)];
        }
    }
}

template<int pass>
__global__ void NodeFof_Link_Count_Insert(
    const int* __restrict__ node_ilist_splits,
    const int* __restrict__ node_ilist,
    const int* __restrict__ isplit,
    const Node* __restrict__ leaves,
    const int* __restrict__ ilist_out_splits, // Input (pass 2)
    int* __restrict__ leaf_igroup, // Output (pass 0) and Input (pass 0-2)
    int* __restrict__ interaction_count, // Output (pass 1)
    int* __restrict__ ilist_out, // Output (pass 2)
    // Parameters:
    float r2link,
    float boxsize,
    int ilist_out_size
) {
    // Does a node based FoF, by evaluating guaranteed links and building an interaction list for
    // possible links at the child level
    //
    // pass 0: Link together leaf nodes base on the interaction list
    //   (step 0b: Contract links in another kernel)
    // pass 1: Count the number of uncertain interactions
    //   (step 1b: Prefix sum)
    // pass 2: Insert the uncertain interactions into a list

    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;

    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    for(int iqoff = ileafQ_start; iqoff < ileafQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last leaf to avoid adding many conditionals
        int ileafQ = min(iqoff + threadIdx.x, ileafQ_end - 1); 
        bool valid = iqoff + threadIdx.x < ileafQ_end;
        NodeWithExt leafQ = NodeLvlToHalfExt(leaves[ileafQ]);
        
        PrefetchList<int> pf_ilist(node_ilist, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

        // Note: leaf_igroup_out[ileafQ] throughout the kernel in pass 0, but not in pass 1 and 2
        //       so we can use it in pass 1 and 2 for pruning.
        int leafQ_igroup = abs(leaf_igroup[ileafQ]);

        int ncount = 0;

        int ilist_offset;
        if(pass == 2)
            ilist_offset = ilist_out_splits[ileafQ];

        while(!pf_ilist.finished()) {
            int nodeT = pf_ilist.next();

            if(nodeT < nodeQ)
                continue; // each interaction needs to be evaluated only once

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            NodeWithExt* leafT = reinterpret_cast<NodeWithExt*>(smem);
            int* igroupT = reinterpret_cast<int*>(leafT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ileafT = itoff + threadIdx.x;

                if(ileafT < ileafT_end) {
                    leafT[threadIdx.x] = NodeLvlToHalfExt(leaves[ileafT]);
                    igroupT[threadIdx.x] = abs(leaf_igroup[ileafT]);
                }
                __syncthreads();
                
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    if(!valid)
                        break;
                    ileafT = itoff+j;

                    if((leafQ_igroup == igroupT[j]) && (ileafQ != ileafT)) { 
                        // already linked -> skip
                        // Note: on pass 0 failing this this check doesn't guarantee that we are not 
                        //       linked, since leaf_igroup might have changed in the mean-time
                        //       this is fine, since we will check again when linking!
                        continue; 
                    }

                    // For not-yet-linked leaves there are three scenarios:
                    // (1) The other leaf is completely inside the linking length -> link (pass 0)
                    // (2) The other leaf is partially inside the linking length -> add to ilist (pass 1+2)
                    // (3) The other leaf is outside the linking length -> do noting

                    // Upper and lower bound to the distance between any two particles in A and B:
                    NodeWithExt lT = leafT[j];
                    float r2max = maxdist2(lT.center, leafQ.center, sumf3(leafQ.extent, lT.extent), boxsize);
                    float r2min = mindist2(lT.center, leafQ.center, sumf3(leafQ.extent, lT.extent), boxsize);
                    
                    if(r2max <= r2link) {
                        if(pass == 0)
                            link_roots(leaf_igroup, ileafQ, ileafT);
                        else if((pass == 2) && (ileafQ == ileafT) && (abs(leaf_igroup[ileafQ]) == ileafQ))
                            leaf_igroup[ileafQ] = -ileafQ; // flag self-interaction
                    }
                    else if((pass >= 1) && (r2min <= r2link)) {
                        if((pass == 2) && (ilist_offset + ncount < ilist_out_size))
                            ilist_out[ilist_offset + ncount] = itoff+j;
                        ncount += 1;
                    }
                }
                __syncthreads();
            }
        }

        // Output our counts
        if((pass == 1) && valid)
            interaction_count[ileafQ] = ncount;
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
        int igr_new = igroup[abs(igr)];
        if(igr_new == igr) break;
        igr = igr_new;
    }

    igroup[idx] = igr;
}

ffi::Error NodeFofAndIlist(
    cudaStream_t stream,
    const int* __restrict__ node_ilist_splits,
    const int* __restrict__ node_ilist,
    const int* __restrict__ isplit,
    const Node* __restrict__ leaves,
    const int* __restrict__ leaf_igroup_in,
    int* __restrict__ leaf_igroup,
    int* __restrict__ ilist_out_splits,
    int* __restrict__ ilist_out,
    // Parameters:
    const float r2link,
    const float boxsize,
    const int nnodes,
    const int nleaves,
    const size_t ilist_out_size,
    const int block_size
) {
    size_t smem_alloc_bytes = block_size * (sizeof(NodeWithExt) + sizeof(int));

    cudaMemsetAsync(ilist_out_splits, 0, sizeof(int)*(nleaves+1), stream);
    cudaMemsetAsync(leaf_igroup, 0, sizeof(int)*(nleaves), stream);
    cudaMemsetAsync(ilist_out, 0, sizeof(int)*ilist_out_size, stream);

    int* interaction_count = ilist_out_splits + 1; // use the ilist_out_splits as temporary storage

    cudaMemcpyAsync(leaf_igroup, leaf_igroup_in, nleaves*sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // pass 0: link
    NodeFof_Link_Count_Insert<0><<< nnodes, block_size, smem_alloc_bytes, stream >>>(
        node_ilist_splits, node_ilist, isplit, leaves, nullptr,
        leaf_igroup, nullptr, nullptr,
        r2link, boxsize, ilist_out_size
    );

    // contract links so that each leaf points to its root
    int contract_blocks = (nleaves + block_size - 1) / block_size;
    KernelContractLinks<<< contract_blocks, block_size, 0, stream >>>(leaf_igroup, nleaves);

    // pass 1: count interactions
    NodeFof_Link_Count_Insert<1><<< nnodes, block_size, smem_alloc_bytes, stream >>>(
        node_ilist_splits, node_ilist, isplit, leaves, nullptr,
        leaf_igroup, interaction_count, nullptr,
        r2link, boxsize, ilist_out_size
    );

    // Get the prefix sum with CUB
    // We can use the interaction list array as a temporary stoarge
    // This should easily fit in general, but better check that it actually does:
    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(
        nullptr, tmp_bytes, ilist_out_splits + 1, ilist_out_splits + 1,  nleaves, stream
    ); // determine the needed allocation size for CUB:

    if (tmp_bytes > ilist_out_size * sizeof(int)) {
        return ffi::Error(ffi::ErrorCode::kOutOfRange,
            "Scan allocation too small!  Needed: " +  std::to_string(tmp_bytes) + " bytes." + 
            "Have:" + std::to_string(ilist_out_size * sizeof(int)) + " bytes. ");
    }
    cub::DeviceScan::InclusiveSum(
        ilist_out, tmp_bytes, interaction_count, ilist_out_splits + 1, nleaves, stream
    );

    // pass 2: insert interactions
    NodeFof_Link_Count_Insert<2><<< nnodes, block_size, smem_alloc_bytes, stream >>>(
        node_ilist_splits, node_ilist, isplit, leaves, ilist_out_splits,
        leaf_igroup, nullptr, ilist_out,
        r2link, boxsize, ilist_out_size
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


__global__ void ParticleFofLink(
    const int* __restrict__ node_ilist_splits,
    const int* __restrict__ node_ilist,
    const int* __restrict__ isplit,
    const float3* __restrict__ pos,
    int* __restrict__ part_igroup,
    // Parameters:
    float r2link,
    float boxsize
) {
    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;

    int ipartQ_start = isplit[nodeQ], ipartQ_end = isplit[nodeQ + 1];
    for(int iqoff = ipartQ_start; iqoff < ipartQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last leaf to avoid adding many conditionals
        int ipartQ = min(iqoff + threadIdx.x, ipartQ_end - 1); 
        bool valid = iqoff + threadIdx.x < ipartQ_end;
        
        PrefetchList<int> pf_ilist(node_ilist, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);
        
        float3 xQ = pos[ipartQ];
        int igroupQ = abs(part_igroup[ipartQ]);

        while(!pf_ilist.finished()) {
            int nodeT = pf_ilist.next();

            if(nodeT < nodeQ)
                continue; // each interaction needs to be evaluated only once

            int ipartT_start = isplit[nodeT], ipartT_end = isplit[nodeT + 1];

            PosAndIgroup* tileT = reinterpret_cast<PosAndIgroup*>(smem);

            for(int itoff=ipartT_start; itoff < ipartT_end; itoff += blockDim.x) {
                int ipartT = itoff + threadIdx.x;

                if(ipartT < ipartT_end) {
                    tileT[threadIdx.x].pos = pos[ipartT];
                    tileT[threadIdx.x].igroup = abs(part_igroup[ipartT]);
                }
                __syncthreads();
                
                for(int j = 0; j < min(ipartT_end - itoff, blockDim.x); j++) {
                    if(!valid)
                        break;

                    float r2 = distance_squared(xQ, tileT[j].pos, boxsize);

                    if((igroupQ != tileT[j].igroup) && (r2 <= r2link))
                        link_roots(part_igroup, ipartQ, itoff+j);
                }
                __syncthreads();
            }
        }
    }
}

ffi::Error ParticleFof(
    cudaStream_t stream,
    const int* __restrict__ node_igroup,
    const int* __restrict__ node_ilist_splits,
    const int* __restrict__ node_ilist,
    const int* __restrict__ isplit,
    const float3* __restrict__ pos,
    int* __restrict__ particle_igroup,
    // Parameters:
    float r2link,
    float boxsize,
    int nnodes,
    int npart,
    int block_size
) {
    cudaMemsetAsync(particle_igroup, 0, sizeof(int)*npart, stream);

    // Initialize particle group pointers from nodes
    NodeToChildLabel<<< nnodes, block_size, 0, stream >>>(
        node_igroup, isplit, particle_igroup
    );

    // Now do particle-particle linking
    // use packed PartTile for shared positions
    size_t smem = block_size * sizeof(PosAndIgroup);
    ParticleFofLink<<< nnodes, block_size, smem, stream >>> (
        node_ilist_splits, node_ilist, isplit, pos,
        particle_igroup,
        r2link, boxsize
    );

    // contract links so that each particle points to its root
    int contract_blocks = (npart + block_size - 1) / block_size;
    KernelContractLinks<<< contract_blocks, block_size, 0, stream >>>(particle_igroup, npart);
    
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