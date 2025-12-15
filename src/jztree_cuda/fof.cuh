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

__global__ void KernelNodeFoFEvalAndCount(
    const int* node_igroup,
    const int* node_ilist_splits,
    const int* node_ilist,
    const int* isplit,
    const Node* leaves,
    // Inputs (2)
    const int* ilist_out_splits,
    // Outputs (1)
    int* leaf_igroup_out,
    int* interaction_count,
    // Outputs (2)
    int* ilist_out,
    // Parameters:
    bool mode_insert,
    float r2link,
    float boxsize,
    int ilist_out_size
) {
    // The goal of this kernel is to find for each leaf in our node the lowest leaf that is 
    // completely contained within the linking and to add uncertain leaves into an interaction
    // list
    //
    // Depending on mode_insert it does the following:
    // mode_insert == false:
    //     (1a) determine the minimum leaf index of those that are completely in the linking length
    //     (1b) and count the number of interactions that need to be evaluated at a finer level
    // mode_insert == true:
    //     (2)  insert the interactions into a list

    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;
    int nodeQ_igroup = node_igroup[nodeQ];

    // bool node_linked = (nodeQ_igroup != nodeQ);
    // Have to find a way to properly identify self-links!

    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    for(int iqoff = ileafQ_start; iqoff < ileafQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last leaf to avoid adding many conditionals
        int ileafQ = min(iqoff + threadIdx.x, ileafQ_end - 1); 
        bool valid = iqoff + threadIdx.x < ileafQ_end;
        Node leafQ = leaves[ileafQ];
        float3 xQ = leafQ.center;
        float3 extQ = LvlToHalfExt(leafQ.level);
        
        PrefetchList<int> pf_ilist(node_ilist, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

        int ncount = 0;

        int igroupQ = ileafQ; // start pointing to self
        if(nodeQ_igroup != nodeQ) {
            // if our parent node was linked to another node,
            // then we are linked to all leaves inside of that node
            // and our leaf's group label must be at most the first leaf in that node
            igroupQ = isplit[nodeQ_igroup];
        }

        int ilist_offset;
        if(mode_insert)
            ilist_offset = ilist_out_splits[ileafQ];

        while(!pf_ilist.finished()) {
            int nodeT = pf_ilist.next();
            int nodeT_igroup = node_igroup[nodeT];

            if((nodeQ != nodeT) && (nodeT_igroup >= nodeQ_igroup))
                continue; // This node cannot have children with a lower index -> skip!

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ileafT = itoff + threadIdx.x;
                if(ileafT < ileafT_end) {
                    Node leafT = leaves[ileafT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                }
                __syncthreads();
                
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    // Upper and lower bound to the distance between any two particles in A and B:
                    float r2max = maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);
                    float r2min = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);
                    
                    // There are three cases:
                    // (1) The other leaf is completely inside the linking length -> link
                    // (2) The other leaf is partially inside the linking length -> add to ilist
                    // (3) The other leaf is outside the linking length -> do noting
                    
                    if(r2max <= r2link) {
                        if(nodeT_igroup == nodeT) // nodeT not linked, we only link to child
                            igroupQ = min(igroupQ, itoff+j);
                        else // nodeT linked, we link to wherever it is linked to
                            igroupQ = min(igroupQ, isplit[nodeT_igroup]);
                    }
                    else if (r2min <= r2link) {
                        if(mode_insert && valid && (ilist_offset + ncount < ilist_out_size))
                            ilist_out[ilist_offset + ncount] = itoff+j;
                        ncount += 1;
                    }
                }
                __syncthreads();
            }
        }

        // Output our counts
        if(valid && !mode_insert) {
            interaction_count[ileafQ] = ncount;
            leaf_igroup_out[ileafQ] = igroupQ;
        }
    }
}

ffi::Error NodeFofAndIlist(
    cudaStream_t stream,
    const int* node_igroup,
    const int* node_ilist_splits,
    const int* node_ilist,
    const int* isplit,
    const Node* leaves,
    int* leaf_igroup_out,
    int* ilist_out_splits,
    int* ilist_out,
    // Parameters:
    float r2link,
    float boxsize,
    int nnodes,
    int nleaves,
    size_t ilist_out_size,
    int block_size
) {
    size_t smem_alloc_bytes = block_size * (2*sizeof(float3));

    cudaMemsetAsync(ilist_out_splits, 0, sizeof(int)*(nleaves+1), stream);
    cudaMemsetAsync(leaf_igroup_out, 0, sizeof(int)*(nleaves), stream);
    cudaMemsetAsync(ilist_out, 0, sizeof(int)*ilist_out_size, stream);

    int* interaction_count = ilist_out_splits + 1; // use the ilist_out_splits as temporary storage

    KernelNodeFoFEvalAndCount<<< nnodes, block_size, smem_alloc_bytes, stream >>>(
        node_igroup, node_ilist_splits, node_ilist, isplit, leaves, nullptr,
        leaf_igroup_out, interaction_count, nullptr,
        false, r2link, boxsize, ilist_out_size
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

    // Now insert the interactions
    KernelNodeFoFEvalAndCount<<< nnodes, block_size, smem_alloc_bytes, stream >>>(
        node_igroup, node_ilist_splits, node_ilist, isplit, leaves, ilist_out_splits,
        nullptr, nullptr, ilist_out,
        true, r2link, boxsize, ilist_out_size
    );
    
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

#endif // FOF_H