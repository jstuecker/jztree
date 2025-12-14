#ifndef KNN_H
#define KNN_H

#include <cub/cub.cuh>
#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"
#include "common/knn_math.cuh"
#include "xla/ffi/api/ffi.h"
#include "common/segment_sort.cuh"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

namespace ffi = xla::ffi;

struct Neighbor {
    float r2;
    int id;
};

struct PosR {
    float3 pos;
    float r;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         Nearest K Heap                                         */
/* ---------------------------------------------------------------------------------------------- */

template<int K>
struct SortedNearestK {
    // Keeps track of the nearest K neighbors that we have seen so far in ascending order of r2.
    // Offers efficient insertion of new candidates

    float r2s[K];
    int   ids[K];

    __device__ __forceinline__ SortedNearestK(float r2=INFINITY, int idx=-1){
        #pragma unroll
        for (int i=0;i<K;++i){ 
            r2s[i]=r2;
            ids[i]=idx;
        }
    }

    __device__ __forceinline__ float max_r2() const {
        return r2s[K-1]; 
    }

    // Insert (r2,id) into ascending r2s[], evicting the largest.
    // I have tried many different variants of this function and this turned out to be the fastest.
    // Important aspects:
    // * no dynamic indexing
    // * no breaks/returns in the loop (stops the compiler from unrolling)
    // * It seems the pattern with the mask helps the compiler to skip unnecessary work
    //   -- effectively achieving an early exit (still not totally sure how it works though...)
    __device__ __forceinline__ void consider(float r2, int id) {
        if (r2 > r2s[K-1]) return;

        bool  active   = true;  // true until we've placed the item

        #pragma unroll
        for (int i = K-2; i >= 0; --i) {
            bool take = active && (r2 < r2s[i]);

            r2s[i+1] = active ? (take ? r2s[i] : r2) : r2s[i+1];
            ids[i+1] = active ? (take ? ids[i]  : id) : ids[i+1];

            active = take;
        }

        // If we never placed, item belongs at slot 0
        r2s[0] = active ? r2 : r2s[0];
        ids[0] = active ? id : ids[0];
    }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                     Ilist based KNN kernel                                     */
/* ---------------------------------------------------------------------------------------------- */

template <int k>
__global__ void IlistKNN(
    const PosR* xT,             // input positions
    const PosR* xQ,             // query positions
    const int* isplitT,         // leaf-ranges in A
    const int* isplitQ,         // leaf-ranges in B
    const int* ilist,           // interaction list
    const float* ir2list,       // (lower) interaction rmax2
    const int* ilist_splitsQ,   // B leaf-ranges in ilist
    Neighbor* knn,              // output knn list
    float boxsize               // if > 0, use for periodic wrapping
) {
    int ileafQ = blockIdx.x;
    int iqstart = isplitQ[ileafQ], iqend = isplitQ[ileafQ + 1];
    int max_part_smem = blockDim.x;

    for(int qoff=iqstart; qoff < iqend; qoff+=blockDim.x) {
        int ipartQ = min(qoff + threadIdx.x, iqend - 1);

        float3 posQ = xQ[ipartQ].pos;

        SortedNearestK<k> nearestK(INFINITY, -1);

        extern __shared__ PosId particles[];

        PrefetchList2<int,float> pf_ilist(ilist, ir2list, ilist_splitsQ[ileafQ], ilist_splitsQ[ileafQ + 1]);

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int ileafT = interaction.first;
            float r2T = interaction.second;

            // r2T are the lower leaf-leaf distances and the interactions are sorted by these rmax2
            // Once we encounter an interaction that is farther away than any of the current nearest 
            // neighbors, we can skip all subsequent interactions. Since the interaction lists are 
            // build on worst case assumptions, this saves a lot of time in practice!
            bool any_accept = syncthreads_or(r2T <= nearestK.max_r2() * (1.0f + 1e-6f));
            if(!any_accept) break;

            /* Now load the leaf */
            int ipartTstart = isplitT[ileafT];
            int ipartTend = isplitT[ileafT + 1];
            for(int ioff = ipartTstart; ioff < ipartTend; ioff += max_part_smem) {
                int nload = min(max_part_smem, ipartTend - ioff);
                for(int i = threadIdx.x; i < nload; i += blockDim.x)
                    particles[i] = {xT[ioff + i].pos, ioff + i};
                __syncthreads();

                // Now search for the nearest neighbors in A
                for (int j = 0; j < nload; j++) {
                    PosId p = particles[j];
                    float r2 = distance_squared(p.pos, posQ, boxsize);
                    nearestK.consider(r2, p.id);
                }

                __syncthreads();
            }
        }
        
        if(qoff + threadIdx.x < iqend) {
            #pragma unroll
            for(int i = 0; i < k; i++) {
                knn[ipartQ * k + i] = {sqrtf(nearestK.r2s[i]), nearestK.ids[i]};
            }
        }
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Histogram Data Structures                                   */
/* ---------------------------------------------------------------------------------------------- */

/* A histogram that can live in registers. As usual with complex data structures in registers,
   we have to do some extra work when updating, so that we can work with static indices only.
*/
template<int BINS>
struct CumHist {
    int c[BINS];
    __device__ __forceinline__ CumHist() {
        #pragma unroll
        for (int i=0; i<BINS; i++) 
            c[i] = 0;
    }
    __device__ __forceinline__ void insert(int b, int num=1) {
        #pragma unroll
        for (int i=0; i<BINS; i++) {
            c[i] += num*(i >= b);
        }
    }

    __device__ __forceinline__  int find(int b) const {
        // Find the smallest bin that is >= b
        int ibin = BINS;
        #pragma unroll
        for(int i = BINS-1; i >= 0; i--) {
            ibin = (c[i] >= b) ? i : ibin;
        }
        return ibin;
    }

    __device__ __forceinline__  int get(int b) const {
        // get the value, using static indexing only
        int val = 0;
        #pragma unroll
        for(int i = BINS-1; i >= 0; i--) {
            val = (i == b) ? c[i] : val;
        }
        return val;
    }
};

template <int BINS>
struct LogBinMap {
    float logrbase2;
    float bins_per_log2;
    #define OFFSET 0.99f
    // We use the OFFSET to make the first bin almost exclusively contain the rbase2 case
    // This helps to not include unnecessary leaves when the immediate neighboord is sufficient

    __device__ __forceinline__ LogBinMap(float rbase2, float bins_per_log2) {
        this->logrbase2 = __log2f(rbase2);
        this->bins_per_log2 = bins_per_log2;
    }

    __device__ __forceinline__ int r2_to_bin(float r2)
    {
        if(isnan(r2) || (r2 >= INFTY)) return BINS; // overflow bin

        float logr2 = __log2f(r2);
        return __float2int_rd((logr2 - logrbase2) * 0.5f * bins_per_log2 + OFFSET);
    }

    __device__ __forceinline__ float bin_end(int ibin)
    {
        if(ibin < BINS)
            return __powf(2.0f, logrbase2 + float(ibin+1.0f-OFFSET)*(2.0f/bins_per_log2));
        else
            return INFTY;
    }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                    Interaction list Building                                   */
/* ---------------------------------------------------------------------------------------------- */

#define INTERACTION_BINS 16

__global__ void KernelCountInteractions(
    const Node* leaves,
    const int* leaves_npart,
    const int* isplit,
    const int* node_ilist,
    const float* node_ir2list,
    const int* node_ilist_splits,
    int* interaction_count,
    float* rmax_out,
    int k,
    float bins_per_log2,
    float boxsize
) {

    // Finds the minimal radius at which we are guaranteed to have at least k neighbors for any
    // point of a leaf and counts the number of other leaves that are needed to interact at that 
    //  distance

    // This is done in 3 steps:
    // (1) We make a histogram of the number of particles that are guaranteed to be included at
    //     a distance r. (Note that this depends on the upper bound of the distance between leaves)
    // (2) We find the radius at which we have at least k particles
    // (3) We go through the leaves again and count how many we need to check at that distance
    //     (Note that here we need to use the lower bound of the distance between leaves)

    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;
    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    for(int iqoff = ileafQ_start; iqoff < ileafQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last leaf to avoid adding many conditionals
        int ileafQ = min(iqoff + threadIdx.x, ileafQ_end - 1); 
        Node leafQ = leaves[ileafQ];
        float3 xQ = leafQ.center;
        float3 extQ = LvlToHalfExt(leafQ.level);

        // We will define bins in units of the diagonal size of leafQ
        // This is the radius at which every point in Q would include every other point in Q
        float rbase2 = 4.0f*dotf3(extQ, extQ); // factor 4, since ext is half the node size
        
        LogBinMap<INTERACTION_BINS> binmap(rbase2, bins_per_log2);
        CumHist<INTERACTION_BINS> rhist;

        PrefetchList2<int,float> pf_ilist(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

        float rmax2 = INFINITY;

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int nodeT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmax2);
            if(!any_accept) break;

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);
            int* npartT = reinterpret_cast<int*>(extT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < ileafT_end) {
                    Node leafT = leaves[ilT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                    npartT[threadIdx.x] = leaves_npart[ilT];
                }
                __syncthreads();
                
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    float r2 = maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    int bin = binmap.r2_to_bin(r2);
                    rhist.insert(bin, npartT[j]);
                }
                __syncthreads();

                int ibin = rhist.find(k);
                // add a small safety margin, since our floating point operations might not
                // be exactly invertible
                rmax2 = binmap.bin_end(ibin) * (1.f + 1e-6f); 
            }
        }

        // // Now we have to go again through all leaves and count how many we need to interact with
        // // Note that here we need to use the minimal distance, not the maximal one
        // // since we need to include any leaf that may contain particles within rmax
        int ncount = 0;

        pf_ilist.restart(node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int nodeT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmax2);
            if(!any_accept) break;

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < ileafT_end) {
                    Node leafT = leaves[ilT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                }
                __syncthreads();
                
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    ncount += (r2 <= rmax2);
                }
                __syncthreads();
            }
        }

        // Output our results:
        if(iqoff + threadIdx.x < ileafQ_end) {
            rmax_out[ileafQ] = rmax2;
            interaction_count[ileafQ] = ncount;
        }
    }
}

__global__ void KernelInsertInteractions(
    const Node* leaves,
    const int* isplit,
    const int* node_ilist,
    const float* node_ir2list,
    const int* node_ilist_splits,
    const int* out_splits,
    const float* rmax2,
    int* ilist_out,
    float* ilist_radii,
    int nmax,
    float boxsize
) {
    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;

    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    for(int iqoff = ileafQ_start; iqoff < ileafQ_end; iqoff += blockDim.x) {
        int ileafQ = min(iqoff + threadIdx.x, ileafQ_end - 1);
        Node leafQ = leaves[ileafQ];
        float3 xQ = leafQ.center;
        float3 extQ = LvlToHalfExt(leafQ.level);
        float rmaxQ2 = rmax2[ileafQ];
        int out_offset = out_splits[ileafQ];

        int ninserted = 0;

        PrefetchList2<int,float> pf_ilist(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);
        
        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int nodeT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmaxQ2);
            if(!any_accept) break;

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < ileafT_end) {
                    Node leafT = leaves[ilT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                }
                __syncthreads();
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    if((iqoff + threadIdx.x < ileafQ_end) && (r2 <= rmaxQ2)){
                        int offset = out_offset + ninserted;

                        if(offset >= nmax)
                            break; // We have run out of space, just return

                        if(r2 == 0.f) {
                            // For the direct neighbourhood we add a tiny contribution of the maximum
                            // distance so that sorting guarantees that we start with the leaf itself
                            r2 = 1e-10f*maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);
                        }

                        ilist_radii[offset] = r2;
                        ilist_out[offset] = itoff + j;
                        ninserted += 1;
                    }
                }
                __syncthreads();
            }
        }
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Host function                                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error ConstructIlist(
    cudaStream_t stream,
    const Node* leaves,
    const int* leaves_npart,
    const int* isplit,
    const int* node_ilist,
    const float* node_ir2list,
    const int* node_ilist_splits,
    // outputs
    float* rmax2,
    int* leaf_ilist,
    float* leaf_ilist_rad,
    int* leaf_ilist_splits,
    // parameters
    const int k,
    const size_t blocksize_fill,
    const size_t blocksize_sort,
    const float rfac_maxbin,
    float boxsize,
    // parameters that can be infered inside ffi:
    const int nnodes,
    const int nleaves,
    const size_t leaf_ilist_size
) {
    cudaMemsetAsync(leaf_ilist_splits, 0, sizeof(int)*(nleaves+1), stream);

    constexpr int BINS = INTERACTION_BINS;
    float bins_per_log2 = BINS / log2f(rfac_maxbin);

    size_t smem_alloc_size = blocksize_fill * (2*sizeof(float3) + sizeof(int));

    KernelCountInteractions<<< nnodes, blocksize_fill, smem_alloc_size, stream>>>(
        leaves, leaves_npart, isplit, node_ilist, node_ir2list, node_ilist_splits,
        leaf_ilist_splits + 1, rmax2,
        k, bins_per_log2, boxsize
    );

    // Get the prefix sum with CUB
    // We can use the interaction list array as a temporary stoarge
    // This should easily fit in general, but better check that it actually does:
    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(
        nullptr, tmp_bytes, leaf_ilist_splits + 1, leaf_ilist_splits + 1,  nleaves, stream
    ); // determine the needed allocation size for CUB:

    if (tmp_bytes > leaf_ilist_size * sizeof(int)) {
        return ffi::Error(ffi::ErrorCode::kOutOfRange,
            "Scan allocation too small!  Needed: " +  std::to_string(tmp_bytes) + " bytes." + 
            "Have:" + std::to_string(leaf_ilist_size * sizeof(int)) + " bytes. ");
    }
    cub::DeviceScan::InclusiveSum(
        leaf_ilist, tmp_bytes, leaf_ilist_splits + 1, leaf_ilist_splits + 1, nleaves, stream
    );

    // Now insert the interactions
    smem_alloc_size = blocksize_fill * 2*sizeof(float3);
    KernelInsertInteractions<<< nnodes, blocksize_fill, smem_alloc_size, stream>>>(
        leaves, isplit, node_ilist, node_ir2list, node_ilist_splits, leaf_ilist_splits, rmax2,
        leaf_ilist, leaf_ilist_rad, // output
        leaf_ilist_size, boxsize    // parameters
    );

    // Now sort the interaction list segments. (This will allow early exit in the knn search)
    // Note: We cannot use CUB here, since its segmented search is not jax-compatible. 
    // (Probably because it uses dynamic dispatches internally)
    // Since most segments are small enough to be sorted in shared memory, this adds a very
    // small overhead (~ O(2ms) for 1M particles). So it is well worth it.
    int smem_size = 512;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));
    segmented_bitonic_sort_kv<<<nleaves, blocksize_sort, smem_bytes, stream>>>(
        leaf_ilist_rad, leaf_ilist, leaf_ilist_splits, nleaves, smem_size
    );
    
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           SegmentSort                                          */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SegmentSort(
    cudaStream_t stream,
    const float* key,
    const int32_t* val,
    const int32_t* isplit,
    float* key_out,
    int32_t* val_out,
    const int32_t nkeys,
    const int32_t nsegs,
    const size_t smem_size
) {
    int blocksize = 64;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));

    cudaMemcpyAsync(key_out, key, nkeys*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(val_out, val, nkeys*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    segmented_bitonic_sort_kv<<< nsegs, blocksize, smem_bytes, stream>>>(
        key_out, val_out, isplit, nsegs, smem_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

#endif // KNN_H