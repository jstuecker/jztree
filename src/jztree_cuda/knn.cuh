#ifndef KNN_H
#define KNN_H

#include <cub/cub.cuh>
#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"
#include "xla/ffi/api/ffi.h"
#include "sort.cuh"

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

struct ConstInteractionList {
    const int32_t* spl;
    const int32_t* iother;
    const float* rad2 = nullptr;
};

struct InteractionList {
    int32_t* spl;
    int32_t* iother;
    float* rad2 = nullptr;
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


template<int K>
struct SortedNearestKWithCounts {
    // Keeps track of the nearest K neighbor radii in ascending order of r2.
    // Allows to insert multiple entries with the same radius

    float r2s[K];
    int cts[K];

    __device__ __forceinline__ SortedNearestKWithCounts(float r2=INFINITY){
        #pragma unroll
        for (int i=0; i<K; i++){ 
            r2s[i] = r2;
            cts[i] = 0;
        }
    }

    __device__ __forceinline__ float max_r2(int count) const {
        bool active = true;
        int sum = 0;
        float r2 = INFINITY;
        #pragma unroll
        for(int i=0; i<K; i++) {
            sum += cts[i];
            r2 = (active && (sum >= count)) ? r2s[i] : r2;
            active = sum < count;
        }
        return r2;
    }


    __device__ __forceinline__ void consider_num(float r2, int num) {
        if (r2 > r2s[K-1]) return;

        bool active = true;  // true until we've placed the item

        #pragma unroll
        for (int i = K-2; i >= 0; i--) {
            bool take = active && (r2 < r2s[i]);

            r2s[i+1] = active ? (take ? r2s[i] : r2) : r2s[i+1];
            cts[i+1] = active ? (take ? cts[i] : num) : cts[i+1];

            active = take;
        }

        // If we never placed, item belongs at slot 0
        r2s[0] = active ? r2 : r2s[0];
        cts[0] = active ? num : cts[0];
    }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                     Ilist based KNN kernel                                     */
/* ---------------------------------------------------------------------------------------------- */

template <int k>
__global__ void KnnLeaf2Leaf(
    const int* ilist_spl,       // leaf-ranges in ilist
    const int* ilist,           // interaction list
    const float* ilist_r2,      // (lower) interaction rmax2
    const int* splT,            // leaf-ranges in A
    const PosR* xT,             // input positions
    const int* splQ,            // leaf-ranges in B
    const PosR* xQ,             // query positions
    Neighbor* knn,              // output knn list
    float boxsize               // if > 0, use for periodic wrapping
) {
    int ileafQ = blockIdx.x;
    int iqstart = splQ[ileafQ], iqend = splQ[ileafQ + 1];
    int max_part_smem = blockDim.x;

    for(int qoff=iqstart; qoff < iqend; qoff+=blockDim.x) {
        int ipartQ = min(qoff + threadIdx.x, iqend - 1);

        float3 posQ = xQ[ipartQ].pos;

        SortedNearestK<k> nearestK(INFINITY, -1);

        extern __shared__ PosId particles[];

        PrefetchList2<int,float> pf_ilist(
            ilist, ilist_r2, ilist_spl[ileafQ], ilist_spl[ileafQ + 1]
        );

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
            int ipartTstart = splT[ileafT];
            int ipartTend = splT[ileafT + 1];
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

template <int kmax>
__global__ void KnnNode2NodeFindRmax(
    ConstInteractionList par_ilist,
    const int* parent_spl,
    const Node* nodes,
    const int* nodes_npart,
    float* rmax_out,
    int k,
    float boxsize
) {
    // Finds the minimal radius at which we are guaranteed to have at least k neighbors for any
    // point inside of each node
    // This is done by going through all interactions and keep track of the maximal needed 
    // radius for each k
    extern __shared__ unsigned char smem[];

    int parentQ = blockIdx.x;
    int inodeQ_start = parent_spl[parentQ], inodeQ_end = parent_spl[parentQ + 1];
    for(int iqoff = inodeQ_start; iqoff < inodeQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last node to avoid adding many conditionals
        int inodeQ = min(iqoff + threadIdx.x, inodeQ_end - 1); 
        Node nodeQ = nodes[inodeQ];
        float3 xQ = nodeQ.center;
        float3 extQ = LvlToHalfExt(nodeQ.level);

        SortedNearestKWithCounts<kmax> nearestK(INFINITY);

        PrefetchList2<int,float> pf_ilist(
            par_ilist.iother, par_ilist.rad2, par_ilist.spl[parentQ], par_ilist.spl[parentQ + 1]
        );

        float rmax2 = INFINITY;

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int parentT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmax2);
            if(!any_accept) break;

            int inodeT_start = parent_spl[parentT], inodeT_end = parent_spl[parentT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);
            int* npartT = reinterpret_cast<int*>(extT + blockDim.x);

            for(int itoff=inodeT_start; itoff < inodeT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < inodeT_end) {
                    Node nodeT = nodes[ilT];
                    xT[threadIdx.x] = nodeT.center;
                    extT[threadIdx.x] = LvlToHalfExt(nodeT.level);
                    npartT[threadIdx.x] = nodes_npart[ilT];
                }
                __syncthreads();
                
                for(int j = 0; j < min(inodeT_end - itoff, blockDim.x); j++) {
                    float r2 = maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    nearestK.consider_num(r2, npartT[j]);
                }
                __syncthreads();
            }

            rmax2 = nearestK.max_r2(k) * (1.f + 1e-6f);
        }

        // Output our results:
        if(iqoff + threadIdx.x < inodeQ_end) {
            rmax_out[inodeQ] = rmax2;
        }
    }
}

template<int pass>
__global__ void KnnNode2NodeCountInsert(
    ConstInteractionList par_ilist,
    const int* parent_spl,
    const Node* nodes,
    const float* node_rmax2,
    int* node_icount,
    InteractionList node_ilist,
    int nmax,
    float boxsize
) {
    // pass 0: Count node-node interactions that intersect node_rmax2
    //   in-between: calculate offsets
    // pass 1: Insert the interactions into the interaction list

    extern __shared__ unsigned char smem[];

    int parentQ = blockIdx.x;

    int inodeQ_start = parent_spl[parentQ], inodeQ_end = parent_spl[parentQ + 1];
    for(int iqoff = inodeQ_start; iqoff < inodeQ_end; iqoff += blockDim.x) {
        int inodeQ = min(iqoff + threadIdx.x, inodeQ_end - 1);
        Node nodeQ = nodes[inodeQ];
        float3 xQ = nodeQ.center;
        float3 extQ = LvlToHalfExt(nodeQ.level);
        float rmaxQ2 = node_rmax2[inodeQ];
        int out_offset = node_ilist.spl[inodeQ];

        int ncount = 0;

        PrefetchList2<int,float> pf_ilist(
            par_ilist.iother, par_ilist.rad2, par_ilist.spl[parentQ], par_ilist.spl[parentQ + 1]
        );
        
        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int parentT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmaxQ2);
            if(!any_accept) break;

            int inodeT_start = parent_spl[parentT], inodeT_end = parent_spl[parentT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);

            for(int itoff=inodeT_start; itoff < inodeT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < inodeT_end) {
                    Node nodeT = nodes[ilT];
                    xT[threadIdx.x] = nodeT.center;
                    extT[threadIdx.x] = LvlToHalfExt(nodeT.level);
                }
                __syncthreads();
                for(int j = 0; j < min(inodeT_end - itoff, blockDim.x); j++) {
                    float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    if((iqoff + threadIdx.x < inodeQ_end) && (r2 <= rmaxQ2)){
                        if(pass == 1) {
                            int offset = out_offset + ncount;

                            if(offset >= nmax)
                                break; // We have run out of space, just return

                            if(r2 == 0.f) {
                                // For the direct neighbourhood we add a tiny contribution of the maximum
                                // distance so that sorting guarantees that we start with the node itself
                                r2 = 1e-10f*maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);
                            }

                            node_ilist.rad2[offset] = r2;
                            node_ilist.iother[offset] = itoff + j;
                        }
                        ncount += 1;
                    }
                }
                __syncthreads();
            }
        }

        if((pass == 0) && (iqoff + threadIdx.x < inodeQ_end)) {
            node_icount[inodeQ] = ncount;
        }
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Host function                                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error KnnNode2Node(
    cudaStream_t stream,
    const int32_t* parent_ilist_spl,
    const int32_t* parent_ilist_ioth,
    const float* parent_ilist_r2,
    const int32_t* parent_spl,
    const Node* nodes,
    const int32_t* nodes_npart,
    // outputs
    float* node_rmax2,
    int32_t* node_ilist_spl,
    int32_t* node_ilist_ioth,
    float* node_ilist_r2,
    // parameters
    const int k,
    const size_t blocksize_fill,
    const size_t blocksize_sort,
    float boxsize,
    // parameters that can be infered inside ffi:
    const int size_parents,
    const int size_nodes,
    const size_t node_ilist_size
) {
    ConstInteractionList par_ilist = {parent_ilist_spl, parent_ilist_ioth, parent_ilist_r2};
    InteractionList node_ilist = {node_ilist_spl, node_ilist_ioth, node_ilist_r2};

    cudaMemsetAsync(node_ilist.spl, 0, sizeof(int32_t)*(size_nodes+1), stream);

    size_t smem_alloc_size = blocksize_fill * (2*sizeof(float3) + sizeof(int));

    if(k <= 4) {
        KnnNode2NodeFindRmax<4><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
            par_ilist, parent_spl, nodes, nodes_npart, node_rmax2, k, boxsize
        );
    }
    if(k <= 16) {
        KnnNode2NodeFindRmax<16><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
            par_ilist, parent_spl, nodes, nodes_npart, node_rmax2, k, boxsize
        );
    }
    else if(k <= 64) {
        KnnNode2NodeFindRmax<64><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
            par_ilist, parent_spl, nodes, nodes_npart, node_rmax2, k, boxsize
        );
    }
    else {
        return ffi::Error(ffi::ErrorCode::kOutOfRange, "Only supporting k up to 64 for now");
    }

    smem_alloc_size = blocksize_fill * 2*sizeof(float3);
    KnnNode2NodeCountInsert<0><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
        par_ilist, parent_spl, nodes, node_rmax2,
        node_ilist.spl + 1, node_ilist, // output
        node_ilist_size, boxsize    // parameters
    );

    // Get the prefix sum with CUB
    // We can use the interaction list array as a temporary stoarge
    // This should easily fit in general, but better check that it actually does:
    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(
        nullptr, tmp_bytes, node_ilist.spl + 1, node_ilist.spl + 1,  size_nodes, stream
    ); // determine the needed allocation size for CUB:

    if (tmp_bytes > node_ilist_size * sizeof(int)) {
        return ffi::Error(ffi::ErrorCode::kOutOfRange,
            "Scan allocation too small!  Needed: " +  std::to_string(tmp_bytes) + " bytes." + 
            "Have:" + std::to_string(node_ilist_size * sizeof(int)) + " bytes. ");
    }
    cub::DeviceScan::InclusiveSum(
        node_ilist_ioth, tmp_bytes, node_ilist.spl + 1, node_ilist.spl + 1, size_nodes, stream
    );

    // Now insert the interactions
    smem_alloc_size = blocksize_fill * 2*sizeof(float3);
    KnnNode2NodeCountInsert<1><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
        par_ilist, parent_spl, nodes, node_rmax2,
        nullptr, node_ilist, // output
        node_ilist_size, boxsize    // parameters
    );

    // Now sort the interaction list segments. (This will allow early exit in the knn search)
    // Note: We cannot use CUB here, since its segmented search is not jax-compatible. 
    // (Probably because it uses dynamic dispatches internally)
    // Since most segments are small enough to be sorted in shared memory, this adds a very
    // small overhead (~ O(2ms) for 1M particles). So it is well worth it.
    int smem_size = 512;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));
    segmented_bitonic_sort_kv<<<size_nodes, blocksize_sort, smem_bytes, stream>>>(
        node_ilist_r2, node_ilist_ioth, node_ilist.spl, size_nodes, smem_size
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
    const int32_t* spl,
    const float* key,
    const int32_t* val,
    float* key_out,
    int32_t* val_out,
    const int32_t size_segs,
    const int32_t size_keys,
    const size_t smem_size
) {
    int blocksize = 64;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));

    cudaMemcpyAsync(key_out, key, size_keys*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(val_out, val, size_keys*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    segmented_bitonic_sort_kv<<< size_segs, blocksize, smem_bytes, stream>>>(
        key_out, val_out, spl, size_segs, smem_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

#endif // KNN_H