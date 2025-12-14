#ifndef KNN_H
#define KNN_H

#include <cub/cub.cuh>
#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"
#include "common/knn_math.cuh"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

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

#endif // KNN_H