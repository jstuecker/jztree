#ifndef CUSTOM_JAX_FMM_H
#define CUSTOM_JAX_FMM_H

#include "multipoles.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                   Evaluate Tree Plane Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */

#define NCOMB(p) (((p) + 1) * ((p) + 2) * ((p) + 3) / 6)

struct NodeInfo {
    float3 center;
    int level;
};

struct NodeWithExt {
    float3 center;
    float3 extent;
};

__device__ __forceinline__ float3 float3sum(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 float3diff(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float norm2(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ __forceinline__ float3 LvlToExt(int level) {
    // Converts a node's or leaf's binary level to its extend per dimension

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 3000) / 3 - 1000;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2);
    int lz = olvl + (omod >= 1);
    
    return make_float3(ldexpf(1.0f, lx), ldexpf(1.0f, ly), ldexpf(1.0f, lz));
}

__device__ __forceinline__ bool OpeningCriterion(
    NodeWithExt nodeA,
    NodeWithExt nodeB,
    float opening_angle
) {
    float r2 = norm2(float3diff(nodeA.center, nodeB.center));
    float L2 = norm2(float3sum(nodeA.extent, nodeB.extent));

    bool need_open = L2 > opening_angle * opening_angle * r2;
    // also open if L2 had an overflow (and r2 is valid)
    need_open = need_open || ((isnan(L2) || isinf(L2)) && !isnan(r2));
    return need_open;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(__activemask(), v, offset);
    return v;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum_shared(T v, T* smem) {
    T val = warp_reduce_sum(v);
    // if((threadIdx.x & 0x1f) == 0)
        atomicAdd(smem, v);
        // smem[0] += val;
    return val;
}

#define BLOCKSIZE 32
#define MAX_NUMA 16

#define ALLTHREADS 0xFFFFFFFF

template<int p>
__global__ void CountInteractionsAndM2L(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const NodeInfo* children,
    const float* mp_values,
    // outputs:
    float* loc_out,
    int* ilist_child_count_out,
    // attributes:
    float softening,
    float opening_angle
) {
    constexpr int ncomb = NCOMB(p);

    // Node A info:
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 child_range = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    // This loop should handle almost always everything on the first pass
    // However, to deal with edge cases we have to loop over scenarios where 
    // the node has more than MAX_NUMA children
    for(int offsetA=child_range.x; offsetA < child_range.y; offsetA += MAX_NUMA) {
        int num_childrenA = min(MAX_NUMA, child_range.y - offsetA);

        // childA info
        __shared__ NodeWithExt childA[MAX_NUMA];
        if(threadIdx.x < num_childrenA) {
            NodeInfo child = children[offsetA + threadIdx.x];
            childA[threadIdx.x] = {child.center, LvlToExt(child.level)};
        }

        int num_open[MAX_NUMA];
        #pragma unroll
        for(int i = 0; i < MAX_NUMA; i++) {
            num_open[i] = 0;
        }

        float LocA[ncomb];
        #pragma unroll
        for(int i = 0; i < ncomb; i++) {
            LocA[i] = 0.0f;
        }

        // Child B info. This is transposed to reduce smem bank conflicts
        __shared__ float3 posB[BLOCKSIZE];
        __shared__ float mpB[ncomb][BLOCKSIZE];
        
        // Interaction list info:
        int2 ilist_range = {spl_ilist[nodeid], spl_ilist[nodeid + 1]};
        __syncthreads();

        __shared__ int2 segments[BLOCKSIZE];
        SegmentManager<BLOCKSIZE> seg_mgr(
            ilist_nodes,
            spl_nodes,
            segments,
            ilist_range.x,
            ilist_range.y
        );


        // Precalculate layout for M2L interactions
        // Which childA am I writing to:
        int a_write = threadIdx.x % num_childrenA;   
        // Where I would read from in the first iteration:
        int read_b_offset = threadIdx.x / num_childrenA;
        // Number of threads that aren't evenly distributed:
        int residual_threads = blockDim.x % num_childrenA; 
        // Number of threads that write to the same childA as me:
        int n_write_a = blockDim.x / num_childrenA + (a_write < residual_threads);
        float3 xaWrite = childA[a_write].center;

        while(!seg_mgr.finished()) {
            int2 id = seg_mgr.next();

            // Each thread loads one other child B to check the opening criterion
            NodeWithExt childB_ext;
            if(id.x >= 0) {
                NodeInfo childB = children[id.x];
                childB_ext = {childB.center, LvlToExt(childB.level)};
            }

            // For each child A, we count the cumulative number of opens and we 
            // flag the M2L interactions that need to be evaluated now

            unsigned int interact_flags_wa = 0;

            bool any_interacts = false;
            #pragma unroll
            for(int i = 0; i < MAX_NUMA; i++) {
                if(i >= num_childrenA)
                    continue;

                bool need_open = OpeningCriterion(childA[i], childB_ext, opening_angle);
                bool actually_open = need_open && (id.x >= 0);
                bool interact_now = !need_open && (id.x >= 0);
                any_interacts = any_interacts || interact_now;

                // Sum over all threads
                num_open[i] += __popc(__ballot_sync(ALLTHREADS, actually_open));
                // Flag the active m2l interactions for this child
                unsigned int interact_flags = __ballot_sync(ALLTHREADS, interact_now);
                // we only store the flag for the child that we need to write to later
                interact_flags_wa = (i == a_write) ? interact_flags : interact_flags_wa;
            }

            __syncthreads();

            // only read the multipoles if at least one interaction happens with this childB
            if(any_interacts) {
                posB[threadIdx.x] = childB_ext.center;

                // Note: This read would probably be more efficient if we coalesced the loads better
                // or maybe if we transposed the multipole layout in advance:
                for(int k=0; k<ncomb; k++) {
                    mpB[k][threadIdx.x] = mp_values[id.x * ncomb + k];
                }
            }

            __syncthreads();

            // Now we have all the data we need for the M2L interactions in shared memory.
            // To avoid reduction operations across threads, we transpose the problem differently here.

            int ninteractionsB_withA = __popc(interact_flags_wa);
            for(int ib=read_b_offset; ib < ninteractionsB_withA; ib += n_write_a) {
                // have to add the m2l interactions between a_write and the ib-th set bit in 
                // interact_flags_wa

                // find the ib-th set bit
                int b_read = __fns(interact_flags_wa, 0, ib+1);

                if(b_read > 32) {
                    // This should not be possible to happen.
                    // To be sure about this, for now invalidate multipoles
                    // later I can delete this part of the code
                    #pragma unroll
                    for(int k = 0; k < ncomb; k++) {
                        LocA[k] = NAN;
                    }

                    continue;
                }

                float mp[ncomb];
                #pragma unroll
                for(int k = 0; k < ncomb; k++) {
                    mp[k] = mpB[k][b_read];
                }

                float3 dx = float3diff(posB[b_read], xaWrite);

                m2l_translator<p>(dx, mp, LocA, softening*softening);
            }
        }

        __syncthreads();
        if(threadIdx.x == 0) { 
            // Since num_open lives in registers, the simplest way of writing it is with a single thread
            #pragma unroll
            for(int i = 0; i < MAX_NUMA; i++) {
                if(offsetA + i < child_range.y) {
                    ilist_child_count_out[offsetA + i] = num_open[i];
                }
            }
        }

        #pragma unroll
        for(int i = 0; i < ncomb; i++) {
            // now atomically add the results
            // note: we can easily avoid the atomic here -- change that later!
            int iout = (offsetA + a_write) * ncomb + i;
            atomicAdd(&loc_out[iout], LocA[i]);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int nbits_set_before(unsigned mask, int bit)
{
    unsigned lower_mask = (bit == 0) ? 0u : ((1u << bit) - 1u);
    return __popc(mask & lower_mask);
}

__global__ void InsertInteractions(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const NodeInfo* children,
    const int* spl_ilist_child,
    // outputs:
    int* child_ilist_out,
    // attributes:
    float opening_angle
) {
    // Node A info:
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 child_range = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    // This loop should handle almost always everything on the first pass
    // However, to deal with edge cases we have to loop over scenarios where 
    // the node has more than MAX_NUMA children
    for(int offsetA=child_range.x; offsetA < child_range.y; offsetA += MAX_NUMA) {
        int num_childrenA = min(MAX_NUMA, child_range.y - offsetA);

        // childA info
        __shared__ NodeWithExt childA[MAX_NUMA];
        __shared__ int ilist_offsets[MAX_NUMA];
        if(threadIdx.x < num_childrenA) {
            NodeInfo child = children[offsetA + threadIdx.x];
            childA[threadIdx.x] = {child.center, LvlToExt(child.level)};
            ilist_offsets[threadIdx.x] = spl_ilist_child[offsetA + threadIdx.x];
        }

        int num_open[MAX_NUMA];
        #pragma unroll
        for(int i = 0; i < MAX_NUMA; i++) {
            num_open[i] = 0;
        }

        // Interaction list info:
        int2 ilist_range = {spl_ilist[nodeid], spl_ilist[nodeid + 1]};
        __syncthreads();

        __shared__ int2 segments[BLOCKSIZE];
        SegmentManager<BLOCKSIZE> seg_mgr(
            ilist_nodes,
            spl_nodes,
            segments,
            ilist_range.x,
            ilist_range.y
        );

        while(!seg_mgr.finished()) {
            int2 id = seg_mgr.next();

            // Each thread loads one other child B to check the opening criterion
            NodeWithExt childB_ext;
            if(id.x >= 0) {
                NodeInfo childB = children[id.x];
                childB_ext = {childB.center, LvlToExt(childB.level)};
            }

            #pragma unroll
            for(int i = 0; i < MAX_NUMA; i++) {
                if(i >= num_childrenA)
                    continue;

                bool need_open = OpeningCriterion(childA[i], childB_ext, opening_angle);
                need_open = need_open && (id.x >= 0);

                unsigned open_mask = __ballot_sync(ALLTHREADS, need_open);

                // Count the number of activated bits before our thread's bit
                int warp_offset = nbits_set_before(open_mask, threadIdx.x & 0x1f);

                if(need_open)
                    child_ilist_out[ilist_offsets[i] + num_open[i] + warp_offset] = id.x;
                
                num_open[i] += __popc(open_mask);
            }
        }
    }
}


/* ---------------------------------------------------------------------------------------------- */
/*                                        New force kernel                                        */
/* ---------------------------------------------------------------------------------------------- */

struct PMass {
    float3 pos;
    float mass;
};

struct ForceAndPot {
    float3 force;
    float  pot;
};

__forceinline__ __device__ void accumulateForceAndPot(PMass xmi, PMass xmj, float softening2, ForceAndPot &fphi_i) {
    float3 dx = float3diff(xmj.pos, xmi.pos);
    float rinv = rsqrtf(norm2(dx) + softening2);

    float minvr = xmj.mass * rinv;
    float minvr3 = minvr * rinv * rinv;

    fphi_i.force.x += minvr3 * dx.x;
    fphi_i.force.y += minvr3 * dx.y;
    fphi_i.force.z += minvr3 * dx.z;
    fphi_i.pot += -minvr;
}

__global__ void NewForceAndPot(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const PMass* posm,
    // outputs:
    ForceAndPot* fphi,
    // attributes:
    float softening,
    int max_leaf_size
) {
    float softening2 = softening * softening;

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;
    
    int2 prange = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    int num = min(prange.y-prange.x,  min(max_leaf_size, blockDim.x));

    // extern __shared__ PMass xm_a[];
    // PMass* xm_b = &xm_a[blockDim.x];
    __shared__ PMass xm_a[BLOCKSIZE];
    __shared__ PMass xm_b[BLOCKSIZE];

    if(threadIdx.x < num) {
        xm_a[threadIdx.x] = posm[prange.x + threadIdx.x];
    }
    else {
        xm_a[threadIdx.x] = {{0.f,0.f,0.f}, 0.f};
    }
    __syncthreads();

    // int2* segments = (int2*)&xm_b[blockDim.x];
    __shared__ int2 segments[BLOCKSIZE];
    SegmentManager<BLOCKSIZE> seg_mgr( // Todo: make this not require a template variable
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1]
    );

    ForceAndPot fphi_a = {{0.f,0.f,0.f}, 0.f};

    while(!seg_mgr.finished()) {
        int2 id = seg_mgr.next();

        __syncthreads();

        // Each thread loads one other particle B
        if(id.x >= 0) {
            xm_b[threadIdx.x] = posm[id.x];
        }
        __syncthreads();

        // Now compute interactions
        for(int j=0; j<seg_mgr.num_loaded; j++) {
            accumulateForceAndPot(
                xm_a[threadIdx.x],
                xm_b[j],
                softening2,
                fphi_a
            );
        }
    }

    if(threadIdx.x < num) {
        fphi[prange.x + threadIdx.x] = fphi_a;
    }
}

#endif