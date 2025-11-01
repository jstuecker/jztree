#ifndef FORCES_CUH
#define FORCES_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

__forceinline__ __device__ void accumulateForceAndPot(
    PMass xmi, 
    PMass xmj, 
    float softening2, 
    ForcePot &fphi_i
) {
    float3 dx = float3diff(xmj.pos, xmi.pos);
    float rinv = rsqrtf(norm2(dx) + softening2);

    float minvr = xmj.mass * rinv;
    float minvr3 = minvr * rinv * rinv;

    fphi_i.force.x += minvr3 * dx.x;
    fphi_i.force.y += minvr3 * dx.y;
    fphi_i.force.z += minvr3 * dx.z;
    fphi_i.pot += -minvr;
}

__forceinline__ __device__ void kahan_add(float &sum, float add, float &c) {
    // Cancels summation error with an extra variable c, that needs to start at 0
    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    float y = add - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

__forceinline__ __device__ void accumulateForceAndPotKahan(
    PMass xmi, 
    PMass xmj, 
    float softening2, 
    ForcePot &fphi_i,
    ForcePot &fphi_kahan
) {
    float3 dx = float3diff(xmj.pos, xmi.pos);
    float rinv = rsqrtf(norm2(dx) + softening2);

    float minvr = xmj.mass * rinv;
    float minvr3 = minvr * rinv * rinv;
    
    kahan_add(fphi_i.force.x, minvr3 * dx.x, fphi_kahan.force.x);
    kahan_add(fphi_i.force.y, minvr3 * dx.y, fphi_kahan.force.y);
    kahan_add(fphi_i.force.z, minvr3 * dx.z, fphi_kahan.force.z);
    kahan_add(fphi_i.pot, -minvr, fphi_kahan.pot);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Simple Force Kernel                                      */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan>
__global__ void ForceAndPotential(const PMass *xm, ForcePot *fphi, int n, float epsilon) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    PMass xmi = xm[blockIdx.x * blockDim.x + threadIdx.x];

    extern __shared__ PMass xmj_shared[];

    ForcePot fphi_i = {0.f, 0.f, 0.f, 0.f};
    ForcePot fphi_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            if(kahan)
                accumulateForceAndPotKahan(xmi, xmj_shared[j], epsilon2, fphi_i, fphi_kahan);
            else
                accumulateForceAndPot(xmi, xmj_shared[j], epsilon2, fphi_i);
        }
    }

    fphi_i.pot += xmi.mass / epsilon; // remove self-interaction from potential

    fphi[blockIdx.x * blockDim.x + threadIdx.x] = fphi_i;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Grouped force kernel                                       */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan>
__global__ void GroupedForceAndPot(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const PMass* posm,
    // outputs:
    ForcePot* fphi,
    // attributes:
    float softening,
    int max_leaf_size
) {
    // Note: I should try to add Khan summation option
    //       I think the numerical error might be dominated by the summation in this kernel

    float softening2 = softening * softening;

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;
    
    int2 prange = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    int num = prange.y-prange.x;

    // Precalculate layout for M2L interactions: blockdim.x -> (num, n_write) + residuals
    int n_write = blockDim.x / num;
    // Which particle I am writing to:
    int a_write = threadIdx.x % num;   
    // Where I would read from in the first iteration:
    int read_b_offset = threadIdx.x / num;
    // Flag residual threads as invalid
    // (To avoid divergence we let these follow allong the calculations, 
    //  but later discard their result)
    int valid = threadIdx.x < num * n_write;

    PMass xaWrite = posm[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    ForcePot fphi_a = {{0.f,0.f,0.f}, 0.f};
    ForcePot fphi_a_kahan = {{0.f,0.f,0.f}, 0.f};

    extern __shared__ PMass xm_b[];

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        // Each thread loads one other particle B
        if(id >= 0) {
            xm_b[threadIdx.x] = posm[id];
        }
        __syncthreads();

        // Now compute interactions
        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            if(kahan)
                accumulateForceAndPotKahan(xaWrite, xm_b[ib], softening2, fphi_a, fphi_a_kahan);
            else
                accumulateForceAndPot(xaWrite, xm_b[ib], softening2, fphi_a);
        }

        __syncthreads();
    }

    if(valid) {
        int iout = prange.x + a_write;
        atomicAdd(&fphi[iout].force.x, fphi_a.force.x);
        atomicAdd(&fphi[iout].force.y, fphi_a.force.y);
        atomicAdd(&fphi[iout].force.z, fphi_a.force.z);
        atomicAdd(&fphi[iout].pot, fphi_a.pot);    
    }
}

#endif