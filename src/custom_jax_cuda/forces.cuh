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
__global__ void ForceAndPotential(
    const PMass *xm,
    ForcePot *fphi,
    int n,
    float epsilon
) {
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

template <bool kahan>
__global__ void BwdForceAndPotential(
    const float4 *gfphi,
    const PMass *xm,
    float4 *gxm,
    int n,
    float epsilon
) {
    // const int steps = div_ceil(n, blockDim.x);
    // float epsilon2 = epsilon * epsilon;

    // PMass xmi = xm[blockIdx.x * blockDim.x + threadIdx.x];

    // extern __shared__ PMass xmj_shared[];

    // ForcePot fphi_i = {0.f, 0.f, 0.f, 0.f};
    // ForcePot fphi_kahan = {0.f, 0.f, 0.f, 0.f};

    // for (int jblock = 0; jblock < steps; jblock += 1) {
    //     int num = min(blockDim.x, n - blockDim.x * jblock);

    //     __syncthreads();
    //     if(threadIdx.x < num)
    //         xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
    //     __syncthreads();

    //     for (int j = 0; j < num; j++) {
    //         if(kahan)
    //             accumulateForceAndPotKahan(xmi, xmj_shared[j], epsilon2, fphi_i, fphi_kahan);
    //         else
    //             accumulateForceAndPot(xmi, xmj_shared[j], epsilon2, fphi_i);
    //     }
    // }

    // fphi_i.pot += xmi.mass / epsilon; // remove self-interaction from potential

    // fphi[blockIdx.x * blockDim.x + threadIdx.x] = fphi_i;
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





/* ---------------------------------------------------------------------------------------------- */
/*                                            Old code                                            */
/* ---------------------------------------------------------------------------------------------- */




/* ---------------------------------------------------------------------------------------------- */
/*                                           Ilist Force                                          */
/* ---------------------------------------------------------------------------------------------- */

__device__ inline void atomicAddFloat4(float4* addr, const float4 val) {
    atomicAdd(&addr->x, val.x);
    atomicAdd(&addr->y, val.y);
    atomicAdd(&addr->z, val.z);
    atomicAdd(&addr->w, val.w);
}

__global__ void IlistForceAndPot(
    const PMass *xm,
    const int32_t *isplit,
    const int2 *interactions,
    const int *iminmax,
    ForcePot *fphi,
    size_t interactions_per_block,
    float epsilon
) {
    const int blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    extern __shared__ PMass xmj_shared[];

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + blockIdx.x * interactions_per_block + iint;
        if (int_id >= imax) {
            return;
        }

        int2 interaction = interactions[int_id];
        int iAstart = isplit[interaction.x], iAend = isplit[interaction.x + 1];
        int iBstart = isplit[interaction.y], iBend = isplit[interaction.y + 1];

        /* We have to have interactions between all particles of x[IAstart:IAend] with x[IBstart:IBend]*/
        /* These may be larger than our warp's blocksize so that we have to loop individual parts of A and B */
        int nA = iAend - iAstart, nB = iBend - iBstart;
        for (int i = 0; i < nA; i += blocksize) {
            ForcePot fphi_i = {0.f, 0.f, 0.f, 0.f};
            PMass xmi = {0.f, 0.f, 0.f, 0.f};
            int ioffA = i + threadIdx.x;

            if(ioffA  < nA) {
                xmi = xm[iAstart + ioffA];
            }

            for (int j = 0; j < nB; j += blocksize) {
                int joffB = j + threadIdx.x;
                if(joffB < nB) {
                    xmj_shared[threadIdx.x] = xm[iBstart + joffB];
                }
                __syncthreads();

                for (int k = 0; k < min(nB - j, blocksize); k++) {
                    accumulateForceAndPot(xmi, xmj_shared[k], epsilon2, fphi_i);
                }
                __syncthreads();
            }

            if(ioffA < nA) {
                // atomicAddFloat4(&fphi[iAstart + ioffA], fphi_i);
                atomicAdd(&fphi[iAstart + ioffA].force.x, fphi_i.force.x);
                atomicAdd(&fphi[iAstart + ioffA].force.y, fphi_i.force.y);
                atomicAdd(&fphi[iAstart + ioffA].force.z, fphi_i.force.z);
                atomicAdd(&fphi[iAstart + ioffA].pot, fphi_i.pot);    
            }
        }
    }
}

__forceinline__ __device__ void accumulateGradients(float4 xmi, float4 xmj, float4 gi, float4 gj, float epsilon2, float4 &gxmi) {
    // calculates the vector jacobian product of the interaction between xmi and xmj
    // gi and gj are the final gradient vectors of fphi_i and fphi_j, respectively.
    // we have to back propagate the gradient towards a gradient with respect to xmi
    // for understanding the maths, please consider the corresponding .ipynb notebook
    float dx = xmj.x - xmi.x;
    float dy = xmj.y - xmi.y;
    float dz = xmj.z - xmi.z;
    float mi = xmi.w;
    float mj = xmj.w;

    float r2 = dx*dx + dy*dy + dz*dz;
    // Mask self-interaction. (It would have a very large contribution here, so subtracting it later is numerically bad.):
    float rinv = r2 > 1e-10f ? rsqrtf(r2 + epsilon2) : 0.f; // 1e-30
    float rinv2 = rinv*rinv;

    float f1 = -rinv*rinv2;
    float f2 = 3*rinv*rinv2*rinv2;

    float3 gm_diff = {gi.x*mj - gj.x*mi, gi.y*mj - gj.y*mi, gi.z*mj - gj.z*mi};
    float fgdiff = f2*(gm_diff.x * dx + gm_diff.y * dy + gm_diff.z * dz);
    // This line handles the potential gradient:
    fgdiff += f1 * (gi.w * mj + gj.w * mi);

    gxmi.x += f1 * gm_diff.x + dx * fgdiff;
    gxmi.y += f1 * gm_diff.y + dy * fgdiff;
    gxmi.z += f1 * gm_diff.z + dz * fgdiff;

    float dx_dot_gj = dx * gj.x + dy * gj.y + dz * gj.z;
    gxmi.w += f1 * dx_dot_gj - rinv * gj.w; // first part for force, second for potential
}

__global__ void BwdIlistForceAndPot(
    const float4 *gfphi,
    const float4 *xm,
    const int32_t *isplit,
    const int2 *interactions,
    const int *iminmax,
    float4 *gxm,
    size_t interactions_per_block,
    float epsilon
) {
    const int blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    extern __shared__ float4 shared_memory[];

    float4* xmj_shared     = shared_memory;
    float4* gfphi_j_shared = &shared_memory[blocksize];

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + blockIdx.x * interactions_per_block + iint;
        if (int_id >= imax) {
            return;
        }

        int2 interaction = interactions[int_id];
        int iAstart = isplit[interaction.x], iAend = isplit[interaction.x + 1];
        int iBstart = isplit[interaction.y], iBend = isplit[interaction.y + 1];

        /* We have to have interactions between all particles of x[IAstart:IAend] with x[IBstart:IBend]*/
        /* These may be larger than our warp's blocksize so that we have to loop individual parts of A and B */
        int nA = iAend - iAstart, nB = iBend - iBstart;
        for (int i = 0; i < nA; i += blocksize) {
            float4 xmi = {0.f, 0.f, 0.f, 0.f};
            float4 gxm_i = {0.f, 0.f, 0.f, 0.f};
            float4 gfphi_i = {0.f, 0.f, 0.f, 0.f};
            int ioffA = i + threadIdx.x;

            if(ioffA  < nA) {
                xmi = xm[iAstart + ioffA];
                gfphi_i = gfphi[iAstart + ioffA];
            }

            for (int j = 0; j < nB; j += blocksize) {
                int joffB = j + threadIdx.x;
                if(joffB < nB) {
                    xmj_shared[threadIdx.x] = xm[iBstart + joffB];
                    gfphi_j_shared[threadIdx.x] = gfphi[iBstart + joffB];
                }
                __syncthreads();

                for (int k = 0; k < min(nB - j, blocksize); k++) {
                    accumulateGradients(xmi, xmj_shared[k], gfphi_i, gfphi_j_shared[k], epsilon2, gxm_i);
                }
                __syncthreads();
            }

            if(ioffA < nA) {
                atomicAddFloat4(&gxm[iAstart + ioffA], gxm_i);
            }
        }
    }
}