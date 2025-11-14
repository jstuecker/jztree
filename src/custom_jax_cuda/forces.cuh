#ifndef FORCES_CUH
#define FORCES_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

__forceinline__ __device__ ForcePot GetForceAndPot(
    PosMass xmi, 
    PosMass xmj, 
    float softening2
) {
    float3 dx = xmj.pos - xmi.pos;
    float rinv = rsqrtf(norm2(dx) + softening2);
    float minvr = xmj.mass * rinv;

    ForcePot fphi = {
        (minvr * rinv * rinv) * dx,
        -minvr
    };

    return fphi;
}

__forceinline__ __device__ PosMass VJP_GFPhiToGXM(
    const PosMass xmi, const PosMass xmj, 
    const ForcePot gi, const ForcePot gj, 
    float epsilon2
) {
    // calculates the vector jacobian product of the interaction between xmi and xmj
    // gi and gj are the final gradient vectors of fphi_i and fphi_j, respectively.
    // we have to back propagate the gradient towards a gradient with respect to xmi
    // for understanding the maths, please consider the corresponding .ipynb notebook
    float3 dx = xmj.pos - xmi.pos;
    float r2 = norm2(dx);
    float rinv = r2 > 1e-10f * epsilon2 ? rsqrtf(r2 + epsilon2) : 0.f;
    float rinv2 = rinv*rinv;

    float f1 = -rinv*rinv2;
    float f2 = 3*rinv*rinv2*rinv2;

    float3 gm_diff = xmj.mass * gi.force - xmi.mass * gj.force;
    float fgdiff = f2*dot(gm_diff, dx) + f1 * (gi.pot * xmj.mass + gj.pot * xmi.mass);

    PosMass gxmi;

    gxmi.pos = f1 * gm_diff + fgdiff * dx;
    gxmi.mass = f1 * dot(dx, gj.force) - rinv * gj.pot;

    return gxmi;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Simple Force Kernel                                      */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan>
__global__ void ForceAndPotential(
    const PosMass *xm,
    ForcePot *fphi,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    PosMass xmi;
    if(ipart < n)
        xmi = xm[ipart];

    extern __shared__ PosMass xmj_shared[];

    ForcePot fphi_i = {0.f, 0.f, 0.f, 0.f};
    ForcePot fphi_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            ForcePot fphi_new = GetForceAndPot(xmi, xmj_shared[j], epsilon2);
            if(kahan)
                kahan_add_f4(fphi_i.f4, fphi_new.f4, fphi_kahan.f4);
            else
                fphi_i.f4 = fphi_i.f4 + fphi_new.f4;
        }
    }

    fphi_i.pot += xmi.mass / epsilon; // remove self-interaction from potential

    if(ipart < n)
        fphi[blockIdx.x * blockDim.x + threadIdx.x] = fphi_i;
}

template <bool kahan>
__global__ void BwdForceAndPotential(
    const ForcePot *gfphi,
    const PosMass *xm,
    PosMass *gxm,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    PosMass xmi;
    ForcePot gfphi_i;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    if(ipart < n) {
        xmi = xm[ipart];
        gfphi_i = gfphi[ipart];
    }

    extern __shared__ PosMass xmj_shared[];
    ForcePot* gfphi_j_shared = (ForcePot*) &xmj_shared[blockDim.x];

    PosMass gxm_i = {0.f, 0.f, 0.f, 0.f};
    PosMass gxm_i_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num) {
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
            gfphi_j_shared[threadIdx.x] = gfphi[jblock * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        for (int j = 0; j < num; j++) {
            PosMass gxm_inc = VJP_GFPhiToGXM(
                xmi, xmj_shared[j],
                gfphi_i, gfphi_j_shared[j],
                epsilon2
            );
            if(kahan)
                kahan_add_f4(gxm_i.f4, gxm_inc.f4, gxm_i_kahan.f4);
            else
                gxm_i.f4 = gxm_i.f4 + gxm_inc.f4;
        }
    }

    if(ipart < n)
        gxm[ipart] = gxm_i;
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
    const PosMass* posm,
    // outputs:
    ForcePot* fphi,
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

    PosMass xaWrite = posm[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    ForcePot fphi_a = {0.f,0.f,0.f,0.f};
    ForcePot fphi_a_kahan = {0.f,0.f,0.f,0.f};

    extern __shared__ PosMass xm_b[];

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        // Each thread loads one other particle B
        if(id >= 0) {
            xm_b[threadIdx.x] = posm[id];
        }
        __syncthreads();

        // Now compute interactions
        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            ForcePot fphi_new = GetForceAndPot(xaWrite, xm_b[ib], softening2);
            if(kahan)
                kahan_add_f4(fphi_a.f4, fphi_new.f4, fphi_a_kahan.f4);
            else
                fphi_a.f4 = fphi_a.f4 + fphi_new.f4;
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
    const PosMass *xm,
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

    extern __shared__ PosMass xmj_shared[];

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
            PosMass xmi = {0.f, 0.f, 0.f, 0.f};
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
                    fphi_i.f4 = fphi_i.f4 + GetForceAndPot(xmi, xmj_shared[k], epsilon2).f4;
                }
                __syncthreads();
            }

            if(ioffA < nA) {    
                atomicAddFloat4(&fphi[iAstart + ioffA].f4, fphi_i.f4);
            }
        }
    }
}