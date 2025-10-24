#include <type_traits>
#include <stdexcept>
#include "multipoles.h"
#include <math_constants.h> // CUDART_NAN_F, CUDART_NAN

// =============================================================
// Multipole Translators (CUDA-only)
// =============================================================

#define MAXP 6

// Helper macro to dispatch a templated kernel on runtime integer p (1..6).
// Usage: LAUNCH_KERNEL_SWITCH(p, KernelName, grid_size, block_size, stream, arg1, arg2, ...)
#define LAUNCH_KERNEL_SWITCH(P, KERNEL, GRID, BLOCK, STREAM, ...) \
    switch(P) { \
        case 1: KERNEL<1><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 2: KERNEL<2><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 3: KERNEL<3><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        /*case 4: KERNEL<4><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 5: KERNEL<5><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 6: KERNEL<6><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;*/ \
        default: break; \
    }

template<int p>
__device__ void setupGn(float r2, float eps2, float* __restrict__ G)
{
    // The derivatives of (1/r d/dr)^n G_0  with G_0 = 1/r
    float rinv = rsqrtf(r2 + eps2); 
    float rinv2 = rinv*rinv;
    G[0] = 1.0f * rinv;

    #pragma unroll
    for (int n = 1; n <= p; n++) {
        G[n] = -(2*n-1) * G[n-1] * rinv2;
    }
}

#define NCOMB(p) (((p) + 1) * ((p) + 2) * ((p) + 3) / 6)

__device__ __forceinline__ float get_xk(const float3& x, int k) {
    return (k == 0) ? x.x : (k == 1 ? x.y : x.z);
}

__host__ __device__ __forceinline__
float powi_upto6(float x, int n) {

    switch (n) {
        case 0: return 1.0f;
        case 1: return x;
        case 2: return x * x;
        case 3: { float x2 = x * x; return x2 * x; }
        case 4: { float x2 = x * x; return x2 * x2; }
        case 5: { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
        case 6: { float x2 = x * x; float x3 = x2 * x; return x3 * x3; }
        default: // fallback if someone passes >6
            return powf(x, (float)n);
    }
}

__device__ __forceinline__ float fact_upto6f(unsigned k) {
    float f = 1.f;
    f *= (k >= 2) ? 2.f : 1.f;
    f *= (k >= 3) ? 3.f : 1.f;
    f *= (k >= 4) ? 4.f : 1.f;
    f *= (k >= 5) ? 5.f : 1.f;
    f *= (k >= 6) ? 6.f : 1.f;
    return f;
}

__device__ __forceinline__ float binomial(unsigned n, unsigned k) {
    if (k > n) return 0.f;
    float res = 1.f;
    for (unsigned i = 1; i <= k; i++) {
        res *= float(n - (k - i)) / float(i);
    }
    return res;
}

__device__ __forceinline__ float fact3f(unsigned kx, unsigned ky, unsigned kz) {
    return fact_upto6f(kx) * fact_upto6f(ky) * fact_upto6f(kz);
}

__device__ __forceinline__ constexpr  int multi_to_flat(const int kx, const int ky, const int kz) {
    int p = kx + ky + kz;
    int npoff = ((p+2)*(p+1)*p) / 6; // offset of the p-th symmeric tensor
    int off = npoff + (kz*(2*p + 3 - kz))/2 + ky;

    return off > 0 ? off : 0; // Ensure we don't return negative indices
}

template<int pmax>
__device__ __forceinline__ constexpr  int3 flat_to_multi(const int kflat) {
    int i = 0, ksum, kz, ky;
    #pragma unroll
    for(ksum=0; ksum <= pmax; ksum++) {
        int nadd = ((ksum+2)*(ksum+1)) >> 1;
        if (i + nadd > kflat)
            break;
        i += nadd;
    }
    #pragma unroll
    for(kz=0; kz <= ksum; kz++) {
        int nadd = (ksum-kz+1);
        if (i + nadd > kflat)
            break;
        i += nadd;
    }
    ky = kflat - i;

    return int3{ksum-ky-kz, ky, kz};
}


template<int p>
__device__ void setupDnG(float3 dx, float eps2, float* __restrict__ Dn) {
    float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

    float G[p+1];
    setupGn<p>(r2, eps2, G);
    Dn[0] = G[p];

    #pragma unroll
    for(int q=p-1; q >= 0; q--) {
        int iflat = NCOMB(p-q)-1;
        #pragma unroll
        for(int nsum=p-q; nsum >= 1; nsum--) {
            #pragma unroll
            for(int nz=nsum; nz >= 0; nz--) {
                #pragma unroll
                for(int ny=nsum-nz; ny >= 0; ny--) {
                    const int nx = nsum - ny - nz;

                    const int k = nz > 0 ? 2 : (ny > 0 ? 1 : 0);
                    const int nk = nz > 0 ? nz : (ny > 0 ? ny : nx);
                    const float xk = k == 0 ? dx.x : (k == 1 ? dx.y : dx.z);

                    const int ilast = multi_to_flat(nx - 1*(k==0), ny - 1*(k==1), nz - 1*(k==2));
                    const int ilast2 = multi_to_flat(nx - 2*(k==0), ny - 2*(k==1), nz - 2*(k==2));

                    Dn[iflat] = xk*Dn[ilast] + (nk-1)*Dn[ilast2];
                    iflat -= 1;
                }
            }
        }

        Dn[0] = G[q];
    }
}


template<int p>
__global__ void IlistM2LKernel(const float3* __restrict__ x, const float* __restrict__ mp, 
        const int2* __restrict__ interactions, const int* __restrict__ iminmax, 
        float* __restrict__ Lout, size_t interactions_per_block, float epsilon) {
    const size_t block_size = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    constexpr int ncomb = NCOMB(p);

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + block_size * (blockIdx.x * interactions_per_block + iint) + threadIdx.x;
        if (int_id >= imax) {
            return;
        }

        int iA = interactions[int_id].x, iB = interactions[int_id].y;
        float3 xA = x[iA], xB = x[iB];
        float3 dx = {xB.x - xA.x, xB.y - xA.y, xB.z - xA.z};
        float Dn[ncomb];
        setupDnG<p>(dx, epsilon2, Dn);

        float Mp[ncomb];
        #pragma unroll
        for (int iM=0; iM < ncomb; iM++) {
            Mp[iM] = mp[iB * ncomb + iM];
        }

        int kflat = 0;
        #pragma unroll
        for(int ksum = 0; ksum <= p; ksum++) {
            #pragma unroll
            for(int kz = 0; kz <= ksum; kz++) {
                #pragma unroll
                for(int ky = 0; ky <= ksum - kz; ky++) {
                    const int kx = ksum - ky - kz;
                    float Lnew = 0.f;

                    int nflat = 0;
                    #pragma unroll
                    for (int nsum = 0; nsum <= p - ksum; nsum++) {
                        #pragma unroll
                        for (int nz = 0; nz <= nsum; nz++) {
                            #pragma unroll
                            for (int ny = 0; ny <= nsum - nz; ny++) {
                                const int nx = nsum - ny - nz;
                                
                                float Dnk = Dn[multi_to_flat(kx + nx, ky + ny, kz + nz)];
                                float Mpn = Mp[nflat];
                                
                                const float infvac = 1./fact3f(nx, ny, nz);
                                Lnew += Dnk * Mpn * infvac;
                                nflat += 1;
                            }
                        }
                    }

                    const float sign = ksum % 2 == 0 ? -1.f : 1.f;
                    const float fac = sign / fact3f(kx, ky, kz);
                    atomicAdd(&Lout[iA * ncomb + kflat], Lnew * fac); 
                    kflat += 1;
                }
            }
        }
    }
}

void launch_IlistM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream, 
        const float3 *x, const float *mp, int2 *interactions, int *iminmax, float *Lout, 
        size_t interactions_per_block, float epsilon) {
    LAUNCH_KERNEL_SWITCH(p, IlistM2LKernel, grid_size, block_size, stream, x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon);
}


__device__ __forceinline__  __device__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

template<int N>
__device__ void add_warp_reduced(const float (&Dn)[N], float* __restrict__ Dsum, bool valid) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        float v = valid ? Dn[k] : 0.f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(&Dsum[k], v);
    }
}

template<int p>
__global__ void IlistLeaf2NodeM2LKernel(const float3* __restrict__  xnodes, 
        const float4* __restrict__ xm, const int32_t* __restrict__ isplit, 
        const int2* __restrict__ interactions, const int* __restrict__ iminmax, 
        float* __restrict__ Lout, size_t interactions_per_block, float epsilon
    )  {
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    constexpr int ncomb = NCOMB(p);
    float Dn[ncomb];

    __shared__ float Dsum[ncomb];
    for (int i = threadIdx.x; i < ncomb; i += blockDim.x) 
        Dsum[i] = 0.0f;
    __syncthreads();

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + iint * gridDim.x + blockIdx.x ;
        if (int_id >= imax) {
            return;
        }

        int2 interaction = interactions[int_id];
        int iNode = interaction.x, iLeaf = interaction.y;
        int iPartStart = isplit[iLeaf], iPartEnd = isplit[iLeaf + 1];

        float3 xnode = xnodes[iNode];

        for(int poffset=0; poffset < iPartEnd - iPartStart; poffset += blockDim.x)
        {
            int ipart = iPartStart + poffset + threadIdx.x;
            bool valid = ipart < iPartEnd;

            float4 xmpart = xm[valid ? ipart : iPartEnd-1];

            float3 dx = {xmpart.x - xnode.x, xmpart.y - xnode.y, xmpart.z - xnode.z};
            setupDnG<p>(dx, epsilon2, Dn);

            add_warp_reduced<ncomb>(Dn, Dsum, valid);
        }

        __syncthreads();

        // Once we are done with all particles we can ouput the results, split over components
        for(int kflat=threadIdx.x; kflat < ncomb; kflat += blockDim.x) {
            int3 k = flat_to_multi<p>(kflat);
            int ksum = k.x + k.y + k.z;
            float sign = (ksum & 1) == 0 ? -1.f : 1.f;
            float Lnew = sign * Dsum[kflat]  / fact3f(k.x, k.y, k.z);
            
            atomicAdd(&Lout[iNode*ncomb + kflat],  Lnew);

            Dsum[kflat] = 0.f;
        }

        __syncthreads();
    }
}

void launch_IlistLeaf2NodeM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream, 
    const float3 *xnodes, const float4 *xm, int32_t *isplit, int2 *interactions, int *iminmax, 
    float *Lout, size_t interactions_per_block, float epsilon) {
    LAUNCH_KERNEL_SWITCH(p, IlistLeaf2NodeM2LKernel, grid_size, block_size, stream, xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon);
}

// PosMass is declared in multipoles.h

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned m = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(m, v, offset);
    return v;
}

__inline__ __device__ void accumulate_m_and_mpos(float* smem, const PosMass& posm, float3 x0) {
    float m = posm.mass;
    float mx = posm.mass * (posm.x - x0.x);
    float my = posm.mass * (posm.y - x0.y);
    float mz = posm.mass * (posm.z - x0.z);

    float msum = warp_reduce_sum(m);
    float mxsum = warp_reduce_sum(mx);
    float mysum = warp_reduce_sum(my);
    float mzsum = warp_reduce_sum(mz);

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&smem[0], msum);
        atomicAdd(&smem[1], mxsum);
        atomicAdd(&smem[2], mysum);
        atomicAdd(&smem[3], mzsum);
    }
}

template<int p>
__global__ void MultipolesFromParticlesKernel(
    const int* __restrict__ isplit,
    const PosMass* __restrict__ part_posm,
    float* __restrict__ mp_out,
    float3* __restrict__ xcom_out
) {
    constexpr int ncomb = NCOMB(p);

    int inode = blockIdx.x;
    int ipart_start = isplit[inode];
    int ipart_end = isplit[inode + 1];

    if (ipart_start >= ipart_end) {
        for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
            mp_out[inode * ncomb + iM] = 0.f;
        }
        if (threadIdx.x == 0) {
            xcom_out[inode] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
        }
        return;
    }

    PosMass p0 = part_posm[ipart_start];
    float3 x0 = {p0.x, p0.y, p0.z};

    __shared__ float mp[ncomb];
    #pragma unroll
    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp[iM] = 0.0f;
    }
    __syncthreads();

    // First pass: compute total mass and center of mass
    for (int ioff = ipart_start; ioff < ipart_end; ioff += blockDim.x) {
        PosMass posm;
        if (ioff + threadIdx.x < ipart_end)
            posm = part_posm[ioff + threadIdx.x];
        else
            posm = PosMass{0.f, 0.f, 0.f, 0.f};
        
        accumulate_m_and_mpos(&mp[0], posm, x0);
    }
    __syncthreads();
    float3 com = { mp[1] / mp[0] + x0.x, mp[2] / mp[0] + x0.y, mp[3] / mp[0] + x0.z};
    __syncthreads();
    if (threadIdx.x == 0) {
        mp[1] = 0.f; mp[2] = 0.f; mp[3] = 0.f;
    }

    // Second pass: compute multipoles around center of mass
    for (int ioff = ipart_start; ioff < ipart_end; ioff += blockDim.x) {
        PosMass dpos;
        if (ioff + threadIdx.x < ipart_end) {
            PosMass posm = part_posm[ioff + threadIdx.x];
            dpos = PosMass{ posm.x - com.x, posm.y - com.y, posm.z - com.z, posm.mass};
        }
        else
            dpos = PosMass{0.f, 0.f, 0.f, 0.f};

        int kflat = -1;
        #pragma unroll
        for(int ksum = 0; ksum <= p; ksum++) {
            #pragma unroll
            for(int kz = 0; kz <= ksum; kz++) {
                #pragma unroll
                for(int ky = 0; ky <= ksum - kz; ky++) {
                    kflat += 1;
                    if (ksum <= 1)
                        continue; // Skip monopole and dipole terms (already computed)
                    const int kx = ksum - ky - kz;

                    float mnew = dpos.mass * powf(dpos.x, kx) * powf(dpos.y, ky) * powf(dpos.z, kz);

                    float msum = warp_reduce_sum(mnew);
                    if ((threadIdx.x & 31) == 0)
                        atomicAdd(&mp[kflat], msum);
                }
            }
        }
    }

    __syncthreads();

    // Write output
    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp_out[inode * ncomb + iM] = mp[iM];
    }
    if (threadIdx.x == 0) {
        xcom_out[inode] = com;
    }
}

template<int p>
__inline__ __device__ void shift_multipoles(float *mp, float *mp_out, float3 dpos) {

    // Shift in x
    int iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;

                float mnew = 0.f;
                #pragma unroll
                for(int i = 0; i <= kx; i++) {
                    float coeff = binomial(kx, i);
                    mnew += coeff * powi_upto6(dpos.x, kx - i) * mp[multi_to_flat(i, ky, kz)];
                }

                mp_out[iflat] = mnew;
                iflat += 1;
            }
        }
    }
        
    // Shift in y
    iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;

                float mnew = 0.f;
                #pragma unroll
                for(int j = 0; j <= ky; j++) {
                    float coeff = binomial(ky, j);
                    mnew += coeff * powi_upto6(dpos.y, ky - j) * mp_out[multi_to_flat(kx, j, kz)];
                }

                mp[iflat] = mnew;
                iflat += 1;
            }
        }
    }

    // Shift in z
    iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;
                float mnew = 0.f;
                #pragma unroll
                for(int l = 0; l <= kz; l++) {
                    float coeff = binomial(kz, l);
                    mnew += coeff * powi_upto6(dpos.z, kz - l) * mp[multi_to_flat(kx, ky, l)];
                }

                mp_out[iflat] = mnew;
                iflat += 1;
            }
        }
    }
}


void launch_MultipolesFromParticlesKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream, 
    const int *isplit, const PosMass *part_posm, float *mp_out, float3 *xcom_out) {
    LAUNCH_KERNEL_SWITCH(p, MultipolesFromParticlesKernel, grid_size, block_size, stream, isplit, part_posm, mp_out, xcom_out);
}

template<int p>
__global__ void CoarsenMultipolesKernel(
    const int* __restrict__ isplit,
    const float* __restrict__ mp_values,
    const float3* __restrict__ xcent,
    float* __restrict__ mp_out,
    float3* __restrict__ xcent_out
) {
    constexpr int ncomb = NCOMB(p);

    int inode = blockIdx.x;
    int istart = isplit[inode];
    int iend = isplit[inode + 1];

    // If there are no children, write zeros and NaN center
    if (istart >= iend) {
        for (int i = threadIdx.x; i < ncomb; i += blockDim.x)
            mp_out[inode * ncomb + i] = 0.f;
        if (threadIdx.x == 0)
            xcent_out[inode] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
        return;
    }

    float3 x0 = xcent[istart];

    __shared__ float mp[ncomb];
    #pragma unroll
    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp[iM] = 0.0f;
    }
    __syncthreads();

    // First pass: compute total mass and center of mass
    for (int ioff = istart; ioff < iend; ioff += blockDim.x) {
        int index = ioff + threadIdx.x;
        PosMass posm;
        if (index < iend)
            posm = PosMass{
                xcent[index].x, xcent[index].y, xcent[index].z,
                mp_values[index * ncomb + 0]
            };
        else
            posm = PosMass{0.f, 0.f, 0.f, 0.f};
        
        accumulate_m_and_mpos(&mp[0], posm, x0);
    }
    __syncthreads();
    float3 com = { mp[1] / mp[0] + x0.x, mp[2] / mp[0] + x0.y, mp[3] / mp[0] + x0.z };
    __syncthreads();
    if (threadIdx.x == 0) {
        mp[1] = 0.f; mp[2] = 0.f; mp[3] = 0.f;
    }
    __syncthreads();

    // Second pass: for each child shift its multipoles to the inode center and accumulate
    for (int ioff = istart; ioff < iend; ioff += blockDim.x) {
        int index = ioff + threadIdx.x;
        // compute displacement from child center to inode com
        // float dx = com.x - xcent[index].x;
        // float dy = com.y - xcent[index].y;
        float3 dpos = make_float3(com.x-xcent[index].x, com.y-xcent[index].y, com.z-xcent[index].z);

        // load source multipoles
        // compute shifted multipoles into a small stack array
        float src[NCOMB(p)];
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            if(index < iend)
                src[i] = mp_values[index * ncomb + i];
            else
                src[i] = 0.f;
        }

        float moved[NCOMB(p)];
        
        shift_multipoles<p>(src, moved, dpos);

        for(int iM=1; iM < ncomb; iM++) {
            float val = index < iend ? moved[iM] : 0.f;
            float sum = warp_reduce_sum(val);
            if ((threadIdx.x & 31) == 0) {
                atomicAdd(&mp[iM], sum);
            }
        }
    }
    __syncthreads();

    // write inode center
    if (threadIdx.x == 0) {
        xcent_out[inode] = com;
    }

    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp_out[inode * ncomb + iM] = mp[iM];
    }
}

void launch_CoarsenMultipolesKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const int *isplit, const float *mp_values, const float3 *xcent, float *mp_out, float3 *xcent_out) {
    LAUNCH_KERNEL_SWITCH(p, CoarsenMultipolesKernel, grid_size, block_size, stream, isplit, mp_values, xcent, mp_out, xcent_out);
}


// =============================================================
// Evaluate Tree Plane Kernel
// =============================================================

template<int p>
__global__ void EvaluateTreePlaneKernel(const EvaluateTreePlaneView v) {
    int2 nrange = v.node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 prange = {v.spl_nodes[nodeid], v.spl_nodes[nodeid + 1]};

    int ipart = prange.x + threadIdx.x;
    bool valid = ipart < prange.y;

    if(valid) {
        v.loc_out[ipart] = threadIdx.x; // placeholder write
    }
}

void launch_EvaluateTreePlaneKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const EvaluateTreePlaneView view) {
    LAUNCH_KERNEL_SWITCH(p, EvaluateTreePlaneKernel, grid_size, block_size, stream, view);
}