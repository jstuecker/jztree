#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// A wrapper to encapsulate an FFI call
template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

// =============================================================
// Multipole Translators
// =============================================================

#define MAXP 6

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
    // Get's the k-th component of a float3 vector
    // Note that we need to do it this way (rather than with an array type vector),
    // because it is not possible to dynamically index register arrays (and if you do
    // they will end up in local memory (=very slow))
    return (k == 0) ? x.x : (k == 1 ? x.y : x.z);
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
    // Set's up the cartesian derivatives of the Green's function
    // Using the method described in Tausch (2003):
    // "The fast multipole method for arbitrary Green’s functions"
    // This works by using a recurrence between the derivatives of the Green's function
    // We start at D0 G(q), go to D1 G(q+1), D2 G(q+2) ...

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
    const size_t blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    constexpr int ncomb = NCOMB(p);

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + blocksize * (blockIdx.x * interactions_per_block + iint) + threadIdx.x;
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
    // This launch mechanic is needed so that p can be treated as a compile time constant
    // I wish there was a simpler way...
    switch(p) {
        case 1: IlistM2LKernel<1><<<grid_size, block_size, 0, stream>>>(
            x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 2: IlistM2LKernel<2><<<grid_size, block_size, 0, stream>>>(
            x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 3: IlistM2LKernel<3><<<grid_size, block_size, 0, stream>>>(
            x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 4: IlistM2LKernel<4><<<grid_size, block_size, 0, stream>>>(
            x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 5: IlistM2LKernel<5><<<grid_size, block_size, 0, stream>>>(
            x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 6: IlistM2LKernel<6><<<grid_size, block_size, 0, stream>>>(
            x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        default: throw std::runtime_error("Unsupported p value for IlistM2LKernel"); break;
    }
}

ffi::Error IlistM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> mp, 
        ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax,  
        ffi::ResultBuffer<ffi::F32> loc, int p, size_t block_size, size_t interactions_per_block, 
        float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    const size_t grid_size = (ninteractions + block_size*interactions_per_block - 1) / (block_size*interactions_per_block);

    auto* xfloat3 = reinterpret_cast<const float3*>(x.typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());

    cudaMemsetAsync(loc->typed_data(), 0, mp.element_count() * sizeof(float), stream);

    launch_IlistM2LKernel(p, grid_size, block_size, stream, xfloat3, mp.typed_data(), 
        interactions_i2, iminmax.typed_data(), loc->typed_data(), interactions_per_block, epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistM2L, IlistM2LHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()     // interactions
        .Arg<ffi::Buffer<ffi::S32>>()     // iminmax
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int>("p")
        .Attr<size_t>("block_size")
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

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
    const int blocksize = blockDim.x;
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

        for(int poffset=0; poffset < iPartEnd - iPartStart; poffset += blocksize)
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
        for(int kflat=threadIdx.x; kflat < ncomb; kflat += blocksize) {
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
    // This launch mechanic is needed so that p can be treated as a compile time constant
    // I wish there was a simpler way...
    switch(p) {
        case 1: IlistLeaf2NodeM2LKernel<1><<<grid_size, block_size, 0, stream>>>(xnodes, xm, 
            isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 2: IlistLeaf2NodeM2LKernel<2><<<grid_size, block_size, 0, stream>>>(xnodes, xm, 
            isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 3: IlistLeaf2NodeM2LKernel<3><<<grid_size, block_size, 0, stream>>>(xnodes, xm, 
            isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 4: IlistLeaf2NodeM2LKernel<4><<<grid_size, block_size, 0, stream>>>(xnodes, xm, 
            isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 5: IlistLeaf2NodeM2LKernel<5><<<grid_size, block_size, 0, stream>>>(xnodes, xm, 
            isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 6: IlistLeaf2NodeM2LKernel<6><<<grid_size, block_size, 0, stream>>>(xnodes, xm, 
            isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;

        default: throw std::runtime_error("Unsupported p value for IlistM2LKernel"); break;
    }
}

ffi::Error IlistLeaf2NodeM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> xnodes, 
        ffi::Buffer<ffi::F32> xm, ffi::Buffer<ffi::S32> isplit, ffi::Buffer<ffi::S32> interactions, 
        ffi::Buffer<ffi::S32> iminmax, ffi::ResultBuffer<ffi::F32> loc, int p, 
        size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    // Since our kernel uses a lot of registers, we cannot reach full occupancy in general.
    // Therefore it is fine to use block_size = 32 (which usually should be avoided, since it
    // limits maximum occupancy to 50%.) The benefit of this is that leafs that have <= 32 particles
    // can be handled at a smaller cost. I have tested this and found it to be the fastest choice
    size_t block_size = 32;

    size_t grid_size = (ninteractions + interactions_per_block - 1) / interactions_per_block;

    auto* xnodes_float3 = reinterpret_cast<const float3*>(xnodes.typed_data());
    auto* xm_float4 = reinterpret_cast<const float4*>(xm.typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());
    
    cudaMemsetAsync(loc->typed_data(), 0, loc->element_count() * sizeof(float), stream);

    launch_IlistLeaf2NodeM2LKernel(p, grid_size, block_size, stream, xnodes_float3, xm_float4, 
        isplit.typed_data(), interactions_i2, iminmax.typed_data(), loc->typed_data(), 
        interactions_per_block, epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistLeaf2NodeM2L, IlistLeaf2NodeM2LHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()     // xnodes
        .Arg<ffi::Buffer<ffi::F32>>()     // xm
        .Arg<ffi::Buffer<ffi::S32>>()     // isplit
        .Arg<ffi::Buffer<ffi::S32>>()     // interactions
        .Arg<ffi::Buffer<ffi::S32>>()     // iminmax
        .Ret<ffi::Buffer<ffi::F32>>()     // Lk output
        .Attr<int>("p")
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_multipoles, m) {
    m.def("ilist_m2l", []() { return EncapsulateFfiCall(IlistM2L); });
    m.def("ilist_leaf2node_m2l", []() { return EncapsulateFfiCall(IlistLeaf2NodeM2L); });
}