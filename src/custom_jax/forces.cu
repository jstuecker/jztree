#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "shared_utils.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// Each custom FFI handler has four parts:
// (1) the CUDA kernel
// (2) the host function that launches the kernel
// (3) the FFI handler registration that registers the handler with the XLA runtime
// To make the python interface easier, we use nanobind (rather than ctypes). This requires
// additionally (3b) that we declare the function to our nanobind module at the bottom of this file.
// And in Python:
// (4) A Python function that calls the FFI handler


// =============================================================
// Potential Kernel
// =============================================================

__global__ void PotentialKernel(const float4 *xm, float *phi, size_t n, float epsilon) {
    const size_t blocksize = blockDim.x;
    const size_t steps = n / blocksize;
    float epsilon2 = epsilon * epsilon;

    float4 xmi = xm[blockIdx.x * blocksize + threadIdx.x];

    extern __shared__ float4 xmj_shared[];

    float phii = 0.f;

    for (size_t jblock = 0; jblock < steps; jblock += 1) {
        // Load the next block of x into shared memory
        // this avoids reading from global memory multiple times
        __syncthreads();
        xmj_shared[threadIdx.x] = xm[jblock * blocksize + threadIdx.x];
        __syncthreads();

        for (size_t j = 0; j < blocksize; j++) {
            float4 xmj = xmj_shared[j];
            
            float dx = xmi.x - xmj.x;
            float dy = xmi.y - xmj.y;
            float dz = xmi.z - xmj.z;
            float m = xmj.w; // we packed mass into the w component of xm

            float r2 = dx*dx + dy*dy + dz*dz;

            phii += -1.0f*m*rsqrtf(r2 + epsilon2);
        }
    }

    // Since it is good to avoid any branches on GPU, we deal with the self-interaction by
    // first adding it above and now subtracting it again:
    phii += xmi.w / epsilon; 

    phi[blockIdx.x * blocksize + threadIdx.x] = phii;
}

ffi::Error PotentialHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> phi, size_t block_size, float epsilon) {
    size_t n = x.element_count() / x.dimensions().back();

    const size_t grid_size = div_ceil(n, block_size);

    auto* xm_float4 = reinterpret_cast<const float4*>(x.typed_data()); // interprete xm as an array of float4. This makes the kernel easier to write.
    PotentialKernel<<<grid_size, block_size, block_size*sizeof(float4), stream>>>(xm_float4, phi->typed_data(), n, epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Potential, PotentialHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()              // x
        .Ret<ffi::Buffer<ffi::F32>>()              // phi
        .Attr<size_t>("block_size")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

// =============================================================
// Force Kernel
// =============================================================

__forceinline__ __device__ void accumulateForce(float4 xmi, float4 xmj, float epsilon2, float3 &force_i) {
    float dx = xmj.x - xmi.x;
    float dy = xmj.y - xmi.y;
    float dz = xmj.z - xmi.z;
    float m = xmj.w;

    float r2 = dx*dx + dy*dy + dz*dz + epsilon2;
    float rinv = rsqrtf(r2);

    float minvr3 = m * rinv * rinv * rinv;

    force_i.x += minvr3 * dx;
    force_i.y += minvr3 * dy;
    force_i.z += minvr3 * dz;
}

__global__ void ForceKernel(const float4 *xm, float3 *force, size_t n, float epsilon) {
    const size_t blocksize = blockDim.x;
    const size_t steps = n / blocksize;
    float epsilon2 = epsilon * epsilon;

    float4 xmi = xm[blockIdx.x * blocksize + threadIdx.x];

    extern __shared__ float4 xmj_shared[];

    float3 force_i = {0.f, 0.f, 0.f};

    for (size_t jblock = 0; jblock < steps; jblock += 1) {
        __syncthreads();
        xmj_shared[threadIdx.x] = xm[jblock * blocksize + threadIdx.x];
        __syncthreads();

        for (size_t j = 0; j < blocksize; j++) {
            accumulateForce(xmi, xmj_shared[j], epsilon2, force_i);
        }
    }

    force[blockIdx.x * blocksize + threadIdx.x] = force_i;
}

/* Also create an alternative kernel that computes force + potential. This creates only a very small overhead (10%ish) */

__forceinline__ __device__ void accumulateForceAndPot(float4 xmi, float4 xmj, float epsilon2, float4 &fphi_i) {
    float dx = xmj.x - xmi.x;
    float dy = xmj.y - xmi.y;
    float dz = xmj.z - xmi.z;
    float m = xmj.w;

    float r2 = dx*dx + dy*dy + dz*dz + epsilon2;
    float rinv = rsqrtf(r2);

    float minvr = m * rinv;
    float minvr3 = minvr * rinv * rinv;

    fphi_i.x += minvr3 * dx;
    fphi_i.y += minvr3 * dy;
    fphi_i.z += minvr3 * dz;
    fphi_i.w += -minvr; // accumulate potential in the w component
}

__global__ void ForceAndPotKernel(const float4 *xm, float4 *fphi, size_t n, float epsilon) {
    const size_t blocksize = blockDim.x;
    const size_t steps = n / blocksize;
    float epsilon2 = epsilon * epsilon;

    float4 xmi = xm[blockIdx.x * blocksize + threadIdx.x];

    extern __shared__ float4 xmj_shared[];

    float4 fphi_i = {0.f, 0.f, 0.f, 0.f};

    for (size_t jblock = 0; jblock < steps; jblock += 1) {
        __syncthreads();
        xmj_shared[threadIdx.x] = xm[jblock * blocksize + threadIdx.x];
        __syncthreads();

        for (size_t j = 0; j < blocksize; j++) {
            accumulateForceAndPot(xmi, xmj_shared[j], epsilon2, fphi_i);
        }
    }

    fphi_i.w += xmi.w / epsilon; // remove self-interaction from potential

    fphi[blockIdx.x * blocksize + threadIdx.x] = fphi_i;
}

ffi::Error ForceHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> force, size_t block_size, float epsilon, bool get_potential = false) {
    size_t n = x.element_count() / x.dimensions().back();

    const size_t grid_size = div_ceil(n, block_size);

    auto* xm_float4 = reinterpret_cast<const float4*>(x.typed_data()); // interprete xm as an array of float4. This makes the kernel easier to write.
    if (get_potential) {
        auto* fphi_float4 = reinterpret_cast<float4*>(force->typed_data());
        ForceAndPotKernel<<<grid_size, block_size, block_size*sizeof(float4), stream>>>(xm_float4, fphi_float4, n, epsilon);
    }
    else {
        auto* force_float3 = reinterpret_cast<float3*>(force->typed_data()); 
        ForceKernel<<<grid_size, block_size, block_size*sizeof(float4), stream>>>(xm_float4, force_float3, n, epsilon);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Force, ForceHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()              // x
        .Ret<ffi::Buffer<ffi::F32>>()              // acc
        .Attr<size_t>("block_size")
        .Attr<float>("epsilon")
        .Attr<bool>("get_potential"),
    {xla::ffi::Traits::kCmdBufferCompatible});

// =============================================================
// Force Via Interaction List
// =============================================================

__device__ inline void atomicAddFloat4(float4* addr, const float4 val) {
    atomicAdd(&addr->x, val.x);
    atomicAdd(&addr->y, val.y);
    atomicAdd(&addr->z, val.z);
    atomicAdd(&addr->w, val.w);
}

__global__ void IlistForceAndPotKernel(const float4 *xm, int32_t *isplit, int2 *interactions, float4 *fphi, size_t interactions_per_block, int *iminmax, float epsilon) {
    const int blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    extern __shared__ float4 xmj_shared[];

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
            float4 fphi_i = {0.f, 0.f, 0.f, 0.f};
            float4 xmi = {0.f, 0.f, 0.f, 0.f};
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
                atomicAddFloat4(&fphi[iAstart + ioffA], fphi_i);
            }
        }
    }
}

ffi::Error IlistForceHost(cudaStream_t stream, ffi::Buffer<ffi::F32> xm, ffi::Buffer<ffi::S32> isplit, ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax, ffi::ResultBuffer<ffi::F32> fphi, size_t block_size, size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;

    const size_t grid_size = div_ceil(ninteractions, interactions_per_block);

    auto* xm_float4 = reinterpret_cast<const float4*>(xm.typed_data()); // interprete xm as an array of float4. This makes the kernel easier to write.
    auto* fphi_float4 = reinterpret_cast<float4*>(fphi->typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());
    
    cudaMemsetAsync(fphi_float4, 0, xm.element_count() * sizeof(float), stream); // Initialize to 0, because of atomic adds

    IlistForceAndPotKernel<<<grid_size, block_size, block_size*sizeof(float4), stream>>>(xm_float4, isplit.typed_data(), interactions_i2, fphi_float4, interactions_per_block, iminmax.typed_data(), epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistForce, IlistForceHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()              // xm
        .Arg<ffi::Buffer<ffi::S32>>()              // isplit
        .Arg<ffi::Buffer<ffi::S32>>()              // interactions
        .Arg<ffi::Buffer<ffi::S32>>()              // iminmax
        .Ret<ffi::Buffer<ffi::F32>>()              // fphi
        .Attr<size_t>("block_size")
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

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

__global__ void BwdIlistForceAndPotKernel(const float4 *gfphi, const float4 *xm, int32_t *isplit, int2 *interactions, float4 *gxm, size_t interactions_per_block, int *iminmax, float epsilon) {
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

ffi::Error BwdIlistFPhiHost(cudaStream_t stream, ffi::Buffer<ffi::F32> g, ffi::Buffer<ffi::F32> xm, ffi::Buffer<ffi::S32> isplit, ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax, ffi::ResultBuffer<ffi::F32> gxm_out, size_t block_size, size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;

    const size_t grid_size = div_ceil(ninteractions, interactions_per_block);

    auto g_float4 = reinterpret_cast<const float4*>(g.typed_data()); 
    auto* xm_float4 = reinterpret_cast<const float4*>(xm.typed_data()); 
    auto* gxm_float4 = reinterpret_cast<float4*>(gxm_out->typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());
    
    cudaMemsetAsync(gxm_float4, 0, xm.element_count() * sizeof(float), stream); // Initialize to 0, because of atomic adds

    BwdIlistForceAndPotKernel<<<grid_size, block_size, 2*block_size*sizeof(float4), stream>>>(g_float4, xm_float4, isplit.typed_data(), interactions_i2, gxm_float4, interactions_per_block, iminmax.typed_data(), epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BwdIlistForce, BwdIlistFPhiHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()              // g
        .Arg<ffi::Buffer<ffi::F32>>()              // xm
        .Arg<ffi::Buffer<ffi::S32>>()              // isplit
        .Arg<ffi::Buffer<ffi::S32>>()              // interactions
        .Arg<ffi::Buffer<ffi::S32>>()              // iminmax
        .Ret<ffi::Buffer<ffi::F32>>()              // fphi
        .Attr<size_t>("block_size")
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

// =============================================================
// Module Registrations
// =============================================================

NB_MODULE(nb_forces, m) {
    m.def("potential", []() { return EncapsulateFfiCall(Potential); });
    m.def("force", []() { return EncapsulateFfiCall(Force); });
    m.def("ilist_fphi", []() { return EncapsulateFfiCall(IlistForce); });
    m.def("ilist_fphi_bwd", []() { return EncapsulateFfiCall(BwdIlistForce); });
}