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

    const size_t grid_size = (n + (block_size - 1)) / block_size;

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

    const size_t grid_size = (n + (block_size - 1)) / block_size;

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

__global__ void IlistForceAndPotKernel(const float4 *xm, int32_t *isplit, int2 *interactions, float4 *fphi, size_t interactions_per_block, float epsilon) {
    const int blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    extern __shared__ float4 xmj_shared[];

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int2 interaction = interactions[blockIdx.x * interactions_per_block + iint];
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

ffi::Error IlistForceHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::S32> isplit, ffi::Buffer<ffi::S32> interactions, ffi::ResultBuffer<ffi::F32> force, size_t block_size, size_t interactions_per_block, float epsilon) {
    size_t n = interactions.element_count() / 2;

    const size_t grid_size = n / interactions_per_block;

    auto* xm_float4 = reinterpret_cast<const float4*>(x.typed_data()); // interprete xm as an array of float4. This makes the kernel easier to write.
    auto* fphi_float4 = reinterpret_cast<float4*>(force->typed_data());
    auto* interactions2 = reinterpret_cast<int2*>(interactions.typed_data());
    
    cudaMemsetAsync(fphi_float4, 0, x.element_count() * sizeof(float), stream); // Initialize to 0, because of atomic adds

    IlistForceAndPotKernel<<<grid_size, block_size, block_size*sizeof(float4), stream>>>(xm_float4, isplit.typed_data(), interactions2, fphi_float4, interactions_per_block, epsilon);

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
        .Arg<ffi::Buffer<ffi::F32>>()              // x
        .Arg<ffi::Buffer<ffi::S32>>()              // isplit
        .Arg<ffi::Buffer<ffi::S32>>()              // interactions
        .Ret<ffi::Buffer<ffi::F32>>()              // acc
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
    m.def("ilist_force", []() { return EncapsulateFfiCall(IlistForce); });
}