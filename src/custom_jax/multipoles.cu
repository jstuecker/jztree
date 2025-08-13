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


__global__ void M2LKernel(const float3 *x, const float *mp, float *Lout, size_t n, int p, float epsilon) {
    const size_t blocksize = blockDim.x;
    const size_t steps = n / blocksize;
    float epsilon2 = epsilon * epsilon;

    int int_id = blockIdx.x * blocksize + threadIdx.x;
    if (int_id >= n) {
        return;
    }

    int ntot = p * (p + 1) * (p+2) / 2;

    for (int i = 0; i < ntot; i++) {
        Lout[int_id * ntot + i] = mp[int_id * ntot + i];
    }

    // float4 xmi = xm[blockIdx.x * blocksize + threadIdx.x];

    // extern __shared__ float4 xmj_shared[];

    // float phii = 0.f;

    // for (size_t jblock = 0; jblock < steps; jblock += 1) {
    //     // Load the next block of x into shared memory
    //     // this avoids reading from global memory multiple times
    //     __syncthreads();
    //     xmj_shared[threadIdx.x] = xm[jblock * blocksize + threadIdx.x];
    //     __syncthreads();

    //     for (size_t j = 0; j < blocksize; j++) {
    //         float4 xmj = xmj_shared[j];
            
    //         float dx = xmi.x - xmj.x;
    //         float dy = xmi.y - xmj.y;
    //         float dz = xmi.z - xmj.z;
    //         float m = xmj.w; // we packed mass into the w component of xm

    //         float r2 = dx*dx + dy*dy + dz*dz;

    //         phii += -1.0f*m*rsqrtf(r2 + epsilon2);
    //     }
    // }

    // // Since it is good to avoid any branches on GPU, we deal with the self-interaction by
    // // first adding it above and now subtracting it again:
    // phii += xmi.w / epsilon; 

    // phi[blockIdx.x * blocksize + threadIdx.x] = phii;


}

ffi::Error M2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> mp, ffi::ResultBuffer<ffi::F32> loc, int p, size_t block_size, float epsilon) {
    size_t n = x.element_count() / x.dimensions().back();

    const size_t grid_size = (n + (block_size - 1)) / block_size;

    auto* xfloat3 = reinterpret_cast<const float3*>(x.typed_data());
    M2LKernel<<<grid_size, block_size, 0, stream>>>(xfloat3, mp.typed_data(), loc->typed_data(), p, n, epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    M2L, M2LHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int>("p")
        .Attr<size_t>("block_size")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_multipoles, m) {
    m.def("m2l", []() { return EncapsulateFfiCall(M2L); });
}