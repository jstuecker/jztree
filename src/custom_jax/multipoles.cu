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

template<int p>
__global__ void IlistM2LKernel(const float3 *x, const float *mp, int2 *interactions, int *iminmax, float *Lout, size_t interactions_per_block, float epsilon) {
    const size_t blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    int int_id = blockIdx.x * blocksize + threadIdx.x;

    int ncomb = p * (p + 1) * (p+2) / 2;

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + blockIdx.x * interactions_per_block + iint;
        if (int_id >= imax) {
            return;
        }

        for (int i = 0; i < ncomb; i++) {
            Lout[int_id * ncomb + i] = mp[int_id * ncomb + i];
        }
    }
}

void launch_IlistM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream, const float3 *x, const float *mp, int2 *interactions, int *iminmax, float *Lout, size_t interactions_per_block, float epsilon) {
    // This launch mechanic is needed so that p can be treated as a compile time constant
    // I wish there was a simpler way...
    switch(p) {
        case 0: IlistM2LKernel<0><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 1: IlistM2LKernel<1><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 2: IlistM2LKernel<2><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 3: IlistM2LKernel<3><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 4: IlistM2LKernel<4><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 5: IlistM2LKernel<5><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 6: IlistM2LKernel<6><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        default: throw std::runtime_error("Unsupported p value for IlistM2LKernel"); break;
    }
}

ffi::Error IlistM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> mp, ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax,  ffi::ResultBuffer<ffi::F32> loc, int p, size_t block_size, size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    const size_t grid_size = (ninteractions + interactions_per_block - 1) / interactions_per_block;

    auto* xfloat3 = reinterpret_cast<const float3*>(x.typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());

    cudaMemsetAsync(loc->typed_data(), 0, mp.element_count() * sizeof(float), stream);

    launch_IlistM2LKernel(p, grid_size, block_size, stream, xfloat3, mp.typed_data(), interactions_i2, iminmax.typed_data(), loc->typed_data(), interactions_per_block, epsilon);

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

NB_MODULE(nb_multipoles, m) {
    m.def("ilist_m2l", []() { return EncapsulateFfiCall(IlistM2L); });
}