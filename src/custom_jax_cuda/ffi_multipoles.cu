#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include <string>
#include "shared_utils.cuh"
#include "multipoles.cuh"
#include "fmm.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;


ffi::Error IlistM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> mp, 
        ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax,  
        ffi::ResultBuffer<ffi::F32> loc, int p, size_t block_size, size_t interactions_per_block, 
        float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    const size_t grid_size = div_ceil(ninteractions, block_size*interactions_per_block);

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


ffi::Error IlistLeaf2NodeM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> xnodes, 
        ffi::Buffer<ffi::F32> xm, ffi::Buffer<ffi::S32> isplit, ffi::Buffer<ffi::S32> interactions, 
        ffi::Buffer<ffi::S32> iminmax, ffi::ResultBuffer<ffi::F32> loc, int p, 
        size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    size_t block_size = 32;

    size_t grid_size = div_ceil(ninteractions, interactions_per_block);

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


ffi::Error MultipolesFromParticlesHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> isplit,
    ffi::Buffer<ffi::F32> part_posm,
    ffi::ResultBuffer<ffi::F32> mp_out,
    ffi::ResultBuffer<ffi::F32> xcom_out,
    size_t p,
    size_t block_size
)  {
    size_t grid_size = isplit.element_count() - 1;

    PosMass* pposm = reinterpret_cast<PosMass*>(part_posm.typed_data());
    float3* xcom = reinterpret_cast<float3*>(xcom_out->typed_data());


    launch_MultipolesFromParticlesKernel(p, grid_size, block_size, stream, 
        isplit.typed_data(), pposm, 
        mp_out->typed_data(), xcom);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


// FFI Host wrapper for coarsen_multipoles
ffi::Error CoarsenMultipolesHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> ispl,
    ffi::Buffer<ffi::F32> mp_values,
    ffi::Buffer<ffi::F32> mp_center,
    ffi::ResultBuffer<ffi::F32> out_mp,
    ffi::ResultBuffer<ffi::F32> out_xcent,
    size_t p,
    size_t block_size
) {
    // grid_size: one coarse node per element in ispl minus one
    size_t grid_size = ispl.element_count() - 1;

    const int *isplit_ptr = reinterpret_cast<const int*>(ispl.typed_data());
    const float3 *center_ptr = reinterpret_cast<const float3*>(mp_center.typed_data());
    const float *values_ptr = reinterpret_cast<const float*>(mp_values.typed_data());

    float *out_mp_ptr = out_mp->typed_data();
    float3 *out_xcent_ptr = reinterpret_cast<float3*>(out_xcent->typed_data());

    launch_CoarsenMultipolesKernel(static_cast<int>(p), grid_size, block_size, stream,
        isplit_ptr, values_ptr, center_ptr, out_mp_ptr, out_xcent_ptr);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


// The actual FFI handler symbols
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MultipolesFromParticles, MultipolesFromParticlesHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()     // isplit
        .Arg<ffi::Buffer<ffi::F32>>()     // part.posm
        .Ret<ffi::Buffer<ffi::F32>>()     // mp output
        .Ret<ffi::Buffer<ffi::F32>>()     // xcom output
        .Attr<size_t>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CoarsenMultipoles, CoarsenMultipolesHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()     // ispl
        .Arg<ffi::Buffer<ffi::F32>>()     // mp.values
        .Arg<ffi::Buffer<ffi::F32>>()     // mp.center
        .Ret<ffi::Buffer<ffi::F32>>()     // out_mp
        .Ret<ffi::Buffer<ffi::F32>>()     // out_xcent
        .Attr<size_t>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});


NB_MODULE(ffi_multipoles, m) {
    m.def("ilist_m2l", []() { return EncapsulateFfiCall(IlistM2L); });
    m.def("ilist_leaf2node_m2l", []() { return EncapsulateFfiCall(IlistLeaf2NodeM2L); });
    m.def("multipoles_from_particles", []() { return EncapsulateFfiCall(MultipolesFromParticles); });
    m.def("coarsen_multipoles", []() { return EncapsulateFfiCall(CoarsenMultipoles); });
}
