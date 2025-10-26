// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt 
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of 
// boilerplate code that is easy to mess up.

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

#include "../shared_utils.cuh"
#include "../fmm.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: TestPositions                             */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error TestPositionsFFIHost (
    cudaStream_t stream,
    ffi::AnyBuffer indices,
    ffi::Result<ffi::AnyBuffer> positions,
    int num,
    float boxsize,
    int p,
    size_t grid_size
) {
    dim3 blockDim(64);
    dim3 gridDim(grid_size);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* indices_val = reinterpret_cast<int*>(indices.untyped_data());
    float3* positions_val = reinterpret_cast<float3*>(positions->untyped_data());;;

    void* args[] = {
        &indices_val,
        &positions_val,
        &num,
        &boxsize
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through switch statements
    const void* kernel;
    switch(p) {
        case 0: kernel = (const void*) TestPositions<0>; break;
        case 1: kernel = (const void*) TestPositions<1>; break;
        case 2: kernel = (const void*) TestPositions<2>; break;
        case 3: kernel = (const void*) TestPositions<3>; break;
        default: return ffi::Error::Internal("Unsupported p value in TestPositionsFFIHost");
    };
    
    cudaLaunchKernel(kernel, gridDim, blockDim, args, 0, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TestPositionsFFI, TestPositionsFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // indices
        .Ret<ffi::AnyBuffer>() // positions
        .Attr<int>("num")
        .Attr<float>("boxsize")
        .Attr<int>("p")
        .Attr<size_t>("grid_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SimpleArange                              */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SimpleArangeFFIHost (
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> output,
    int size,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(output->dimensions()[0], block_size));
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* output_val = reinterpret_cast<int*>(output->untyped_data());;

    void* args[] = {
        &output_val,
        &size
    };
    cudaLaunchKernel((const void*)SimpleArange, gridDim, blockDim, args, 0, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SimpleArangeFFI, SimpleArangeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ret<ffi::AnyBuffer>() // output
        .Attr<int>("size")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fmm, m) {
    m.def("TestPositions", []() { return EncapsulateFfiCall(&TestPositionsFFI); });
    m.def("SimpleArange", []() { return EncapsulateFfiCall(&SimpleArangeFFI); });
}