// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt 
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of 
// boilerplate code that is easy to mess up.

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

#include "../shared_utils.cuh"
#include "../ffi_example.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SimpleArange                              */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SimpleArangeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output,
    int size,
    int p,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(output->element_count(), block_size));
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* add_val = reinterpret_cast<int*>(add.untyped_data());
    int* output_val = reinterpret_cast<int*>(output->untyped_data());

    void* args[] = {
        &add_val,
        &output_val,
        &size
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through switch statements
    const void* kernel;
    switch(p) {
        case 1: kernel = (const void*) SimpleArange<1>; break;
        case 2: kernel = (const void*) SimpleArange<2>; break;
        case 3: kernel = (const void*) SimpleArange<3>; break;
        default: return ffi::Error::Internal(
            "Unsupported p=" + std::to_string(p) + " in SimpleArangeFFIHost"\
            " -- Only supporting values: (1,2,3)"
        );
    };
    
    cudaLaunchKernel(kernel, gridDim, blockDim, args, 0, stream);

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
        .Arg<ffi::AnyBuffer>() // add
        .Ret<ffi::AnyBuffer>() // output
        .Attr<int>("size")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SetToConstantCall                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SetToConstantCallFFIHost(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> output,
    int value,
    size_t block_size,
    int tpar
) {
    int size = output->element_count();
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through switch statements
    auto fptr = SetToConstantCall<16>;
    switch(tpar) {
        case 16: fptr = SetToConstantCall<16>; break;
        case 32: fptr = SetToConstantCall<32>; break;
        case 64: fptr = SetToConstantCall<64>; break;
        default: return ffi::Error::Internal(
            "Unsupported tpar=" + std::to_string(tpar) + " in SetToConstantCallFFIHost"\
            " -- Only supporting values: (16,32,64)"
        );
    };

    // Now call our function
    fptr(
        stream,
        reinterpret_cast<int*>(output->untyped_data()),
        value,
        size,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SetToConstantCallFFI, SetToConstantCallFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ret<ffi::AnyBuffer>() // output
        .Attr<int>("value")
        .Attr<size_t>("block_size")
        .Attr<int>("tpar"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_example, m) {
    m.def("SimpleArange", []() { return EncapsulateFfiCall(&SimpleArangeFFI); });
    m.def("SetToConstantCall", []() { return EncapsulateFfiCall(&SetToConstantCallFFI); });
}