// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt 
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of 
// boilerplate code that is easy to mess up.

#include <map>
#include <tuple>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// A wrapper to encapsulate an FFI call
template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}
#include "../common/math.cuh"
#include "../tools.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: RearangeSegments                          */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error RearangeSegmentsFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer data_in,
    ffi::AnyBuffer seg_spl_out,
    ffi::AnyBuffer seg_offset_in,
    ffi::Result<ffi::AnyBuffer> data_out,
    int64_t size,
    int64_t dtype_bytes,
    size_t grid_size,
    size_t block_size
) {
    int64_t size_seg = seg_spl_out.element_count()-1;
    dim3 blockDim(block_size);
    dim3 gridDim(grid_size);
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    uint8_t* data_in_val = reinterpret_cast<uint8_t*>(data_in.untyped_data());
    int64_t* seg_spl_out_val = reinterpret_cast<int64_t*>(seg_spl_out.untyped_data());
    int64_t* seg_offset_in_val = reinterpret_cast<int64_t*>(seg_offset_in.untyped_data());
    uint8_t* data_out_val = reinterpret_cast<uint8_t*>(data_out->untyped_data());

    void* args[] = {
        &data_in_val,
        &seg_spl_out_val,
        &seg_offset_in_val,
        &data_out_val,
        &size,
        &size_seg,
        &dtype_bytes
    };
    cudaLaunchKernel((const void*)RearangeSegments, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RearangeSegmentsFFI, RearangeSegmentsFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // data_in
        .Arg<ffi::AnyBuffer>() // seg_spl_out
        .Arg<ffi::AnyBuffer>() // seg_offset_in
        .Ret<ffi::AnyBuffer>() // data_out
        .Attr<int64_t>("size")
        .Attr<int64_t>("dtype_bytes")
        .Attr<size_t>("grid_size")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_tools, m) {
    m.def("RearangeSegments", []() { return EncapsulateFfiCall(&RearangeSegmentsFFI); });
}