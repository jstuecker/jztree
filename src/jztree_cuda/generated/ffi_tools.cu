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

using DT = ffi::DataType;

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
    void* data_in_arg = data_in.untyped_data();
    void* seg_spl_out_arg = seg_spl_out.untyped_data();
    void* seg_offset_in_arg = seg_offset_in.untyped_data();
    void* data_out_arg = data_out->untyped_data();
    void* args[] = {
        &data_in_arg,
        &seg_spl_out_arg,
        &seg_offset_in_arg,
        &data_out_arg,
        &size,
        &size_seg,
        &dtype_bytes
    };
    const void* instance = (const void*)RearangeSegments;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

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
/*                             FFI call to CUDA kernel: MapInRange                                */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error MapInRangeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer range,
    ffi::AnyBuffer input,
    ffi::AnyBuffer map,
    ffi::Result<ffi::AnyBuffer> output,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(input.element_count(), block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* range_arg = range.untyped_data();
    void* input_arg = input.untyped_data();
    void* map_arg = map.untyped_data();
    void* output_arg = output->untyped_data();
    void* args[] = {
        &range_arg,
        &input_arg,
        &map_arg,
        &output_arg
    };
    const void* instance = (const void*)MapInRange;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MapInRangeFFI, MapInRangeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // range
        .Arg<ffi::AnyBuffer>() // input
        .Arg<ffi::AnyBuffer>() // map
        .Ret<ffi::AnyBuffer>() // output
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_tools, m) {
    m.def("RearangeSegments", []() { return EncapsulateFfiCall(&RearangeSegmentsFFI); });
    m.def("MapInRange", []() { return EncapsulateFfiCall(&MapInRangeFFI); });
}