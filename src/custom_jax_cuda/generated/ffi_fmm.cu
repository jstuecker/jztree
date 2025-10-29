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
/*                             FFI call to CUDA kernel: CountInteractionsAndM2L                   */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error CountInteractionsAndM2LFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer children,
    ffi::AnyBuffer mp_values,
    ffi::Result<ffi::AnyBuffer> loc_out,
    ffi::Result<ffi::AnyBuffer> child_count_out,
    float softening,
    float opening_angle,
    int p
) {
    dim3 blockDim(32);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(loc_out->untyped_data(), 0, loc_out->size_bytes(), stream);
    cudaMemsetAsync(child_count_out->untyped_data(), 0, child_count_out->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int2* node_range_val = reinterpret_cast<int2*>(node_range.untyped_data());
    int* spl_nodes_val = reinterpret_cast<int*>(spl_nodes.untyped_data());
    int* spl_ilist_val = reinterpret_cast<int*>(spl_ilist.untyped_data());
    int* ilist_nodes_val = reinterpret_cast<int*>(ilist_nodes.untyped_data());
    NodeInfo* children_val = reinterpret_cast<NodeInfo*>(children.untyped_data());
    float* mp_values_val = reinterpret_cast<float*>(mp_values.untyped_data());
    float* loc_out_val = reinterpret_cast<float*>(loc_out->untyped_data());
    int* child_count_out_val = reinterpret_cast<int*>(child_count_out->untyped_data());

    void* args[] = {
        &node_range_val,
        &spl_nodes_val,
        &spl_ilist_val,
        &ilist_nodes_val,
        &children_val,
        &mp_values_val,
        &loc_out_val,
        &child_count_out_val,
        &softening,
        &opening_angle
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through switch statements
    const void* kernel;
    switch(p) {
        case 1: kernel = (const void*) CountInteractionsAndM2L<1>; break;
        case 2: kernel = (const void*) CountInteractionsAndM2L<2>; break;
        case 3: kernel = (const void*) CountInteractionsAndM2L<3>; break;
        default: return ffi::Error::Internal(
            "Unsupported p=" + std::to_string(p) + " in CountInteractionsAndM2LFFIHost"\
            " -- Only supporting values: (1,2,3)"
        );
    };
    
    cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CountInteractionsAndM2LFFI, CountInteractionsAndM2LFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // children
        .Arg<ffi::AnyBuffer>() // mp_values
        .Ret<ffi::AnyBuffer>() // loc_out
        .Ret<ffi::AnyBuffer>() // child_count_out
        .Attr<float>("softening")
        .Attr<float>("opening_angle")
        .Attr<int>("p"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fmm, m) {
    m.def("CountInteractionsAndM2L", []() { return EncapsulateFfiCall(&CountInteractionsAndM2LFFI); });
}