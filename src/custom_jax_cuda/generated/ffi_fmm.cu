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
/*                             FFI call to CUDA kernel: EvaluateTreePlane                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error EvaluateTreePlaneFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer xchild,
    ffi::AnyBuffer mp_values,
    ffi::Result<ffi::AnyBuffer> loc_out,
    ffi::Result<ffi::AnyBuffer> spl_child_ilist_out,
    ffi::Result<ffi::AnyBuffer> child_ilist_out,
    float epsilon,
    int p,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(spl_nodes.element_count() - 1);
    
    // Initialize output buffers
    cudaMemsetAsync(loc_out->untyped_data(), 0, loc_out->size_bytes(), stream);
    cudaMemsetAsync(spl_child_ilist_out->untyped_data(), 0, spl_child_ilist_out->size_bytes(), stream);
    cudaMemsetAsync(child_ilist_out->untyped_data(), 0, child_ilist_out->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int2* node_range_val = reinterpret_cast<int2*>(node_range.untyped_data());
    int* spl_nodes_val = reinterpret_cast<int*>(spl_nodes.untyped_data());
    int* spl_ilist_val = reinterpret_cast<int*>(spl_ilist.untyped_data());
    int* ilist_nodes_val = reinterpret_cast<int*>(ilist_nodes.untyped_data());
    float3* xchild_val = reinterpret_cast<float3*>(xchild.untyped_data());
    float* mp_values_val = reinterpret_cast<float*>(mp_values.untyped_data());
    float* loc_out_val = reinterpret_cast<float*>(loc_out->untyped_data());
    int* spl_child_ilist_out_val = reinterpret_cast<int*>(spl_child_ilist_out->untyped_data());
    int* child_ilist_out_val = reinterpret_cast<int*>(child_ilist_out->untyped_data());;

    void* args[] = {
        &node_range_val,
        &spl_nodes_val,
        &spl_ilist_val,
        &ilist_nodes_val,
        &xchild_val,
        &mp_values_val,
        &loc_out_val,
        &spl_child_ilist_out_val,
        &child_ilist_out_val,
        &epsilon
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through switch statements
    const void* kernel;
    switch(p) {
        case 1: kernel = (const void*) EvaluateTreePlane<1>; break;
        case 2: kernel = (const void*) EvaluateTreePlane<2>; break;
        case 3: kernel = (const void*) EvaluateTreePlane<3>; break;
        default: return ffi::Error::Internal("Unsupported p value in EvaluateTreePlaneFFIHost");
    };
    
    cudaLaunchKernel(kernel, gridDim, blockDim, args, 0, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EvaluateTreePlaneFFI, EvaluateTreePlaneFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // xchild
        .Arg<ffi::AnyBuffer>() // mp_values
        .Ret<ffi::AnyBuffer>() // loc_out
        .Ret<ffi::AnyBuffer>() // spl_child_ilist_out
        .Ret<ffi::AnyBuffer>() // child_ilist_out
        .Attr<float>("epsilon")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: TestPositions                             */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error TestPositionsFFIHost(
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

ffi::Error SimpleArangeFFIHost(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> output,
    size_t block_size
) {
    int size = output->element_count();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(output->element_count(), block_size));
    
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
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fmm, m) {
    m.def("EvaluateTreePlane", []() { return EncapsulateFfiCall(&EvaluateTreePlaneFFI); });
    m.def("TestPositions", []() { return EncapsulateFfiCall(&TestPositionsFFI); });
    m.def("SimpleArange", []() { return EncapsulateFfiCall(&SimpleArangeFFI); });
}