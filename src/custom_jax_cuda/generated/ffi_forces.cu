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
#include "../forces.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: ForceAndPotential                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error ForceAndPotentialFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xm,
    ffi::Result<ffi::AnyBuffer> fphi,
    float epsilon,
    bool kahan,
    size_t block_size
) {
    int n = xm.element_count()/4;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(xm.element_count()/4, block_size));
    size_t smem = blockDim.x * sizeof(float4);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    PMass* xm_val = reinterpret_cast<PMass*>(xm.untyped_data());
    ForcePot* fphi_val = reinterpret_cast<ForcePot*>(fphi->untyped_data());

    void* args[] = {
        &xm_val,
        &fphi_val,
        &n,
        &epsilon
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<bool>;
    using TFunctionType = decltype(ForceAndPotential<true>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{true}] = ForceAndPotential<true>;
    instance_map[{false}] = ForceAndPotential<false>;

    auto it = instance_map.find({kahan});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in ForceAndPotentialFFIHost -- Only supporting:\n"\
            "(true), (false)"
        );
    }

    TFunctionType* instance = it->second;
    
    cudaLaunchKernel((const void*)instance, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ForceAndPotentialFFI, ForceAndPotentialFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xm
        .Ret<ffi::AnyBuffer>() // fphi
        .Attr<float>("epsilon")
        .Attr<bool>("kahan")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: GroupedForceAndPot                        */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error GroupedForceAndPotFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer posm,
    ffi::Result<ffi::AnyBuffer> fphi,
    float softening,
    int max_leaf_size
) {
    dim3 blockDim(128);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = blockDim.x * sizeof(float4);
    
    // Initialize output buffers
    cudaMemsetAsync(fphi->untyped_data(), 0, fphi->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int2* node_range_val = reinterpret_cast<int2*>(node_range.untyped_data());
    int* spl_nodes_val = reinterpret_cast<int*>(spl_nodes.untyped_data());
    int* spl_ilist_val = reinterpret_cast<int*>(spl_ilist.untyped_data());
    int* ilist_nodes_val = reinterpret_cast<int*>(ilist_nodes.untyped_data());
    PMass* posm_val = reinterpret_cast<PMass*>(posm.untyped_data());
    ForcePot* fphi_val = reinterpret_cast<ForcePot*>(fphi->untyped_data());

    void* args[] = {
        &node_range_val,
        &spl_nodes_val,
        &spl_ilist_val,
        &ilist_nodes_val,
        &posm_val,
        &fphi_val,
        &softening,
        &max_leaf_size
    };
    cudaLaunchKernel((const void*)GroupedForceAndPot, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GroupedForceAndPotFFI, GroupedForceAndPotFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // posm
        .Ret<ffi::AnyBuffer>() // fphi
        .Attr<float>("softening")
        .Attr<int>("max_leaf_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_forces, m) {
    m.def("ForceAndPotential", []() { return EncapsulateFfiCall(&ForceAndPotentialFFI); });
    m.def("GroupedForceAndPot", []() { return EncapsulateFfiCall(&GroupedForceAndPotFFI); });
}