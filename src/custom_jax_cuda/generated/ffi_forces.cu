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
/*                             FFI call to CUDA kernel: BwdForceAndPotential                      */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error BwdForceAndPotentialFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer gfphi,
    ffi::AnyBuffer xm,
    ffi::Result<ffi::AnyBuffer> gxm,
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
    float4* gfphi_val = reinterpret_cast<float4*>(gfphi.untyped_data());
    PMass* xm_val = reinterpret_cast<PMass*>(xm.untyped_data());
    float4* gxm_val = reinterpret_cast<float4*>(gxm->untyped_data());

    void* args[] = {
        &gfphi_val,
        &xm_val,
        &gxm_val,
        &n,
        &epsilon
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<bool>;
    using TFunctionType = decltype(BwdForceAndPotential<true>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{true}] = BwdForceAndPotential<true>;
    instance_map[{false}] = BwdForceAndPotential<false>;

    auto it = instance_map.find({kahan});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in BwdForceAndPotentialFFIHost -- Only supporting:\n"\
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
    BwdForceAndPotentialFFI, BwdForceAndPotentialFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // gfphi
        .Arg<ffi::AnyBuffer>() // xm
        .Ret<ffi::AnyBuffer>() // gxm
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
    int max_leaf_size,
    bool kahan
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
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<bool>;
    using TFunctionType = decltype(GroupedForceAndPot<true>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{true}] = GroupedForceAndPot<true>;
    instance_map[{false}] = GroupedForceAndPot<false>;

    auto it = instance_map.find({kahan});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in GroupedForceAndPotFFIHost -- Only supporting:\n"\
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
        .Attr<int>("max_leaf_size")
        .Attr<bool>("kahan"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: IlistForceAndPot                          */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error IlistForceAndPotFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xm,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer interactions,
    ffi::AnyBuffer iminmax,
    ffi::Result<ffi::AnyBuffer> fphi,
    size_t interactions_per_block,
    float epsilon,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(interactions.element_count()/2, interactions_per_block));
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(fphi->untyped_data(), 0, fphi->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    PMass* xm_val = reinterpret_cast<PMass*>(xm.untyped_data());
    int32_t* isplit_val = reinterpret_cast<int32_t*>(isplit.untyped_data());
    int2* interactions_val = reinterpret_cast<int2*>(interactions.untyped_data());
    int* iminmax_val = reinterpret_cast<int*>(iminmax.untyped_data());
    ForcePot* fphi_val = reinterpret_cast<ForcePot*>(fphi->untyped_data());

    void* args[] = {
        &xm_val,
        &isplit_val,
        &interactions_val,
        &iminmax_val,
        &fphi_val,
        &interactions_per_block,
        &epsilon
    };
    cudaLaunchKernel((const void*)IlistForceAndPot, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistForceAndPotFFI, IlistForceAndPotFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xm
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // interactions
        .Arg<ffi::AnyBuffer>() // iminmax
        .Ret<ffi::AnyBuffer>() // fphi
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: BwdIlistForceAndPot                       */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error BwdIlistForceAndPotFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer gfphi,
    ffi::AnyBuffer xm,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer interactions,
    ffi::AnyBuffer iminmax,
    ffi::Result<ffi::AnyBuffer> gxm,
    size_t interactions_per_block,
    float epsilon,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(interactions.element_count()/2, interactions_per_block));
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(gxm->untyped_data(), 0, gxm->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float4* gfphi_val = reinterpret_cast<float4*>(gfphi.untyped_data());
    float4* xm_val = reinterpret_cast<float4*>(xm.untyped_data());
    int32_t* isplit_val = reinterpret_cast<int32_t*>(isplit.untyped_data());
    int2* interactions_val = reinterpret_cast<int2*>(interactions.untyped_data());
    int* iminmax_val = reinterpret_cast<int*>(iminmax.untyped_data());
    float4* gxm_val = reinterpret_cast<float4*>(gxm->untyped_data());

    void* args[] = {
        &gfphi_val,
        &xm_val,
        &isplit_val,
        &interactions_val,
        &iminmax_val,
        &gxm_val,
        &interactions_per_block,
        &epsilon
    };
    cudaLaunchKernel((const void*)BwdIlistForceAndPot, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BwdIlistForceAndPotFFI, BwdIlistForceAndPotFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // gfphi
        .Arg<ffi::AnyBuffer>() // xm
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // interactions
        .Arg<ffi::AnyBuffer>() // iminmax
        .Ret<ffi::AnyBuffer>() // gxm
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_forces, m) {
    m.def("ForceAndPotential", []() { return EncapsulateFfiCall(&ForceAndPotentialFFI); });
    m.def("BwdForceAndPotential", []() { return EncapsulateFfiCall(&BwdForceAndPotentialFFI); });
    m.def("GroupedForceAndPot", []() { return EncapsulateFfiCall(&GroupedForceAndPotFFI); });
    m.def("IlistForceAndPot", []() { return EncapsulateFfiCall(&IlistForceAndPotFFI); });
    m.def("BwdIlistForceAndPot", []() { return EncapsulateFfiCall(&BwdIlistForceAndPotFFI); });
}