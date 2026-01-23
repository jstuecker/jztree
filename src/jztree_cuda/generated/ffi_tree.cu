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
#include "../tree.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: FlagLeafBoundaries                        */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error FlagLeafBoundariesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer posz,
    ffi::AnyBuffer lvl_bound,
    ffi::AnyBuffer npart,
    ffi::Result<ffi::AnyBuffer> split_flags,
    int max_size,
    int scan_size,
    size_t block_size
) {
    int size_part = posz.element_count()/3;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_part+1, block_size));
    size_t smem = (block_size + 2*scan_size + 1) * sizeof(int32_t);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float3* posz_val = reinterpret_cast<float3*>(posz.untyped_data());
    int* lvl_bound_val = reinterpret_cast<int*>(lvl_bound.untyped_data());
    int* npart_val = reinterpret_cast<int*>(npart.untyped_data());
    int8_t* split_flags_val = reinterpret_cast<int8_t*>(split_flags->untyped_data());

    void* args[] = {
        &posz_val,
        &lvl_bound_val,
        &npart_val,
        &split_flags_val,
        &max_size,
        &size_part,
        &scan_size
    };
    cudaLaunchKernel((const void*)FlagLeafBoundaries, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlagLeafBoundariesFFI, FlagLeafBoundariesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // posz
        .Arg<ffi::AnyBuffer>() // lvl_bound
        .Arg<ffi::AnyBuffer>() // npart
        .Ret<ffi::AnyBuffer>() // split_flags
        .Attr<int>("max_size")
        .Attr<int>("scan_size")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: FindNodeBoundaries                        */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error FindNodeBoundariesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_in,
    ffi::AnyBuffer pos_boundary,
    ffi::AnyBuffer nleaves,
    ffi::Result<ffi::AnyBuffer> nodes_levels,
    ffi::Result<ffi::AnyBuffer> nodes_lbound,
    ffi::Result<ffi::AnyBuffer> nodes_rbound,
    size_t block_size
) {
    int size_nodes = nodes_levels->element_count();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_nodes, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float3* pos_in_val = reinterpret_cast<float3*>(pos_in.untyped_data());
    float3* pos_boundary_val = reinterpret_cast<float3*>(pos_boundary.untyped_data());
    int* nleaves_val = reinterpret_cast<int*>(nleaves.untyped_data());
    int32_t* nodes_levels_val = reinterpret_cast<int32_t*>(nodes_levels->untyped_data());
    int32_t* nodes_lbound_val = reinterpret_cast<int32_t*>(nodes_lbound->untyped_data());
    int32_t* nodes_rbound_val = reinterpret_cast<int32_t*>(nodes_rbound->untyped_data());

    void* args[] = {
        &pos_in_val,
        &pos_boundary_val,
        &nleaves_val,
        &nodes_levels_val,
        &nodes_lbound_val,
        &nodes_rbound_val,
        &size_nodes
    };
    cudaLaunchKernel((const void*)FindNodeBoundaries, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FindNodeBoundariesFFI, FindNodeBoundariesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos_in
        .Arg<ffi::AnyBuffer>() // pos_boundary
        .Arg<ffi::AnyBuffer>() // nleaves
        .Ret<ffi::AnyBuffer>() // nodes_levels
        .Ret<ffi::AnyBuffer>() // nodes_lbound
        .Ret<ffi::AnyBuffer>() // nodes_rbound
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: GetBoundaryExtendPerLevel                 */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error GetBoundaryExtendPerLevelFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_ref,
    ffi::AnyBuffer irange,
    ffi::AnyBuffer posz,
    ffi::Result<ffi::AnyBuffer> index_of_lvl,
    size_t block_size,
    bool left
) {
    int size = posz.element_count()/3;

    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<bool>;
    using TFunctionType = decltype(GetBoundaryExtendPerLevel<true>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{true}] = GetBoundaryExtendPerLevel<true>;
    instance_map[{false}] = GetBoundaryExtendPerLevel<false>;

    auto it = instance_map.find({left});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (left)"\
            " in GetBoundaryExtendPerLevelFFIHost -- Only supporting:\n"\
            "(true), (false)"
        );
    }

    TFunctionType* instance = it->second;

    // Now call our function
    std::string result = instance(
        stream,
        reinterpret_cast<float3*>(pos_ref.untyped_data()),
        reinterpret_cast<int*>(irange.untyped_data()),
        reinterpret_cast<float3*>(posz.untyped_data()),
        reinterpret_cast<int32_t*>(index_of_lvl->untyped_data()),
        size,
        block_size
    );
    // Check if the function returned an error string
    if (!result.empty()) {
        return ffi::Error::Internal(result);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GetBoundaryExtendPerLevelFFI, GetBoundaryExtendPerLevelFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos_ref
        .Arg<ffi::AnyBuffer>() // irange
        .Arg<ffi::AnyBuffer>() // posz
        .Ret<ffi::AnyBuffer>() // index_of_lvl
        .Attr<size_t>("block_size")
        .Attr<bool>("left"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: GetNodeGeometry                           */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error GetNodeGeometryFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos,
    ffi::AnyBuffer lbound,
    ffi::AnyBuffer rbound,
    ffi::AnyBuffer nnodes,
    ffi::Result<ffi::AnyBuffer> level,
    ffi::Result<ffi::AnyBuffer> center,
    ffi::Result<ffi::AnyBuffer> extent,
    size_t block_size
) {
    int size_nodes = level->element_count();
    int size_part = pos.element_count()/3;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_nodes, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float3* pos_val = reinterpret_cast<float3*>(pos.untyped_data());
    int* lbound_val = reinterpret_cast<int*>(lbound.untyped_data());
    int* rbound_val = reinterpret_cast<int*>(rbound.untyped_data());
    int* nnodes_val = reinterpret_cast<int*>(nnodes.untyped_data());
    int32_t* level_val = reinterpret_cast<int32_t*>(level->untyped_data());
    float3* center_val = reinterpret_cast<float3*>(center->untyped_data());
    float3* extent_val = reinterpret_cast<float3*>(extent->untyped_data());

    void* args[] = {
        &pos_val,
        &lbound_val,
        &rbound_val,
        &nnodes_val,
        &level_val,
        &center_val,
        &extent_val,
        &size_nodes,
        &size_part
    };
    cudaLaunchKernel((const void*)GetNodeGeometry, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GetNodeGeometryFFI, GetNodeGeometryFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos
        .Arg<ffi::AnyBuffer>() // lbound
        .Arg<ffi::AnyBuffer>() // rbound
        .Arg<ffi::AnyBuffer>() // nnodes
        .Ret<ffi::AnyBuffer>() // level
        .Ret<ffi::AnyBuffer>() // center
        .Ret<ffi::AnyBuffer>() // extent
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_tree, m) {
    m.def("FlagLeafBoundaries", []() { return EncapsulateFfiCall(&FlagLeafBoundariesFFI); });
    m.def("FindNodeBoundaries", []() { return EncapsulateFfiCall(&FindNodeBoundariesFFI); });
    m.def("GetBoundaryExtendPerLevel", []() { return EncapsulateFfiCall(&GetBoundaryExtendPerLevelFFI); });
    m.def("GetNodeGeometry", []() { return EncapsulateFfiCall(&GetNodeGeometryFFI); });
}