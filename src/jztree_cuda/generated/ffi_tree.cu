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

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: FlagLeafBoundaries                        */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error FlagLeafBoundariesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer posz,
    ffi::AnyBuffer lvl_bound,
    ffi::AnyBuffer npart,
    ffi::Result<ffi::AnyBuffer> split_flags,
    ffi::Result<ffi::AnyBuffer> lvl,
    int max_size,
    int scan_size,
    size_t block_size
) {
    int size_part = posz.dimensions()[0];
    int dim = posz.dimensions()[1];
    DT tvec = posz.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_part+1, block_size));
    size_t smem = (block_size + 2*scan_size + 1) * sizeof(int32_t);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* posz_arg = posz.untyped_data();
    void* lvl_bound_arg = lvl_bound.untyped_data();
    void* npart_arg = npart.untyped_data();
    void* split_flags_arg = split_flags->untyped_data();
    void* lvl_arg = lvl->untyped_data();
    void* args[] = {
        &posz_arg,
        &lvl_bound_arg,
        &npart_arg,
        &split_flags_arg,
        &lvl_arg,
        &max_size,
        &size_part,
        &scan_size
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, reinterpret_cast<TFunc>(&FlagLeafBoundaries<2, float>) },
        { {2, DT::F64}, reinterpret_cast<TFunc>(&FlagLeafBoundaries<2, double>) },
        { {3, DT::F32}, reinterpret_cast<TFunc>(&FlagLeafBoundaries<3, float>) },
        { {3, DT::F64}, reinterpret_cast<TFunc>(&FlagLeafBoundaries<3, double>) }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in FlagLeafBoundariesFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    const void* instance = it->second;

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
    FlagLeafBoundariesFFI, FlagLeafBoundariesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // posz
        .Arg<ffi::AnyBuffer>() // lvl_bound
        .Arg<ffi::AnyBuffer>() // npart
        .Ret<ffi::AnyBuffer>() // split_flags
        .Ret<ffi::AnyBuffer>() // lvl
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
    int lvl_max,
    int lvl_invalid,
    size_t block_size
) {
    int size_nodes = nodes_levels->element_count();
    int dim = pos_in.dimensions()[1];
    DT tvec = pos_in.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_nodes, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* pos_in_arg = pos_in.untyped_data();
    void* pos_boundary_arg = pos_boundary.untyped_data();
    void* nleaves_arg = nleaves.untyped_data();
    void* nodes_levels_arg = nodes_levels->untyped_data();
    void* nodes_lbound_arg = nodes_lbound->untyped_data();
    void* nodes_rbound_arg = nodes_rbound->untyped_data();
    void* args[] = {
        &pos_in_arg,
        &pos_boundary_arg,
        &nleaves_arg,
        &nodes_levels_arg,
        &nodes_lbound_arg,
        &nodes_rbound_arg,
        &size_nodes,
        &lvl_max,
        &lvl_invalid
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, reinterpret_cast<TFunc>(&FindNodeBoundaries<2, float>) },
        { {2, DT::F64}, reinterpret_cast<TFunc>(&FindNodeBoundaries<2, double>) },
        { {3, DT::F32}, reinterpret_cast<TFunc>(&FindNodeBoundaries<3, float>) },
        { {3, DT::F64}, reinterpret_cast<TFunc>(&FindNodeBoundaries<3, double>) }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in FindNodeBoundariesFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    const void* instance = it->second;

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
    FindNodeBoundariesFFI, FindNodeBoundariesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos_in
        .Arg<ffi::AnyBuffer>() // pos_boundary
        .Arg<ffi::AnyBuffer>() // nleaves
        .Ret<ffi::AnyBuffer>() // nodes_levels
        .Ret<ffi::AnyBuffer>() // nodes_lbound
        .Ret<ffi::AnyBuffer>() // nodes_rbound
        .Attr<int>("lvl_max")
        .Attr<int>("lvl_invalid")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: GetBoundaryExtendPerLevel                 */
/* ---------------------------------------------------------------------------------------------- */


using GetBoundaryExtendPerLevelDispatchFn = std::string (*) (cudaStream_t stream,
    const void* pos_ref,
    const void* irange,
    const void* posz,
    void* index_of_lvl,
    int size,
    size_t block_size,
    int lvl_min,
    int lvl_max
);
template<bool left, int dim, typename tvec>
static std::string GetBoundaryExtendPerLevelDispatchWrapper(cudaStream_t stream,
    const void* pos_ref,
    const void* irange,
    const void* posz,
    void* index_of_lvl,
    int size,
    size_t block_size,
    int lvl_min,
    int lvl_max
) {
    return GetBoundaryExtendPerLevel<left, dim, tvec> (stream,
        reinterpret_cast<const Vec<dim,tvec>*>(pos_ref),
        reinterpret_cast<const int*>(irange),
        reinterpret_cast<const Vec<dim,tvec>*>(posz),
        reinterpret_cast<int32_t*>(index_of_lvl),
        size,
        block_size,
        lvl_min,
        lvl_max
    );
}


ffi::Error GetBoundaryExtendPerLevelFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_ref,
    ffi::AnyBuffer irange,
    ffi::AnyBuffer posz,
    ffi::Result<ffi::AnyBuffer> index_of_lvl,
    size_t block_size,
    int lvl_min,
    int lvl_max,
    bool left
) {
    int size = posz.dimensions()[0];
    int dim = posz.dimensions()[1];
    DT tvec = posz.element_type();


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, DT>;
    using TFunc = GetBoundaryExtendPerLevelDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 2, DT::F32}, &GetBoundaryExtendPerLevelDispatchWrapper<true, 2, float> },
        { {true, 2, DT::F64}, &GetBoundaryExtendPerLevelDispatchWrapper<true, 2, double> },
        { {true, 3, DT::F32}, &GetBoundaryExtendPerLevelDispatchWrapper<true, 3, float> },
        { {true, 3, DT::F64}, &GetBoundaryExtendPerLevelDispatchWrapper<true, 3, double> },
        { {false, 2, DT::F32}, &GetBoundaryExtendPerLevelDispatchWrapper<false, 2, float> },
        { {false, 2, DT::F64}, &GetBoundaryExtendPerLevelDispatchWrapper<false, 2, double> },
        { {false, 3, DT::F32}, &GetBoundaryExtendPerLevelDispatchWrapper<false, 3, float> },
        { {false, 3, DT::F64}, &GetBoundaryExtendPerLevelDispatchWrapper<false, 3, double> }
    };

    const TTuple key = TTuple(left, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (left, dim, tvec)"\
            " in GetBoundaryExtendPerLevelFFIHost -- Only supporting:\n"\
            "(true, 2, float), (true, 2, double), (true, 3, float), (true, 3, double), (false, 2, float), (false, 2, double), (false, 3, float), (false, 3, double)"
        );
    }
    GetBoundaryExtendPerLevelDispatchFn instance = it->second;

    // Now call our function
    std::string result = instance(stream,
        pos_ref.untyped_data(),
        irange.untyped_data(),
        posz.untyped_data(),
        index_of_lvl->untyped_data(),
        size,
        block_size,
        lvl_min,
        lvl_max
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
        .Attr<int>("lvl_min")
        .Attr<int>("lvl_max")
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
    int lvl_invalid,
    uint32_t mode_flags,
    bool upper_extent,
    size_t block_size
) {
    int size_nodes = lbound.element_count();
    int size_part = pos.dimensions()[0];
    int dim = pos.dimensions()[1];
    DT tvec = pos.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_nodes, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* pos_arg = pos.untyped_data();
    void* lbound_arg = lbound.untyped_data();
    void* rbound_arg = rbound.untyped_data();
    void* nnodes_arg = nnodes.untyped_data();
    void* level_arg = level->untyped_data();
    void* center_arg = center->untyped_data();
    void* extent_arg = extent->untyped_data();
    void* args[] = {
        &pos_arg,
        &lbound_arg,
        &rbound_arg,
        &nnodes_arg,
        &level_arg,
        &center_arg,
        &extent_arg,
        &size_nodes,
        &size_part,
        &lvl_invalid,
        &mode_flags,
        &upper_extent
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, reinterpret_cast<TFunc>(&GetNodeGeometry<2, float>) },
        { {2, DT::F64}, reinterpret_cast<TFunc>(&GetNodeGeometry<2, double>) },
        { {3, DT::F32}, reinterpret_cast<TFunc>(&GetNodeGeometry<3, float>) },
        { {3, DT::F64}, reinterpret_cast<TFunc>(&GetNodeGeometry<3, double>) }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in GetNodeGeometryFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    const void* instance = it->second;

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
        .Attr<int>("lvl_invalid")
        .Attr<uint32_t>("mode_flags")
        .Attr<bool>("upper_extent")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: CenterOfMass                              */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error CenterOfMassFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer pos,
    ffi::AnyBuffer mass,
    ffi::Result<ffi::AnyBuffer> com_out,
    bool kahan,
    size_t block_size
) {
    int nnodes = isplit.element_count() - 1;
    int dim = pos.dimensions()[1];
    DT tvec = pos.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* isplit_arg = isplit.untyped_data();
    void* pos_arg = pos.untyped_data();
    void* mass_arg = mass.untyped_data();
    void* com_out_arg = com_out->untyped_data();
    void* args[] = {
        &isplit_arg,
        &pos_arg,
        &mass_arg,
        &com_out_arg,
        &nnodes,
        &kahan
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, reinterpret_cast<TFunc>(&CenterOfMass<2, float>) },
        { {2, DT::F64}, reinterpret_cast<TFunc>(&CenterOfMass<2, double>) },
        { {3, DT::F32}, reinterpret_cast<TFunc>(&CenterOfMass<3, float>) },
        { {3, DT::F64}, reinterpret_cast<TFunc>(&CenterOfMass<3, double>) }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in CenterOfMassFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    const void* instance = it->second;

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
    CenterOfMassFFI, CenterOfMassFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // pos
        .Arg<ffi::AnyBuffer>() // mass
        .Ret<ffi::AnyBuffer>() // com_out
        .Attr<bool>("kahan")
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
    m.def("CenterOfMass", []() { return EncapsulateFfiCall(&CenterOfMassFFI); });
}