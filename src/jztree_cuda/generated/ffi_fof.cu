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
#include "../fof.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: NodeToChildLabel                          */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error NodeToChildLabelFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer parent_igroup,
    ffi::AnyBuffer parent_is_local,
    ffi::AnyBuffer parent_lvl,
    ffi::AnyBuffer parent_spl,
    ffi::Result<ffi::AnyBuffer> node_igroup,
    float r2link,
    size_t block_size
) {
    int size_parent = parent_igroup.element_count();
    dim3 blockDim(block_size);
    dim3 gridDim(parent_igroup.element_count());
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(node_igroup->untyped_data(), 0, node_igroup->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* parent_igroup_arg = parent_igroup.untyped_data();
    void* parent_is_local_arg = parent_is_local.untyped_data();
    void* parent_lvl_arg = parent_lvl.untyped_data();
    void* parent_spl_arg = parent_spl.untyped_data();
    void* node_igroup_arg = node_igroup->untyped_data();
    void* args[] = {
        &parent_igroup_arg,
        &parent_is_local_arg,
        &parent_lvl_arg,
        &parent_spl_arg,
        &node_igroup_arg,
        &size_parent,
        &r2link
    };
    const void* instance = (const void*)NodeToChildLabel;

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
    NodeToChildLabelFFI, NodeToChildLabelFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // parent_igroup
        .Arg<ffi::AnyBuffer>() // parent_is_local
        .Arg<ffi::AnyBuffer>() // parent_lvl
        .Arg<ffi::AnyBuffer>() // parent_spl
        .Ret<ffi::AnyBuffer>() // node_igroup
        .Attr<float>("r2link")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: FofNode2Node                              */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error FofNode2NodeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer parent_ilist_spl,
    ffi::AnyBuffer parent_ilist,
    ffi::AnyBuffer parent_spl,
    ffi::AnyBuffer nodes,
    ffi::AnyBuffer node_igroup_in,
    ffi::Result<ffi::AnyBuffer> node_igroup,
    ffi::Result<ffi::AnyBuffer> node_ilist_spl,
    ffi::Result<ffi::AnyBuffer> node_ilist,
    float r2link,
    float boxsize,
    int block_size
) {
    int size_parent = parent_spl.element_count() - 1;
    int size_node = node_igroup->element_count();
    size_t size_node_ilist = node_ilist->element_count();

    // Now call our function
    ffi::Error result = FofNode2Node(stream,
        reinterpret_cast<const int*>(parent_ilist_spl.untyped_data()),
        reinterpret_cast<const int*>(parent_ilist.untyped_data()),
        reinterpret_cast<const int*>(parent_spl.untyped_data()),
        reinterpret_cast<const Node*>(nodes.untyped_data()),
        reinterpret_cast<const int*>(node_igroup_in.untyped_data()),
        reinterpret_cast<int*>(node_igroup->untyped_data()),
        reinterpret_cast<int*>(node_ilist_spl->untyped_data()),
        reinterpret_cast<int*>(node_ilist->untyped_data()),
        r2link,
        boxsize,
        size_parent,
        size_node,
        size_node_ilist,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FofNode2NodeFFI, FofNode2NodeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // parent_ilist_spl
        .Arg<ffi::AnyBuffer>() // parent_ilist
        .Arg<ffi::AnyBuffer>() // parent_spl
        .Arg<ffi::AnyBuffer>() // nodes
        .Arg<ffi::AnyBuffer>() // node_igroup_in
        .Ret<ffi::AnyBuffer>() // node_igroup
        .Ret<ffi::AnyBuffer>() // node_ilist_spl
        .Ret<ffi::AnyBuffer>() // node_ilist
        .Attr<float>("r2link")
        .Attr<float>("boxsize")
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: FofLeaf2Leaf                              */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error FofLeaf2LeafFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer ilist_spl,
    ffi::AnyBuffer ilist,
    ffi::AnyBuffer spl,
    ffi::AnyBuffer pos,
    ffi::AnyBuffer part_igroup_in,
    ffi::Result<ffi::AnyBuffer> part_igroup,
    float r2link,
    float boxsize,
    int block_size
) {
    int size_leaves = spl.element_count() - 1;
    int size_part = part_igroup->element_count();

    // Now call our function
    ffi::Error result = FofLeaf2Leaf(stream,
        reinterpret_cast<const int*>(ilist_spl.untyped_data()),
        reinterpret_cast<const int*>(ilist.untyped_data()),
        reinterpret_cast<const int*>(spl.untyped_data()),
        reinterpret_cast<const float3*>(pos.untyped_data()),
        reinterpret_cast<const int*>(part_igroup_in.untyped_data()),
        reinterpret_cast<int*>(part_igroup->untyped_data()),
        r2link,
        boxsize,
        size_leaves,
        size_part,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FofLeaf2LeafFFI, FofLeaf2LeafFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // ilist_spl
        .Arg<ffi::AnyBuffer>() // ilist
        .Arg<ffi::AnyBuffer>() // spl
        .Arg<ffi::AnyBuffer>() // pos
        .Arg<ffi::AnyBuffer>() // part_igroup_in
        .Ret<ffi::AnyBuffer>() // part_igroup
        .Attr<float>("r2link")
        .Attr<float>("boxsize")
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: InsertLinks                               */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error InsertLinksFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer igroup_in,
    ffi::AnyBuffer igroupLinkA,
    ffi::AnyBuffer igroupLinkB,
    ffi::AnyBuffer num_links,
    ffi::Result<ffi::AnyBuffer> igroup,
    int block_size
) {
    int size_links = igroupLinkA.element_count();
    int size_groups = igroup_in.element_count();

    // Now call our function
    ffi::Error result = InsertLinks(stream,
        reinterpret_cast<const int*>(igroup_in.untyped_data()),
        reinterpret_cast<const int*>(igroupLinkA.untyped_data()),
        reinterpret_cast<const int*>(igroupLinkB.untyped_data()),
        reinterpret_cast<const int*>(num_links.untyped_data()),
        reinterpret_cast<int*>(igroup->untyped_data()),
        size_links,
        size_groups,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    InsertLinksFFI, InsertLinksFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // igroup_in
        .Arg<ffi::AnyBuffer>() // igroupLinkA
        .Arg<ffi::AnyBuffer>() // igroupLinkB
        .Arg<ffi::AnyBuffer>() // num_links
        .Ret<ffi::AnyBuffer>() // igroup
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fof, m) {
    m.def("NodeToChildLabel", []() { return EncapsulateFfiCall(&NodeToChildLabelFFI); });
    m.def("FofNode2Node", []() { return EncapsulateFfiCall(&FofNode2NodeFFI); });
    m.def("FofLeaf2Leaf", []() { return EncapsulateFfiCall(&FofLeaf2LeafFFI); });
    m.def("InsertLinks", []() { return EncapsulateFfiCall(&InsertLinksFFI); });
}