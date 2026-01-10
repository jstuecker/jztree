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
    ffi::AnyBuffer node_igroup,
    ffi::AnyBuffer node_lvl,
    ffi::AnyBuffer isplit,
    ffi::Result<ffi::AnyBuffer> leaf_igroup,
    float r2link,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(node_igroup.element_count());
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(leaf_igroup->untyped_data(), 0, leaf_igroup->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* node_igroup_val = reinterpret_cast<int*>(node_igroup.untyped_data());
    int* node_lvl_val = reinterpret_cast<int*>(node_lvl.untyped_data());
    int* isplit_val = reinterpret_cast<int*>(isplit.untyped_data());
    int* leaf_igroup_val = reinterpret_cast<int*>(leaf_igroup->untyped_data());

    void* args[] = {
        &node_igroup_val,
        &node_lvl_val,
        &isplit_val,
        &leaf_igroup_val,
        &r2link
    };
    cudaLaunchKernel((const void*)NodeToChildLabel, gridDim, blockDim, args, smem, stream);

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
        .Arg<ffi::AnyBuffer>() // node_igroup
        .Arg<ffi::AnyBuffer>() // node_lvl
        .Arg<ffi::AnyBuffer>() // isplit
        .Ret<ffi::AnyBuffer>() // leaf_igroup
        .Attr<float>("r2link")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: NodeFofAndIlist                           */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error NodeFofAndIlistFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_ilist_splits,
    ffi::AnyBuffer node_ilist,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer leaves,
    ffi::AnyBuffer leaf_igroup_in,
    ffi::Result<ffi::AnyBuffer> leaf_igroup,
    ffi::Result<ffi::AnyBuffer> ilist_out_splits,
    ffi::Result<ffi::AnyBuffer> ilist_out,
    float r2link,
    float boxsize,
    int block_size
) {
    int nnodes = isplit.element_count() - 1;
    int nleaves = leaf_igroup_in.element_count();
    size_t ilist_out_size = ilist_out->element_count();

    // Now call our function
    ffi::Error result = NodeFofAndIlist(
        stream,
        reinterpret_cast<int*>(node_ilist_splits.untyped_data()),
        reinterpret_cast<int*>(node_ilist.untyped_data()),
        reinterpret_cast<int*>(isplit.untyped_data()),
        reinterpret_cast<Node*>(leaves.untyped_data()),
        reinterpret_cast<int*>(leaf_igroup_in.untyped_data()),
        reinterpret_cast<int*>(leaf_igroup->untyped_data()),
        reinterpret_cast<int*>(ilist_out_splits->untyped_data()),
        reinterpret_cast<int*>(ilist_out->untyped_data()),
        r2link,
        boxsize,
        nnodes,
        nleaves,
        ilist_out_size,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NodeFofAndIlistFFI, NodeFofAndIlistFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_ilist_splits
        .Arg<ffi::AnyBuffer>() // node_ilist
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // leaves
        .Arg<ffi::AnyBuffer>() // leaf_igroup_in
        .Ret<ffi::AnyBuffer>() // leaf_igroup
        .Ret<ffi::AnyBuffer>() // ilist_out_splits
        .Ret<ffi::AnyBuffer>() // ilist_out
        .Attr<float>("r2link")
        .Attr<float>("boxsize")
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: ParticleFof                               */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error ParticleFofFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_ilist_splits,
    ffi::AnyBuffer node_ilist,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer pos,
    ffi::AnyBuffer particle_igroup_in,
    ffi::Result<ffi::AnyBuffer> particle_igroup,
    float r2link,
    float boxsize,
    int block_size
) {
    int nnodes = isplit.element_count() - 1;
    int npart = particle_igroup->element_count();

    // Now call our function
    ffi::Error result = ParticleFof(
        stream,
        reinterpret_cast<int*>(node_ilist_splits.untyped_data()),
        reinterpret_cast<int*>(node_ilist.untyped_data()),
        reinterpret_cast<int*>(isplit.untyped_data()),
        reinterpret_cast<float3*>(pos.untyped_data()),
        reinterpret_cast<int*>(particle_igroup_in.untyped_data()),
        reinterpret_cast<int*>(particle_igroup->untyped_data()),
        r2link,
        boxsize,
        nnodes,
        npart,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ParticleFofFFI, ParticleFofFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_ilist_splits
        .Arg<ffi::AnyBuffer>() // node_ilist
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // pos
        .Arg<ffi::AnyBuffer>() // particle_igroup_in
        .Ret<ffi::AnyBuffer>() // particle_igroup
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
    ffi::Error result = InsertLinks(
        stream,
        reinterpret_cast<int*>(igroup_in.untyped_data()),
        reinterpret_cast<int*>(igroupLinkA.untyped_data()),
        reinterpret_cast<int*>(igroupLinkB.untyped_data()),
        reinterpret_cast<int*>(num_links.untyped_data()),
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
    m.def("NodeFofAndIlist", []() { return EncapsulateFfiCall(&NodeFofAndIlistFFI); });
    m.def("ParticleFof", []() { return EncapsulateFfiCall(&ParticleFofFFI); });
    m.def("InsertLinks", []() { return EncapsulateFfiCall(&InsertLinksFFI); });
}