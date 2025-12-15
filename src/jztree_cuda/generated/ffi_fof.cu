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
/*                             FFI call to CUDA kernel: NodeFofAndIlist                           */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error NodeFofAndIlistFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_igroup,
    ffi::AnyBuffer node_ilist_splits,
    ffi::AnyBuffer node_ilist,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer leaves,
    ffi::Result<ffi::AnyBuffer> leaf_igroup,
    ffi::Result<ffi::AnyBuffer> ilist_out_splits,
    ffi::Result<ffi::AnyBuffer> ilist_out,
    float r2link,
    float boxsize,
    int block_size
) {
    int nnodes = isplit.element_count() - 1;
    int nleaves = leaf_igroup->element_count();
    size_t ilist_out_size = ilist_out->element_count();

    // Now call our function
    ffi::Error result = NodeFofAndIlist(
        stream,
        reinterpret_cast<int*>(node_igroup.untyped_data()),
        reinterpret_cast<int*>(node_ilist_splits.untyped_data()),
        reinterpret_cast<int*>(node_ilist.untyped_data()),
        reinterpret_cast<int*>(isplit.untyped_data()),
        reinterpret_cast<Node*>(leaves.untyped_data()),
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
        .Arg<ffi::AnyBuffer>() // node_igroup
        .Arg<ffi::AnyBuffer>() // node_ilist_splits
        .Arg<ffi::AnyBuffer>() // node_ilist
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // leaves
        .Ret<ffi::AnyBuffer>() // leaf_igroup
        .Ret<ffi::AnyBuffer>() // ilist_out_splits
        .Ret<ffi::AnyBuffer>() // ilist_out
        .Attr<float>("r2link")
        .Attr<float>("boxsize")
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fof, m) {
    m.def("NodeFofAndIlist", []() { return EncapsulateFfiCall(&NodeFofAndIlistFFI); });
}