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
#include "../knn.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: IlistKNN                                  */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error IlistKNNFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xT,
    ffi::AnyBuffer xQ,
    ffi::AnyBuffer isplitT,
    ffi::AnyBuffer isplitQ,
    ffi::AnyBuffer ilist,
    ffi::AnyBuffer ir2list,
    ffi::AnyBuffer ilist_splitsQ,
    ffi::Result<ffi::AnyBuffer> knn,
    float boxsize,
    int k
) {
    dim3 blockDim(32);
    dim3 gridDim(isplitQ.element_count() - 1);
    size_t smem = blockDim.x * sizeof(PosId);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    PosR* xT_val = reinterpret_cast<PosR*>(xT.untyped_data());
    PosR* xQ_val = reinterpret_cast<PosR*>(xQ.untyped_data());
    int* isplitT_val = reinterpret_cast<int*>(isplitT.untyped_data());
    int* isplitQ_val = reinterpret_cast<int*>(isplitQ.untyped_data());
    int* ilist_val = reinterpret_cast<int*>(ilist.untyped_data());
    float* ir2list_val = reinterpret_cast<float*>(ir2list.untyped_data());
    int* ilist_splitsQ_val = reinterpret_cast<int*>(ilist_splitsQ.untyped_data());
    Neighbor* knn_val = reinterpret_cast<Neighbor*>(knn->untyped_data());

    void* args[] = {
        &xT_val,
        &xQ_val,
        &isplitT_val,
        &isplitQ_val,
        &ilist_val,
        &ir2list_val,
        &ilist_splitsQ_val,
        &knn_val,
        &boxsize
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(IlistKNN<4>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{4}] = IlistKNN<4>;
    instance_map[{8}] = IlistKNN<8>;
    instance_map[{12}] = IlistKNN<12>;
    instance_map[{16}] = IlistKNN<16>;
    instance_map[{32}] = IlistKNN<32>;
    instance_map[{64}] = IlistKNN<64>;

    auto it = instance_map.find({k});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (k)"\
            " in IlistKNNFFIHost -- Only supporting:\n"\
            "(4), (8), (12), (16), (32), (64)"
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
    IlistKNNFFI, IlistKNNFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xT
        .Arg<ffi::AnyBuffer>() // xQ
        .Arg<ffi::AnyBuffer>() // isplitT
        .Arg<ffi::AnyBuffer>() // isplitQ
        .Arg<ffi::AnyBuffer>() // ilist
        .Arg<ffi::AnyBuffer>() // ir2list
        .Arg<ffi::AnyBuffer>() // ilist_splitsQ
        .Ret<ffi::AnyBuffer>() // knn
        .Attr<float>("boxsize")
        .Attr<int>("k"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: KnnNode2Node                              */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error KnnNode2NodeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer parent_ilist_spl,
    ffi::AnyBuffer parent_ilist,
    ffi::AnyBuffer parent_ilist_r2,
    ffi::AnyBuffer parent_spl,
    ffi::AnyBuffer nodes,
    ffi::AnyBuffer nodes_npart,
    ffi::Result<ffi::AnyBuffer> rmax2,
    ffi::Result<ffi::AnyBuffer> node_ilist_splits,
    ffi::Result<ffi::AnyBuffer> node_ilist,
    ffi::Result<ffi::AnyBuffer> node_ilist_r2,
    int k,
    size_t blocksize_fill,
    size_t blocksize_sort,
    float rfac_maxbin,
    float boxsize
) {
    int nnodes = parent_spl.element_count() - 1;
    int nleaves = nodes_npart.element_count();
    size_t node_ilist_size = node_ilist->element_count();

    // Now call our function
    ffi::Error result = KnnNode2Node(
        stream,
        reinterpret_cast<int*>(parent_ilist_spl.untyped_data()),
        reinterpret_cast<int*>(parent_ilist.untyped_data()),
        reinterpret_cast<float*>(parent_ilist_r2.untyped_data()),
        reinterpret_cast<int*>(parent_spl.untyped_data()),
        reinterpret_cast<Node*>(nodes.untyped_data()),
        reinterpret_cast<int*>(nodes_npart.untyped_data()),
        reinterpret_cast<float*>(rmax2->untyped_data()),
        reinterpret_cast<int*>(node_ilist_splits->untyped_data()),
        reinterpret_cast<int*>(node_ilist->untyped_data()),
        reinterpret_cast<float*>(node_ilist_r2->untyped_data()),
        k,
        blocksize_fill,
        blocksize_sort,
        rfac_maxbin,
        boxsize,
        nnodes,
        nleaves,
        node_ilist_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KnnNode2NodeFFI, KnnNode2NodeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // parent_ilist_spl
        .Arg<ffi::AnyBuffer>() // parent_ilist
        .Arg<ffi::AnyBuffer>() // parent_ilist_r2
        .Arg<ffi::AnyBuffer>() // parent_spl
        .Arg<ffi::AnyBuffer>() // nodes
        .Arg<ffi::AnyBuffer>() // nodes_npart
        .Ret<ffi::AnyBuffer>() // rmax2
        .Ret<ffi::AnyBuffer>() // node_ilist_splits
        .Ret<ffi::AnyBuffer>() // node_ilist
        .Ret<ffi::AnyBuffer>() // node_ilist_r2
        .Attr<int>("k")
        .Attr<size_t>("blocksize_fill")
        .Attr<size_t>("blocksize_sort")
        .Attr<float>("rfac_maxbin")
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SegmentSort                               */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SegmentSortFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer key,
    ffi::AnyBuffer val,
    ffi::AnyBuffer isplit,
    ffi::Result<ffi::AnyBuffer> key_out,
    ffi::Result<ffi::AnyBuffer> val_out,
    size_t smem_size
) {
    int32_t nkeys = key.element_count();
    int32_t nsegs = isplit.element_count() - 1;

    // Now call our function
    ffi::Error result = SegmentSort(
        stream,
        reinterpret_cast<float*>(key.untyped_data()),
        reinterpret_cast<int32_t*>(val.untyped_data()),
        reinterpret_cast<int32_t*>(isplit.untyped_data()),
        reinterpret_cast<float*>(key_out->untyped_data()),
        reinterpret_cast<int32_t*>(val_out->untyped_data()),
        nkeys,
        nsegs,
        smem_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SegmentSortFFI, SegmentSortFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // key
        .Arg<ffi::AnyBuffer>() // val
        .Arg<ffi::AnyBuffer>() // isplit
        .Ret<ffi::AnyBuffer>() // key_out
        .Ret<ffi::AnyBuffer>() // val_out
        .Attr<size_t>("smem_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_knn, m) {
    m.def("IlistKNN", []() { return EncapsulateFfiCall(&IlistKNNFFI); });
    m.def("KnnNode2Node", []() { return EncapsulateFfiCall(&KnnNode2NodeFFI); });
    m.def("SegmentSort", []() { return EncapsulateFfiCall(&SegmentSortFFI); });
}