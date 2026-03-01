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

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: KnnLeaf2Leaf                              */
/* ---------------------------------------------------------------------------------------------- */


using KnnLeaf2LeafDispatchFn = std::string (*) (cudaStream_t stream,
    const void* ilist_spl,
    const void* ilist_iother,
    const void* ilist_r2,
    const void* splT,
    const void* xT,
    const void* splQ,
    const void* xQ,
    void* knn_rad,
    void* knn_id,
    void* rinfo_buf,
    int k,
    float boxsize,
    size_t size_leaves_query,
    size_t size_part_query
);
template<int dim, typename tvec>
static std::string KnnLeaf2LeafDispatchWrapper(cudaStream_t stream,
    const void* ilist_spl,
    const void* ilist_iother,
    const void* ilist_r2,
    const void* splT,
    const void* xT,
    const void* splQ,
    const void* xQ,
    void* knn_rad,
    void* knn_id,
    void* rinfo_buf,
    int k,
    float boxsize,
    size_t size_leaves_query,
    size_t size_part_query
) {
    return KnnLeaf2Leaf<dim, tvec> (stream,
        reinterpret_cast<const int*>(ilist_spl),
        reinterpret_cast<const int*>(ilist_iother),
        reinterpret_cast<const float*>(ilist_r2),
        reinterpret_cast<const int*>(splT),
        reinterpret_cast<const Vec<dim,tvec>*>(xT),
        reinterpret_cast<const int*>(splQ),
        reinterpret_cast<const Vec<dim,tvec>*>(xQ),
        reinterpret_cast<tvec*>(knn_rad),
        reinterpret_cast<int*>(knn_id),
        reinterpret_cast<void*>(rinfo_buf),
        k,
        boxsize,
        size_leaves_query,
        size_part_query
    );
}


ffi::Error KnnLeaf2LeafFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer ilist_spl,
    ffi::AnyBuffer ilist_iother,
    ffi::AnyBuffer ilist_r2,
    ffi::AnyBuffer splT,
    ffi::AnyBuffer xT,
    ffi::AnyBuffer splQ,
    ffi::AnyBuffer xQ,
    ffi::Result<ffi::AnyBuffer> knn_rad,
    ffi::Result<ffi::AnyBuffer> knn_id,
    ffi::Result<ffi::AnyBuffer> rinfo_buf,
    int k,
    float boxsize
) {
    size_t size_leaves_query = splQ.element_count() - 1;
    size_t size_part_query = xQ.dimensions()[0];
    int dim = xT.dimensions()[1];
    DT tvec = xT.element_type();


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = KnnLeaf2LeafDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, &KnnLeaf2LeafDispatchWrapper<2, float> },
        { {2, DT::F64}, &KnnLeaf2LeafDispatchWrapper<2, double> },
        { {3, DT::F32}, &KnnLeaf2LeafDispatchWrapper<3, float> },
        { {3, DT::F64}, &KnnLeaf2LeafDispatchWrapper<3, double> }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in KnnLeaf2LeafFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    KnnLeaf2LeafDispatchFn instance = it->second;

    // Now call our function
    std::string result = instance(stream,
        ilist_spl.untyped_data(),
        ilist_iother.untyped_data(),
        ilist_r2.untyped_data(),
        splT.untyped_data(),
        xT.untyped_data(),
        splQ.untyped_data(),
        xQ.untyped_data(),
        knn_rad->untyped_data(),
        knn_id->untyped_data(),
        rinfo_buf->untyped_data(),
        k,
        boxsize,
        size_leaves_query,
        size_part_query
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
    KnnLeaf2LeafFFI, KnnLeaf2LeafFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // ilist_spl
        .Arg<ffi::AnyBuffer>() // ilist_iother
        .Arg<ffi::AnyBuffer>() // ilist_r2
        .Arg<ffi::AnyBuffer>() // splT
        .Arg<ffi::AnyBuffer>() // xT
        .Arg<ffi::AnyBuffer>() // splQ
        .Arg<ffi::AnyBuffer>() // xQ
        .Ret<ffi::AnyBuffer>() // knn_rad
        .Ret<ffi::AnyBuffer>() // knn_id
        .Ret<ffi::AnyBuffer>() // rinfo_buf
        .Attr<int>("k")
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: KnnNode2Node                              */
/* ---------------------------------------------------------------------------------------------- */


using KnnNode2NodeDispatchFn = std::string (*) (cudaStream_t stream,
    const void* parent_ilist_spl,
    const void* parent_ilist_ioth,
    const void* parent_ilist_r2,
    const void* parent_spl,
    const void* nodes,
    const void* nodes_npart,
    void* node_rmax2,
    void* node_ilist_spl,
    void* node_ilist_ioth,
    void* node_ilist_r2,
    int k,
    size_t blocksize_fill,
    size_t blocksize_sort,
    float boxsize,
    int size_parents,
    int size_nodes,
    size_t node_ilist_size
);
template<int dim, typename tvec>
static std::string KnnNode2NodeDispatchWrapper(cudaStream_t stream,
    const void* parent_ilist_spl,
    const void* parent_ilist_ioth,
    const void* parent_ilist_r2,
    const void* parent_spl,
    const void* nodes,
    const void* nodes_npart,
    void* node_rmax2,
    void* node_ilist_spl,
    void* node_ilist_ioth,
    void* node_ilist_r2,
    int k,
    size_t blocksize_fill,
    size_t blocksize_sort,
    float boxsize,
    int size_parents,
    int size_nodes,
    size_t node_ilist_size
) {
    return KnnNode2Node<dim, tvec> (stream,
        reinterpret_cast<const int32_t*>(parent_ilist_spl),
        reinterpret_cast<const int32_t*>(parent_ilist_ioth),
        reinterpret_cast<const float*>(parent_ilist_r2),
        reinterpret_cast<const int32_t*>(parent_spl),
        reinterpret_cast<const Node<dim,tvec>*>(nodes),
        reinterpret_cast<const int32_t*>(nodes_npart),
        reinterpret_cast<float*>(node_rmax2),
        reinterpret_cast<int32_t*>(node_ilist_spl),
        reinterpret_cast<int32_t*>(node_ilist_ioth),
        reinterpret_cast<float*>(node_ilist_r2),
        k,
        blocksize_fill,
        blocksize_sort,
        boxsize,
        size_parents,
        size_nodes,
        node_ilist_size
    );
}


ffi::Error KnnNode2NodeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer parent_ilist_spl,
    ffi::AnyBuffer parent_ilist_ioth,
    ffi::AnyBuffer parent_ilist_r2,
    ffi::AnyBuffer parent_spl,
    ffi::AnyBuffer nodes,
    ffi::AnyBuffer nodes_npart,
    ffi::Result<ffi::AnyBuffer> node_rmax2,
    ffi::Result<ffi::AnyBuffer> node_ilist_spl,
    ffi::Result<ffi::AnyBuffer> node_ilist_ioth,
    ffi::Result<ffi::AnyBuffer> node_ilist_r2,
    int k,
    size_t blocksize_fill,
    size_t blocksize_sort,
    float boxsize,
    int dim
) {
    int size_parents = parent_spl.element_count() - 1;
    int size_nodes = nodes_npart.element_count();
    size_t node_ilist_size = node_ilist_ioth->element_count();
    DT tvec = nodes.element_type();


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = KnnNode2NodeDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, &KnnNode2NodeDispatchWrapper<2, float> },
        { {2, DT::F64}, &KnnNode2NodeDispatchWrapper<2, double> },
        { {3, DT::F32}, &KnnNode2NodeDispatchWrapper<3, float> },
        { {3, DT::F64}, &KnnNode2NodeDispatchWrapper<3, double> }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in KnnNode2NodeFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    KnnNode2NodeDispatchFn instance = it->second;

    // Now call our function
    std::string result = instance(stream,
        parent_ilist_spl.untyped_data(),
        parent_ilist_ioth.untyped_data(),
        parent_ilist_r2.untyped_data(),
        parent_spl.untyped_data(),
        nodes.untyped_data(),
        nodes_npart.untyped_data(),
        node_rmax2->untyped_data(),
        node_ilist_spl->untyped_data(),
        node_ilist_ioth->untyped_data(),
        node_ilist_r2->untyped_data(),
        k,
        blocksize_fill,
        blocksize_sort,
        boxsize,
        size_parents,
        size_nodes,
        node_ilist_size
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
    KnnNode2NodeFFI, KnnNode2NodeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // parent_ilist_spl
        .Arg<ffi::AnyBuffer>() // parent_ilist_ioth
        .Arg<ffi::AnyBuffer>() // parent_ilist_r2
        .Arg<ffi::AnyBuffer>() // parent_spl
        .Arg<ffi::AnyBuffer>() // nodes
        .Arg<ffi::AnyBuffer>() // nodes_npart
        .Ret<ffi::AnyBuffer>() // node_rmax2
        .Ret<ffi::AnyBuffer>() // node_ilist_spl
        .Ret<ffi::AnyBuffer>() // node_ilist_ioth
        .Ret<ffi::AnyBuffer>() // node_ilist_r2
        .Attr<int>("k")
        .Attr<size_t>("blocksize_fill")
        .Attr<size_t>("blocksize_sort")
        .Attr<float>("boxsize")
        .Attr<int>("dim"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SegmentSort                               */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error SegmentSortFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer spl,
    ffi::AnyBuffer key,
    ffi::AnyBuffer val,
    ffi::Result<ffi::AnyBuffer> key_out,
    ffi::Result<ffi::AnyBuffer> val_out,
    size_t smem_size
) {
    int32_t size_segs = spl.element_count() - 1;
    int32_t size_keys = key.element_count();

    // Now call our function
    std::string result = SegmentSort(stream,
        reinterpret_cast<const int32_t*>(spl.untyped_data()),
        reinterpret_cast<const float*>(key.untyped_data()),
        reinterpret_cast<const int32_t*>(val.untyped_data()),
        reinterpret_cast<float*>(key_out->untyped_data()),
        reinterpret_cast<int32_t*>(val_out->untyped_data()),
        size_segs,
        size_keys,
        smem_size
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
    SegmentSortFFI, SegmentSortFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // spl
        .Arg<ffi::AnyBuffer>() // key
        .Arg<ffi::AnyBuffer>() // val
        .Ret<ffi::AnyBuffer>() // key_out
        .Ret<ffi::AnyBuffer>() // val_out
        .Attr<size_t>("smem_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_knn, m) {
    m.def("KnnLeaf2Leaf", []() { return EncapsulateFfiCall(&KnnLeaf2LeafFFI); });
    m.def("KnnNode2Node", []() { return EncapsulateFfiCall(&KnnNode2NodeFFI); });
    m.def("SegmentSort", []() { return EncapsulateFfiCall(&SegmentSortFFI); });
}