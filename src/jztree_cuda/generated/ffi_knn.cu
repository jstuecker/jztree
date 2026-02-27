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


ffi::Error KnnLeaf2LeafFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer ilist_spl,
    ffi::AnyBuffer ilist,
    ffi::AnyBuffer ilist_r2,
    ffi::AnyBuffer splT,
    ffi::AnyBuffer xT,
    ffi::AnyBuffer splQ,
    ffi::AnyBuffer xQ,
    ffi::Result<ffi::AnyBuffer> knn_rad,
    ffi::Result<ffi::AnyBuffer> knn_id,
    float boxsize,
    int k
) {
    int dim = xT.dimensions()[1];
    DT tvec = xT.element_type();
    dim3 blockDim(32);
    dim3 gridDim(splQ.element_count() - 1);
    size_t smem = blockDim.x * (dim*ffi::ByteWidth(xT.element_type()) + sizeof(int32_t));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* ilist_spl_arg = ilist_spl.untyped_data();
    void* ilist_arg = ilist.untyped_data();
    void* ilist_r2_arg = ilist_r2.untyped_data();
    void* splT_arg = splT.untyped_data();
    void* xT_arg = xT.untyped_data();
    void* splQ_arg = splQ.untyped_data();
    void* xQ_arg = xQ.untyped_data();
    void* knn_rad_arg = knn_rad->untyped_data();
    void* knn_id_arg = knn_id->untyped_data();
    void* args[] = {
        &ilist_spl_arg,
        &ilist_arg,
        &ilist_r2_arg,
        &splT_arg,
        &xT_arg,
        &splQ_arg,
        &xQ_arg,
        &knn_rad_arg,
        &knn_id_arg,
        &boxsize
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {4, 2, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<4, 2, float>) },
        { {4, 2, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<4, 2, double>) },
        { {4, 3, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<4, 3, float>) },
        { {4, 3, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<4, 3, double>) },
        { {8, 2, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<8, 2, float>) },
        { {8, 2, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<8, 2, double>) },
        { {8, 3, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<8, 3, float>) },
        { {8, 3, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<8, 3, double>) },
        { {12, 2, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<12, 2, float>) },
        { {12, 2, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<12, 2, double>) },
        { {12, 3, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<12, 3, float>) },
        { {12, 3, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<12, 3, double>) },
        { {16, 2, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<16, 2, float>) },
        { {16, 2, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<16, 2, double>) },
        { {16, 3, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<16, 3, float>) },
        { {16, 3, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<16, 3, double>) },
        { {32, 2, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<32, 2, float>) },
        { {32, 2, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<32, 2, double>) },
        { {32, 3, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<32, 3, float>) },
        { {32, 3, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<32, 3, double>) },
        { {64, 2, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<64, 2, float>) },
        { {64, 2, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<64, 2, double>) },
        { {64, 3, DT::F32}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<64, 3, float>) },
        { {64, 3, DT::F64}, reinterpret_cast<TFunc>(&KnnLeaf2Leaf<64, 3, double>) }
    };

    const TTuple key = TTuple(k, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (k, dim, tvec)"\
            " in KnnLeaf2LeafFFIHost -- Only supporting:\n"\
            "(4, 2, float), (4, 2, double), (4, 3, float), (4, 3, double), (8, 2, float), (8, 2, double), (8, 3, float), (8, 3, double), (12, 2, float), (12, 2, double), (12, 3, float), (12, 3, double), (16, 2, float), (16, 2, double), (16, 3, float), (16, 3, double), (32, 2, float), (32, 2, double), (32, 3, float), (32, 3, double), (64, 2, float), (64, 2, double), (64, 3, float), (64, 3, double)"
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
    KnnLeaf2LeafFFI, KnnLeaf2LeafFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // ilist_spl
        .Arg<ffi::AnyBuffer>() // ilist
        .Arg<ffi::AnyBuffer>() // ilist_r2
        .Arg<ffi::AnyBuffer>() // splT
        .Arg<ffi::AnyBuffer>() // xT
        .Arg<ffi::AnyBuffer>() // splQ
        .Arg<ffi::AnyBuffer>() // xQ
        .Ret<ffi::AnyBuffer>() // knn_rad
        .Ret<ffi::AnyBuffer>() // knn_id
        .Attr<float>("boxsize")
        .Attr<int>("k"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: KnnNode2Node                              */
/* ---------------------------------------------------------------------------------------------- */


using KnnNode2NodeDispatchFn = ffi::Error (*) (cudaStream_t stream,
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
static ffi::Error KnnNode2NodeDispatchWrapper(cudaStream_t stream,
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
    ffi::Error result = instance(stream,
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
    ffi::Error result = SegmentSort(stream,
        reinterpret_cast<const int32_t*>(spl.untyped_data()),
        reinterpret_cast<const float*>(key.untyped_data()),
        reinterpret_cast<const int32_t*>(val.untyped_data()),
        reinterpret_cast<float*>(key_out->untyped_data()),
        reinterpret_cast<int32_t*>(val_out->untyped_data()),
        size_segs,
        size_keys,
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