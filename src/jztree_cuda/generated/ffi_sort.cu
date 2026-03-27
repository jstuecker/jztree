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
#include "../sort.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: PosZorderSort                             */
/* ---------------------------------------------------------------------------------------------- */


using PosZorderSortDispatchFn = std::string (*) (cudaStream_t stream,
    const void* pos_in,
    void* pos_out,
    void* id_out,
    void* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
);
template<int dim, typename tvec>
static std::string PosZorderSortDispatchWrapper(cudaStream_t stream,
    const void* pos_in,
    void* pos_out,
    void* id_out,
    void* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
) {
    return PosZorderSort<dim, tvec> (stream,
        reinterpret_cast<const Vec<dim, tvec>*>(pos_in),
        reinterpret_cast<Vec<dim, tvec>*>(pos_out),
        reinterpret_cast<same_width_int<tvec>*>(id_out),
        reinterpret_cast<int*>(tmp_buffer),
        size,
        tmp_bytes,
        block_size
    );
}


ffi::Error PosZorderSortFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_in,
    ffi::Result<ffi::AnyBuffer> pos_out,
    ffi::Result<ffi::AnyBuffer> id_out,
    ffi::Result<ffi::AnyBuffer> tmp_buffer,
    size_t block_size
) {
    size_t size = pos_in.dimensions()[0];
    size_t tmp_bytes = tmp_buffer->size_bytes();
    int dim = pos_in.dimensions()[1];
    DT tvec = pos_in.element_type();


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = PosZorderSortDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, &PosZorderSortDispatchWrapper<2, float> },
        { {2, DT::F64}, &PosZorderSortDispatchWrapper<2, double> },
        { {2, DT::S32}, &PosZorderSortDispatchWrapper<2, int32_t> },
        { {2, DT::S64}, &PosZorderSortDispatchWrapper<2, int64_t> },
        { {3, DT::F32}, &PosZorderSortDispatchWrapper<3, float> },
        { {3, DT::F64}, &PosZorderSortDispatchWrapper<3, double> },
        { {3, DT::S32}, &PosZorderSortDispatchWrapper<3, int32_t> },
        { {3, DT::S64}, &PosZorderSortDispatchWrapper<3, int64_t> },
        { {4, DT::F32}, &PosZorderSortDispatchWrapper<4, float> },
        { {4, DT::F64}, &PosZorderSortDispatchWrapper<4, double> },
        { {4, DT::S32}, &PosZorderSortDispatchWrapper<4, int32_t> },
        { {4, DT::S64}, &PosZorderSortDispatchWrapper<4, int64_t> },
        { {5, DT::F32}, &PosZorderSortDispatchWrapper<5, float> },
        { {5, DT::F64}, &PosZorderSortDispatchWrapper<5, double> },
        { {5, DT::S32}, &PosZorderSortDispatchWrapper<5, int32_t> },
        { {5, DT::S64}, &PosZorderSortDispatchWrapper<5, int64_t> },
        { {6, DT::F32}, &PosZorderSortDispatchWrapper<6, float> },
        { {6, DT::F64}, &PosZorderSortDispatchWrapper<6, double> },
        { {6, DT::S32}, &PosZorderSortDispatchWrapper<6, int32_t> },
        { {6, DT::S64}, &PosZorderSortDispatchWrapper<6, int64_t> },
        { {7, DT::F32}, &PosZorderSortDispatchWrapper<7, float> },
        { {7, DT::F64}, &PosZorderSortDispatchWrapper<7, double> },
        { {7, DT::S32}, &PosZorderSortDispatchWrapper<7, int32_t> },
        { {7, DT::S64}, &PosZorderSortDispatchWrapper<7, int64_t> },
        { {8, DT::F32}, &PosZorderSortDispatchWrapper<8, float> },
        { {8, DT::F64}, &PosZorderSortDispatchWrapper<8, double> },
        { {8, DT::S32}, &PosZorderSortDispatchWrapper<8, int32_t> },
        { {8, DT::S64}, &PosZorderSortDispatchWrapper<8, int64_t> },
        { {9, DT::F32}, &PosZorderSortDispatchWrapper<9, float> },
        { {9, DT::F64}, &PosZorderSortDispatchWrapper<9, double> },
        { {9, DT::S32}, &PosZorderSortDispatchWrapper<9, int32_t> },
        { {9, DT::S64}, &PosZorderSortDispatchWrapper<9, int64_t> },
        { {10, DT::F32}, &PosZorderSortDispatchWrapper<10, float> },
        { {10, DT::F64}, &PosZorderSortDispatchWrapper<10, double> },
        { {10, DT::S32}, &PosZorderSortDispatchWrapper<10, int32_t> },
        { {10, DT::S64}, &PosZorderSortDispatchWrapper<10, int64_t> }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in PosZorderSortFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (2, int32_t), (2, int64_t), (3, float), (3, double), (3, int32_t), (3, int64_t), (4, float), (4, double), (4, int32_t), (4, int64_t), (5, float), (5, double), (5, int32_t), (5, int64_t), (6, float), (6, double), (6, int32_t), (6, int64_t), (7, float), (7, double), (7, int32_t), (7, int64_t), (8, float), (8, double), (8, int32_t), (8, int64_t), (9, float), (9, double), (9, int32_t), (9, int64_t), (10, float), (10, double), (10, int32_t), (10, int64_t)"
        );
    }
    PosZorderSortDispatchFn instance = it->second;

    // Now call our function
    std::string result = instance(stream,
        pos_in.untyped_data(),
        pos_out->untyped_data(),
        id_out->untyped_data(),
        tmp_buffer->untyped_data(),
        size,
        tmp_bytes,
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
    PosZorderSortFFI, PosZorderSortFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos_in
        .Ret<ffi::AnyBuffer>() // pos_out
        .Ret<ffi::AnyBuffer>() // id_out
        .Ret<ffi::AnyBuffer>() // tmp_buffer
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SearchSortedZ                             */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error SearchSortedZFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer posz_have,
    ffi::AnyBuffer posz_query,
    ffi::Result<ffi::AnyBuffer> indices,
    bool leaf_search,
    size_t block_size
) {
    size_t n_have = posz_have.dimensions()[0];
    size_t n_query = posz_query.dimensions()[0];
    int dim = posz_have.dimensions()[1];
    DT tvec = posz_have.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(n_query, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* posz_have_arg = posz_have.untyped_data();
    void* posz_query_arg = posz_query.untyped_data();
    void* indices_arg = indices->untyped_data();
    void* args[] = {
        &posz_have_arg,
        &posz_query_arg,
        &indices_arg,
        &n_have,
        &n_query,
        &leaf_search
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<2, float>) },
        { {2, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<2, double>) },
        { {2, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<2, int32_t>) },
        { {2, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<2, int64_t>) },
        { {3, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<3, float>) },
        { {3, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<3, double>) },
        { {3, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<3, int32_t>) },
        { {3, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<3, int64_t>) },
        { {4, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<4, float>) },
        { {4, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<4, double>) },
        { {4, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<4, int32_t>) },
        { {4, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<4, int64_t>) },
        { {5, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<5, float>) },
        { {5, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<5, double>) },
        { {5, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<5, int32_t>) },
        { {5, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<5, int64_t>) },
        { {6, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<6, float>) },
        { {6, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<6, double>) },
        { {6, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<6, int32_t>) },
        { {6, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<6, int64_t>) },
        { {7, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<7, float>) },
        { {7, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<7, double>) },
        { {7, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<7, int32_t>) },
        { {7, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<7, int64_t>) },
        { {8, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<8, float>) },
        { {8, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<8, double>) },
        { {8, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<8, int32_t>) },
        { {8, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<8, int64_t>) },
        { {9, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<9, float>) },
        { {9, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<9, double>) },
        { {9, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<9, int32_t>) },
        { {9, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<9, int64_t>) },
        { {10, DT::F32}, reinterpret_cast<TFunc>(&SearchSortedZ<10, float>) },
        { {10, DT::F64}, reinterpret_cast<TFunc>(&SearchSortedZ<10, double>) },
        { {10, DT::S32}, reinterpret_cast<TFunc>(&SearchSortedZ<10, int32_t>) },
        { {10, DT::S64}, reinterpret_cast<TFunc>(&SearchSortedZ<10, int64_t>) }
    };

    const TTuple key = TTuple(dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"\
            " in SearchSortedZFFIHost -- Only supporting:\n"\
            "(2, float), (2, double), (2, int32_t), (2, int64_t), (3, float), (3, double), (3, int32_t), (3, int64_t), (4, float), (4, double), (4, int32_t), (4, int64_t), (5, float), (5, double), (5, int32_t), (5, int64_t), (6, float), (6, double), (6, int32_t), (6, int64_t), (7, float), (7, double), (7, int32_t), (7, int64_t), (8, float), (8, double), (8, int32_t), (8, int64_t), (9, float), (9, double), (9, int32_t), (9, int64_t), (10, float), (10, double), (10, int32_t), (10, int64_t)"
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
    SearchSortedZFFI, SearchSortedZFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // posz_have
        .Arg<ffi::AnyBuffer>() // posz_query
        .Ret<ffi::AnyBuffer>() // indices
        .Attr<bool>("leaf_search")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_sort, m) {
    m.def("PosZorderSort", []() { return EncapsulateFfiCall(&PosZorderSortFFI); });
    m.def("SearchSortedZ", []() { return EncapsulateFfiCall(&SearchSortedZFFI); });
}