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

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: PosZorderSort                             */
/* ---------------------------------------------------------------------------------------------- */


using PosZorderSortDispatchFn = std::string (*) (cudaStream_t stream,
    const void* pos_in,
    void* pos_id_out,
    void* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
);
template<int dim>
static std::string PosZorderSortDispatchWrapper(cudaStream_t stream,
    const void* pos_in,
    void* pos_id_out,
    void* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
) {
    return PosZorderSort<dim> (stream,
        reinterpret_cast<const Pos<dim>*>(pos_in),
        reinterpret_cast<PosId<dim>*>(pos_id_out),
        reinterpret_cast<int*>(tmp_buffer),
        size,
        tmp_bytes,
        block_size
    );
}


ffi::Error PosZorderSortFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_in,
    ffi::Result<ffi::AnyBuffer> pos_id_out,
    ffi::Result<ffi::AnyBuffer> tmp_buffer,
    size_t block_size,
    int dim
) {
    size_t size = pos_in.element_count()/3;
    size_t tmp_bytes = tmp_buffer->size_bytes();

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int>;

    using TFunctionType =
        PosZorderSortDispatchFn
    ;

    static const std::map<TTuple, TFunctionType> instance_map = {
        { {2}, &PosZorderSortDispatchWrapper<2> },
        { {3}, &PosZorderSortDispatchWrapper<3> }
    };

    const TTuple key = TTuple{dim};

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim)"\
            " in PosZorderSortFFIHost -- Only supporting:\n"\
            "(2), (3)"
        );
    }
    PosZorderSortDispatchFn instance = it->second;

    // Now call our function
    std::string result = instance(stream,
        pos_in.untyped_data(),
        pos_id_out->untyped_data(),
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
        .Ret<ffi::AnyBuffer>() // pos_id_out
        .Ret<ffi::AnyBuffer>() // tmp_buffer
        .Attr<size_t>("block_size")
        .Attr<int>("dim"),
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
    size_t n_have = posz_have.element_count()/3;
    size_t n_query = posz_query.element_count()/3;
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
    const void* instance = (const void*)SearchSortedZ;

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