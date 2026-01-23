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

ffi::Error PosZorderSortFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_in,
    ffi::Result<ffi::AnyBuffer> pos_id_out,
    ffi::Result<ffi::AnyBuffer> tmp_buffer,
    size_t block_size
) {
    size_t size = pos_in.element_count()/3;
    size_t tmp_bytes = tmp_buffer->size_bytes();

    // Now call our function
    std::string result = PosZorderSort(
        stream,
        reinterpret_cast<float3*>(pos_in.untyped_data()),
        reinterpret_cast<PosId*>(pos_id_out->untyped_data()),
        reinterpret_cast<int*>(tmp_buffer->untyped_data()),
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
    size_t n_have = posz_have.element_count()/3;
    size_t n_query = posz_query.element_count()/3;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(n_query, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float3* posz_have_val = reinterpret_cast<float3*>(posz_have.untyped_data());
    float3* posz_query_val = reinterpret_cast<float3*>(posz_query.untyped_data());
    int32_t* indices_val = reinterpret_cast<int32_t*>(indices->untyped_data());

    void* args[] = {
        &posz_have_val,
        &posz_query_val,
        &indices_val,
        &n_have,
        &n_query,
        &leaf_search
    };
    cudaLaunchKernel((const void*)SearchSortedZ, gridDim, blockDim, args, smem, stream);

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