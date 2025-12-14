#include <type_traits>

#include <cmath>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "common/segment_sort.cuh"
#include "common/data.cuh"
#include "common/math.cuh"
#include "common/knn_math.cuh"
#include "common/iterators.cuh"
#include <cub/cub.cuh>

#define INTERACTION_BINS 16

namespace nb = nanobind;
namespace ffi = xla::ffi;

template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Segment Sort FFI                                        */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error HostSegmentSort(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> key,
    ffi::Buffer<ffi::S32> val,
    ffi::Buffer<ffi::S32> isplit,
    ffi::ResultBuffer<ffi::F32> key_out,
    ffi::ResultBuffer<ffi::S32> val_out,
    int smem_size
)
{
    int nsegs = isplit.element_count() - 1;
    int blocksize = 64;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int));

    // copy data to output buffers
    cudaMemcpyAsync(key_out->typed_data(), key.typed_data(), key.size_bytes(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(val_out->typed_data(), val.typed_data(), val.size_bytes(), cudaMemcpyDeviceToDevice, stream);

    segmented_bitonic_sort_kv<<< nsegs, blocksize, smem_bytes, stream>>>(
        key_out->typed_data(), val_out->typed_data(), isplit.typed_data(), nsegs, smem_size);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SegmentSort, HostSegmentSort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() 
        .Arg<ffi::Buffer<ffi::F32>>() // key
        .Arg<ffi::Buffer<ffi::S32>>() // val
        .Arg<ffi::Buffer<ffi::S32>>() // isplit
        .Ret<ffi::Buffer<ffi::F32>>() // key_out
        .Ret<ffi::Buffer<ffi::S32>>() // val_out
        .Attr<int>("smem_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});


NB_MODULE(ffi_knn, m) {
    m.def("SegmentSort", []() { return EncapsulateFfiCall(SegmentSort); });
}