#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// A wrapper to encapsulate an FFI call
template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

__global__ void TreeKernel(const int32_t *i_in, int32_t *i_out, size_t n) {
    const size_t blocksize = blockDim.x;

    for (size_t jblock = 0; jblock < n; jblock += blocksize) {
        size_t j = jblock + threadIdx.x;
        if (j < n) {
            i_out[j] = 2*i_in[j];
        }
    }
}

ffi::Error TreeHost(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count();

    size_t grid_size = (n + block_size - 1) / block_size;

    TreeKernel<<<grid_size, block_size, 0, stream>>>(key_in.typed_data(), id_out->typed_data(), n);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tree, TreeHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids (should be S32, not F32)
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_tree, m) {
    m.def("tree", []() { return EncapsulateFfiCall(Tree); });
}