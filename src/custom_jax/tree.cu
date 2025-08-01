#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include <cub/cub.cuh>

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

ffi::Error HostSort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count();

    // Determine temporary device storage size
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, key_in.typed_data(), id_out->typed_data(), n, 0, 32, stream);
    if (temp_storage_bytes > 0) {
        // ALlocate and execute the sort
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, key_in.typed_data(), id_out->typed_data(), n, 0, 32, stream);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

__global__ void InitIndicesKernel(int32_t *indices, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        indices[idx] = idx;
    }
}

ffi::Error TreeHost(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count();

    // Create indices array and temporary sorted keys array for argsort
    int32_t *d_indices;
    int32_t *d_sorted_keys;
    cudaMalloc(&d_indices, n * sizeof(int32_t));
    cudaMalloc(&d_sorted_keys, n * sizeof(int32_t));
    
    // Initialize indices 0, 1, 2, ..., n-1
    int threads = min(1024, (int)n);
    int blocks = (n + threads - 1) / threads;
    InitIndicesKernel<<<blocks, threads, 0, stream>>>(d_indices, n);

    // Determine temporary device storage size for sorting pairs
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                    key_in.typed_data(), d_sorted_keys,
                                    d_indices, id_out->typed_data(), 
                                    n, 0, 32, stream);
    
    if (temp_storage_bytes > 0) {
        // Allocate temporary storage and execute the argsort
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        key_in.typed_data(), d_sorted_keys,
                                        d_indices, id_out->typed_data(),
                                        n, 0, 32, stream);
        cudaFree(d_temp_storage);
    }
    
    // Clean up temporary arrays
    cudaFree(d_indices);
    cudaFree(d_sorted_keys);

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