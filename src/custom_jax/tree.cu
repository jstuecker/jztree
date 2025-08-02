#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

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

ffi::Error HostArgsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
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

// Custom comparison function for lexicographic ordering of triplets
__device__ bool LexicographicCompare(const int32_t* data, int32_t a, int32_t b) {
    const int32_t* triplet_a = &data[a * 3];
    const int32_t* triplet_b = &data[b * 3];
    
    // Compare first element
    if (triplet_a[0] != triplet_b[0]) {
        return triplet_a[0] < triplet_b[0];
    }
    // If first elements are equal, compare second element
    if (triplet_a[1] != triplet_b[1]) {
        return triplet_a[1] < triplet_b[1];
    }
    // If first two elements are equal, compare third element
    return triplet_a[2] < triplet_b[2];
}

ffi::Error HostI3Zsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count() / 3;  // Number of triplets
    const int32_t* data = key_in.typed_data();
    
    // Initialize indices 0, 1, 2, ..., n-1
    int threads = min(1024, (int)n);
    int blocks = (n + threads - 1) / threads;
    InitIndicesKernel<<<blocks, threads, 0, stream>>>(id_out->typed_data(), n);
    
    // Use Thrust to sort indices based on lexicographic comparison of triplets
    thrust::device_ptr<int32_t> indices_ptr(id_out->typed_data());
    thrust::sort(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + n,
        [=] __device__ (int32_t a, int32_t b) {
            return LexicographicCompare(data, a, b);
        });

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

ffi::Error HostF3Zsort(cudaStream_t stream, ffi::Buffer<ffi::F32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Argsort, HostArgsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    I3zsort, HostI3Zsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    F3zsort, HostF3Zsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()        // x
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_tree, m) {
    m.def("argsort", []() { return EncapsulateFfiCall(Argsort); });
    m.def("i3zsort", []() { return EncapsulateFfiCall(I3zsort); });
    m.def("f3zsort", []() { return EncapsulateFfiCall(F3zsort); });
}