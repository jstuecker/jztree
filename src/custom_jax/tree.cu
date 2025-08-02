#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

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

__global__ void InitRangeKernel(int32_t *indices, size_t n) {
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
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    InitRangeKernel<<<blocks, block_size, 0, stream>>>(d_indices, n);

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

__device__ bool less_msb(int x, int y) {
    // Compares whether the most significant bit of x is less than that of y
    // (If it is equal, returns false)
    return (x < y) && (x < (x ^ y));
}

// Custom comparison function for z-order sort of 3 integers
__device__ bool I3ZsortCompare(const int32_t* data, int32_t a, int32_t b) {
    const int32_t* triplet_a = &data[a * 3];
    const int32_t* triplet_b = &data[b * 3];

    int32_t xorab[] = {triplet_a[0] ^ triplet_b[0], triplet_a[1] ^ triplet_b[1], triplet_a[2] ^ triplet_b[2]};

    // Figure out which dimension has the most significant bit differing
    int ms_dim = less_msb(xorab[0], xorab[1]) ? 1 : 0;
    ms_dim = less_msb(xorab[ms_dim], xorab[2]) ? 2 : ms_dim;

    return triplet_a[ms_dim] < triplet_b[ms_dim];
}

ffi::Error HostI3Zsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count() / 3;  // Number of triplets
    const int32_t* data = key_in.typed_data();
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    InitRangeKernel<<<blocks, block_size, 0, stream>>>(id_out->typed_data(), n);
    
    // Use Thrust to sort with custom comparison function
    thrust::device_ptr<int32_t> indices_ptr(id_out->typed_data());
    thrust::sort(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + n,
        [=] __device__ (int32_t a, int32_t b) {
            return I3ZsortCompare(data, a, b);
        });

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

__device__ int32_t float_xor_msb(float a, float b) {
    // finds the most significant bit that differs between x and y
    if (signbit(a) != signbit(b)) {
        return 128;  // The sign is the highest significant bit
    }
    int32_t a_bits = __float_as_int(fabsf(a));
    int32_t b_bits = __float_as_int(fabsf(b));

    int32_t a_exp = (a_bits >> 23) - 127;
    int32_t b_exp = (b_bits >> 23) - 127;

    if (a_exp == b_exp) { // If both floats have the same exponent, we need to compare mantissas
        int32_t mantissa_msb = 32 - __clz(a_bits ^ b_bits);
        return a_exp + (mantissa_msb - 24);
    }
    else { // If exponents differ, return the larger exponent
        return a_exp > b_exp ? a_exp : b_exp;
    }
}

// Custom comparison function for z-order sort of 3 floats
__device__ bool F3ZsortCompare(const float* data, int32_t a, int32_t b) {
    const float* xa = &data[a * 3];
    const float* xb = &data[b * 3];
    
    // Get the most significant differing bit for each dimension
    int32_t msbs[] = {
        float_xor_msb(xa[0], xb[0]),
        float_xor_msb(xa[1], xb[1]),
        float_xor_msb(xa[2], xb[2])
    };

    int ms_dim = msbs[1] > msbs[0] ? 1 : 0;
    ms_dim = msbs[2] > msbs[ms_dim] ? 2 : ms_dim;

    // Compare the values in the dimension with the most significant differing bit
    return xa[ms_dim] < xb[ms_dim];
}

ffi::Error HostF3Zsort(cudaStream_t stream, ffi::Buffer<ffi::F32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count() / 3;  // Number of triplets
    const float* data = key_in.typed_data();
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    InitRangeKernel<<<blocks, block_size, 0, stream>>>(id_out->typed_data(), n);
    
    // Use Thrust to sort with custom comparison function
    thrust::device_ptr<int32_t> indices_ptr(id_out->typed_data());
    thrust::sort(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + n,
        [=] __device__ (int32_t a, int32_t b) {
            return F3ZsortCompare(data, a, b);
        });

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

struct custom_t
{
  int key1, key2, key3;

  custom_t() = default;
  custom_t(int3 keys) : key1(keys.x), key2(keys.y), key3(keys.z) {}
};

struct decomposer_t
{
  __host__ __device__ thrust::tuple<int&, int&, int&> operator()(custom_t& key) const
  {
    return thrust::tuple<int&, int&, int&>(key.key1, key.key2, key.key3);
  }
};


ffi::Error HostI3Argsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count()/3;

    // Create indices array and temporary sorted keys array for argsort
    int32_t *d_indices;
    custom_t *d_sorted_keys;
    cudaMalloc(&d_indices, n * sizeof(int3));
    cudaMalloc(&d_sorted_keys, n * sizeof(custom_t));
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    InitRangeKernel<<<blocks, block_size, 0, stream>>>(d_indices, n);

    const custom_t* d_keys_in = reinterpret_cast<const custom_t*>(key_in.typed_data());

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_sorted_keys, 
        d_indices, id_out->typed_data(), n, decomposer_t{});

    if (temp_storage_bytes > 0) {
        // Allocate temporary storage and execute the argsort
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, 
            d_sorted_keys, d_indices, id_out->typed_data(), n, decomposer_t{});

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    I3Argsort, HostI3Argsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_tree, m) {
    m.def("argsort", []() { return EncapsulateFfiCall(Argsort); });
    m.def("i3zsort", []() { return EncapsulateFfiCall(I3zsort); });
    m.def("f3zsort", []() { return EncapsulateFfiCall(F3zsort); });
    m.def("i3argsort", []() { return EncapsulateFfiCall(I3Argsort); });
}