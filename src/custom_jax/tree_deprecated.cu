// This file contains deprecated/discarded functions that are kept for comparison purposes
// It's included as a header from tree.cu
// All necessary includes and declarations should be in the parent file

__device__ bool less_msb(int x, int y) {
    // Compares whether the most significant bit of x is less than that of y
    // (If it is equal, returns false)
    return (x < y) && (x < (x ^ y));
}


__global__ void InitRangeKernel(int32_t *indices, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        indices[idx] = idx;
    }
}

// Deprecated: Simple argsort using CUB DeviceRadixSort
ffi::Error OldHostArgsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count();

    // Create indices array and temporary sorted keys array for argsort
    int32_t *d_indices;
    int32_t *d_sorted_keys;
    cudaMalloc(&d_indices, n * sizeof(int32_t));
    cudaMalloc(&d_sorted_keys, n * sizeof(int32_t));
    
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

ffi::Error OldHostSort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
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

// Custom comparison function for z-order sort of 3 integers
__device__ bool OldI3ZsortCompare(const int32_t* data, int32_t a, int32_t b) {
    const int32_t* triplet_a = &data[a * 3];
    const int32_t* triplet_b = &data[b * 3];

    int32_t xorab[] = {triplet_a[0] ^ triplet_b[0], triplet_a[1] ^ triplet_b[1], triplet_a[2] ^ triplet_b[2]};

    // Figure out which dimension has the most significant bit differing
    int ms_dim = less_msb(xorab[0], xorab[1]) ? 1 : 0;
    ms_dim = less_msb(xorab[ms_dim], xorab[2]) ? 2 : ms_dim;

    return triplet_a[ms_dim] < triplet_b[ms_dim];
}

ffi::Error OldHostI3Zsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count() / 3;  // Number of triplets
    const int32_t* data = key_in.typed_data();
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    InitRangeKernel<<<blocks, block_size, 0, stream>>>(id_out->typed_data(), n);
    
    // Use Thrust to sort with custom comparison function
    thrust::device_ptr<int32_t> indices_ptr(id_out->typed_data());
    thrust::sort(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + n,
        [=] __device__ (int32_t a, int32_t b) {
            return OldI3ZsortCompare(data, a, b);
        });

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


// Custom comparison function for z-order sort of 3 floats
__device__ bool OldF3ZsortCompare(const float* data, int32_t a, int32_t b) {
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

ffi::Error OldHostF3Zsort(cudaStream_t stream, ffi::Buffer<ffi::F32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count() / 3;  // Number of triplets
    const float* data = key_in.typed_data();
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    InitRangeKernel<<<blocks, block_size, 0, stream>>>(id_out->typed_data(), n);
    
    // Use Thrust to sort with custom comparison function
    thrust::device_ptr<int32_t> indices_ptr(id_out->typed_data());
    thrust::sort(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + n,
        [=] __device__ (int32_t a, int32_t b) {
            return OldF3ZsortCompare(data, a, b);
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

ffi::Error OldHostI3Argsort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count()/3;

    // Create indices array and temporary sorted keys array for argsort
    int32_t *d_indices;
    custom_t *d_sorted_keys;
    cudaMalloc(&d_indices, n * sizeof(int32_t));
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


struct OldKeyId {
    int3 key;
    int32_t id;
};

struct OldKeyIdLess {
    __device__ __forceinline__
    bool operator()(const OldKeyId &a, const OldKeyId &b) {
        // Count leading zeros of the difference (higher means lower MSB)
        int clz0 = __clz(a.key.x ^ b.key.x);
        int clz1 = __clz(a.key.y ^ b.key.y);
        int clz2 = __clz(a.key.z ^ b.key.z);

        // Find dimension with least clz → most significant differing bit
        int ms_dim = (clz0 <= clz1 && clz0 <= clz2) ? 0 : ((clz1 <= clz2) ? 1 : 2);

        // Perform the comparison on the most significant dimension
        if (ms_dim == 0) return a.key.x < b.key.x;
        if (ms_dim == 1) return a.key.y < b.key.y;
        return a.key.z < b.key.z;
    }
};

__global__ void OldKeyArangeKernel(const int3* key_in, OldKeyId *keyid_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keyid_out[idx].key = key_in[idx];
        keyid_out[idx].id = idx;
    }
}

ffi::Error OldHostI3zMergesort(cudaStream_t stream, ffi::Buffer<ffi::S32> key_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = key_in.element_count()/3;

    int3* keys_in = reinterpret_cast<int3*>(key_in.typed_data());
    OldKeyId* keyids = reinterpret_cast<OldKeyId*>(id_out->typed_data());
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    OldKeyArangeKernel<<<blocks, block_size, 0, stream>>>(keys_in, keyids, n);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceMergeSort::SortKeys<OldKeyId*, int64_t, OldKeyIdLess>(nullptr, temp_storage_bytes, keyids, n, OldKeyIdLess(), stream);

    if (temp_storage_bytes > 0) {
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceMergeSort::SortKeys<OldKeyId*, int64_t, OldKeyIdLess>(d_temp_storage, temp_storage_bytes, keyids, n, OldKeyIdLess(), stream);
        cudaFree(d_temp_storage);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OldArgsort, OldHostArgsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OldI3zsort, OldHostI3Zsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OldF3zsort, OldHostF3Zsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()        // x
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OldI3Argsort, OldHostI3Argsort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OldI3zMergesort, OldHostI3zMergesort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()        // ids
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});