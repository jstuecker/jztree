#ifndef SEGMENT_SORT_CUH
#define SEGMENT_SORT_CUH

#include <cub/cub.cuh>
#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

#include "common/data.cuh"
#include "common/math.cuh"

#define INFTY  INFINITY //__int_as_float(0x7f800000)

/* ---------------------------------------------------------------------------------------------- */
/*                                           Zorder Sort                                          */
/* ---------------------------------------------------------------------------------------------- */

// Wrapper of the z-order comparison function to use with CUB
template <int dim=3>
struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId<dim> &a, const PosId<dim> &b) {
        return z_pos_less<dim>(a.pos, b.pos);
    }
};

// Prepare keys and ids for sorting
template <int dim=3>
__global__ void PosKeyArangeKernel(const Pos<dim>* pos_in, PosId<dim> *keyid_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keyid_out[idx].pos = pos_in[idx];
        keyid_out[idx].id = idx;
    }
}

template<int dim>
std::string PosZorderSort(
    cudaStream_t stream, 
    const Pos<dim>* pos_in, 
    PosId<dim>* pos_id_out,
    int* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
) {
    // Initialize indices 0, 1, 2, ..., size-1
    PosKeyArangeKernel<dim><<< div_ceil(size, block_size), block_size, 0, stream>>>(pos_in, pos_id_out, size);

    // We have an annoying problem here:
    // CUB requires a temporary storage buffer and it will usually tell us dynamically what the
    // size of it is (On first call with zero pointer).
    // Unfortunately, we may not allocate storage dynamically in an FFI call
    // Therefore, we have to estimate the storage requirements in advance in python/jax and pass
    // a sufficiently large buffer to the function (via "tmp_buffer").
    // Empirically I have found that the storage tends to be a bit larger than n * sizeof(PosId)
    // that is why we will pre-allocate something that is a few percent larger than that.
    // However, below we throw an error if our assumption ever turns out wrong.

    // find out the required storage size
    size_t required_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys<PosId<dim>*, int64_t, PosIdLess<dim>>(
        nullptr, required_storage_bytes, pos_id_out, size, PosIdLess<dim>()
    );
    
    // Check if the provided buffer is large enough
    if (tmp_bytes < required_storage_bytes) {
        return std::string(
            "The buffer in ZorderSort is too small. Please contact me if this check fails.") +
            std::string(" Have: ") + std::to_string(tmp_bytes) +
            std::string(". Required: ") + std::to_string(required_storage_bytes) +
            std::string(". Diff: ") + std::to_string((long long)required_storage_bytes - (long long)tmp_bytes);
    }

    // Run the sort
    cub::DeviceMergeSort::SortKeys<PosId<dim>*, int64_t, PosIdLess<dim>>(
        tmp_buffer, required_storage_bytes, pos_id_out, size, PosIdLess<dim>(), stream
    );
    
    return std::string();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          SearchSortedZ                                         */
/* ---------------------------------------------------------------------------------------------- */

__global__ void SearchSortedZ(
    const float3* posz_have,
    const float3* posz_query,
    int32_t* indices,
    size_t n_have,
    size_t n_query,
    bool leaf_search
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_query)
        return;

    float3 xquery = posz_query[idx];

    // Binary search for the indices between which xquery would need to be inserted to 
    // maintain order
    int imin = 0, imax = n_have;
    while (imin+1 < imax) {
        int itest = (imin + imax) >> 1;
        if (z_pos_less3(posz_have[itest], xquery)) {
            imin = itest;
        } else {
            imax = itest;
        }
    }

    int iout;
    if(leaf_search) {
        // In this scenario, we need to learn whether the particle belongs to the left or right leaf
        // it always belongs to the one with the smaller difference level
        int lv1 = msb_diff_level(posz_have[imin], xquery);
        int lv2 = (imax < n_have) ? msb_diff_level(posz_have[imax], xquery) : 388;
        if(lv1 <= lv2)
            iout = imin;
        else
            iout = imax;
    }
    else {
        // If we are doing a normal binary search, the index is in general imin + 1
        // and we only need to take care of the boundary cases
        if(imin == 0)
            iout = z_pos_less3(posz_have[0], xquery) ? 1 : 0;
        else if(imin == n_have - 1)
            iout = z_pos_less3(posz_have[n_have - 1], xquery) ? n_have : n_have - 1;
        else
            iout = imin + 1;
    }

    indices[idx] = iout;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Segment Sort                                          */
/* ---------------------------------------------------------------------------------------------- */

// Utility: next power-of-two >= n (cap at MAX_THREADS)
// Smallest power of two >= x (32-bit)
__device__ __forceinline__ unsigned int next_pow2_u32(unsigned int x) {
    if (x == 0) return 1u;                 // define as 1 for x=0
    if (x > 0x80000000u) return 0u;        // overflow (no 33rd bit in u32)
    return 1u << (32 - __clz(x - 1));      // __clz: count leading zeros
}

__device__ __forceinline__ 
void compare_and_swap(float *val, int32_t *idx, int i, int j, int len) {
    if(i < j && j < len && val[i] > val[j]) {
        float t = val[i];
        val[i] = val[j];
        val[j] = t;

        int32_t ti = idx[i];
        idx[i] = idx[j];
        idx[j] = ti;
    }
}

// Bitonic sort network -- works in-place and can use non-power-of-two lengths
__device__ void bitonic_sort(float* skeys, int32_t* svals, int len) {
    // see https://stackoverflow.com/questions/73147204
    int nPow2 = next_pow2_u32(len);

    for (int k = 2; k <= nPow2; k <<= 1) {
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            compare_and_swap(skeys, svals, i, i ^ (k - 1), len);
        }
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < len; i += blockDim.x) {
                compare_and_swap(skeys, svals, i, i ^ j, len);
            }
        }
    }
    __syncthreads();
}

// -------- Kernel: segmented sort (keys + values) --------
__global__ void segmented_bitonic_sort_kv(
    float* __restrict__ keys,
    int32_t* __restrict__ values,
    const int32_t* __restrict__ offsets,
    int32_t num_segments,
    int32_t smem_size // the maximum segment length where we do the sort in shared memory
)
{
    int seg = blockIdx.x;
    if (seg >= num_segments) return;

    int32_t start = offsets[seg];
    int32_t end   = offsets[seg + 1];
    int32_t len   = max(end - start, 0);
    if (len <= 1) return;

    if(len <= smem_size) { // We can do the sort in shared memory
        extern __shared__ unsigned char smem[];
        float*   skeys = reinterpret_cast<float*>(smem);
        int32_t* svals = reinterpret_cast<int32_t*>(skeys + smem_size);

        // Load to shared memory
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            skeys[i] = keys[start + i];
            svals[i] = values[start + i];
        }
        __syncthreads();

        // Sort in shared memory
        bitonic_sort(skeys, svals, len);

        // Store back
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            keys[start + i]   = skeys[i];
            values[start + i] = svals[i];
        }
    }
    else { // Fall back to global memory sort (inefficient, but rare)
        bitonic_sort(&keys[start], &values[start], len);
    }
}

#endif