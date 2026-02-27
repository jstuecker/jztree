#ifndef SEGMENT_SORT_CUH
#define SEGMENT_SORT_CUH

#include <cub/cub.cuh>
#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

#include "common/data.cuh"
#include "common/math.cuh"

#define INFTY  INFINITY //__int_as_float(0x7f800000)

template <bool mode, typename in_type, typename out_type, int offset>
__global__ void DtypeTest(
    const in_type* in,
    out_type* out,
    size_t size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx + offset < size) {
        if(mode) {
            out[idx] = in[idx + offset] * 2.;
        } else {
            out[idx] = in[idx + offset] * 3.;
        }
    }
}


/* ---------------------------------------------------------------------------------------------- */
/*                                           Zorder Sort                                          */
/* ---------------------------------------------------------------------------------------------- */

// Wrapper of the z-order comparison function to use with CUB
template <int dim, typename tvec>
struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId<dim,tvec> &a, const PosId<dim,tvec> &b) {
        return z_pos_less<dim,tvec>(a.pos, b.pos);
    }
};

// Prepare keys and ids for sorting
template <int dim=3, typename tvec>
__global__ void PosKeyArangeKernel(
    const Vec<dim, tvec>* pos_in,
    PosId<dim, tvec> *keyid_out,
    size_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keyid_out[idx].pos = pos_in[idx];
        keyid_out[idx].id = idx;
    }
}

template<int dim, typename tvec>
std::string PosZorderSort(
    cudaStream_t stream, 
    const Vec<dim, tvec>* pos_in, 
    PosId<dim, tvec>* pos_id_out,
    int* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
) {
    // Initialize indices 0, 1, 2, ..., size-1
    PosKeyArangeKernel<dim, tvec><<< div_ceil(size, block_size), block_size, 0, stream>>>(
        pos_in, pos_id_out, size
    );

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
    cub::DeviceMergeSort::SortKeys<PosId<dim, tvec>*, int64_t, PosIdLess<dim, tvec>>(
        nullptr, required_storage_bytes, pos_id_out, size, PosIdLess<dim, tvec>()
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
    cub::DeviceMergeSort::SortKeys<PosId<dim, tvec>*, int64_t, PosIdLess<dim, tvec>>(
        tmp_buffer, required_storage_bytes, pos_id_out, size, PosIdLess<dim, tvec>(), stream
    );
    
    return std::string();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Radix based z-order sort                                    */
/* ---------------------------------------------------------------------------------------------- */

// template<int dim, typename tint>
// struct RadixKey
// {
//     Vec<dim,tint> key;
//     int32_t  id;
// };

// Initialization kernel: fill pos_id_out[i] with key parts + id.
// Replace the logic with your own.

template<typename tvec, typename tint>
__device__ tint to_int(tvec val) {
    // define an integer so that  val1 < val2 <=> int1 < int2
    if constexpr (std::is_same_v<tvec, float>) {
        // If sign is negative, flip all bits, else flip sign
        uint32_t b = __float_as_uint(val);
        uint32_t mask = (b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return b ^ mask;
    }
    else if constexpr (std::is_same_v<tvec, double>) {
        uint64_t b = (uint64_t)__double_as_longlong(val);
        uint64_t mask = (b & 0x8000000000000000ull)
                    ? 0xFFFFFFFFFFFFFFFFull
                    : 0x8000000000000000ull;
        return b ^ mask;
    }
    else if constexpr (std::is_same_v<tvec, int32_t>) {
        return (tint) val ^ 0x80000000u; // flip sign bit
    }
    else if constexpr (std::is_same_v<tvec, int64_t>) {
        return (tint) val ^ 0x8000000000000000ull;
    }
    else
        return (tint) val;
}

template<typename tvec, typename tint>
__device__ tvec from_int(tint ival) {
    // define an integer so that  val1 < val2 <=> int1 < int2
    if constexpr (std::is_same_v<tvec, float>) {
        // If sign is negative, flip all bits, else flip sign
        uint32_t b = (ival & 0x80000000u) ? (ival ^ 0x80000000u) : ~ival;
        return __uint_as_float(b);
    }
    else if constexpr (std::is_same_v<tvec, double>) {
        uint64_t b = (ival & 0x8000000000000000ull) ? (ival ^ 0x8000000000000000ull) : ~ival;
        return __longlong_as_double((long long)b);
    }
    else if constexpr (std::is_same_v<tvec, int32_t>) {
        return (tvec) ival ^ 0x80000000u;
    }
    else if constexpr (std::is_same_v<tvec, int64_t>) {
        return (tvec) ival ^ 0x8000000000000000ull;
    }
    else
        return (tvec) ival;
}

template <bool deinterleave, int dim, typename tint>
__device__ __forceinline__ void interleave_bits(
    const Vec<dim,tint> &in, Vec<dim,tint> &out)
{
    static_assert(std::is_same_v<tint,uint32_t> || std::is_same_v<tint,uint64_t>);
    constexpr int nbits = sizeof(tint) * 8;

    #pragma unroll
    for (int k = 0; k < dim; ++k) out[k] = tint{0};

    #pragma unroll
    for (int i = 0; i < nbits; ++i) {
        #pragma unroll
        for (int j = 0; j < dim; ++j) {
            int p = i * dim + j;

            if constexpr (!deinterleave) {
                tint bit = (in[j] >> i) & tint{1};
                out[p/nbits] |= bit << (p % nbits);
            } else {
                tint bit = (in[p/nbits] >> (p % nbits)) & tint{1};
                out[j] |= bit << i;
            }
        }
    }
}



template<int dim, typename tvec, typename tint>
__global__ void InterleaveKernel(
    const Vec<dim, tvec>* pos_in,
    PosId<dim, tint>* key_out,
    size_t n
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec<dim,tvec> in = pos_in[i];
    Vec<dim,tint> tmp;

    #pragma unroll
    for(int d=0; d<dim; d++)
        tmp[d] = to_int<tvec,tint>(in[d]);
    
    PosId<dim,tint> out{};
    out.id = i;
    interleave_bits<false,dim,tint>(reversed_vec(tmp), out.pos); // reverse, since CUB sorts by last

    key_out[i] = out;
}


template<int dim, typename tvec, typename tint>
__global__ void DeinterleaveKernel(
    const PosId<dim, tint>* key_in,
    PosId<dim, tvec>* pos_id_out,
    size_t n
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PosId<dim,tint> in = key_in[i];
    Vec<dim,tint> tmp;

    interleave_bits<true,dim,tint>(in.pos, tmp); // deinterleave
    tmp = reversed_vec(tmp); // reverse, since CUB sorts by last

    PosId<dim, tvec> out;
    out.id = in.id;
    #pragma unroll
    for(int d=0; d<dim; d++)
        out.pos[d] = from_int<tvec,tint>(tmp[d]);
    
    pos_id_out[i] = out;
}

// Decomposer for radix sort.
// IMPORTANT: return tuple in order (least significant ... most significant).
template<int dim, typename tint>
struct PosIdDecomposer
{
    __host__ __device__
    ::cuda::std::tuple<tint&, tint&, tint&>
    operator()(PosId<dim, tint>& x) const
    {
        // Least -> most: k2 (tertiary), k1 (secondary), k0 (primary)
        return { x.pos[2], x.pos[1], x.pos[0] };
    }

    // Optional const overload (some toolchains like having it)
    __host__ __device__
    ::cuda::std::tuple<const tint&, const tint&, const tint&>
    operator()(const PosId<dim, tint>& x) const
    {
        return { x.pos[2], x.pos[1], x.pos[0] };
    }
};

template<int dim, typename tvec>
std::string PosZorderSortRadix(
    cudaStream_t stream,
    const Vec<dim, tvec>* pos_in,
    PosId<dim, tvec>* pos_id_out,
    PosId<dim, tvec>* pos_id_tmp,
    void* tmp_buffer,          // device temp storage
    size_t size,
    size_t tmp_bytes,
    size_t block_size)
{
    constexpr bool is32bit = sizeof(tvec) == 4;
    constexpr bool is64bit = sizeof(tvec) == 8;

    static_assert(is32bit || is64bit, "only 32bit or 64bit types supported");

    using tint = std::conditional_t<is32bit, std::uint32_t, std::uint64_t>;

    // Can use output buffer as temporary
    PosId<dim, tint> *keys_tmp = reinterpret_cast<PosId<dim, tint>*>(pos_id_tmp);
    PosId<dim, tint> *keys_tmp2 = reinterpret_cast<PosId<dim, tint>*>(pos_id_out);

    // 1) init
    InterleaveKernel<dim, tvec, tint> <<< div_ceil(size, block_size), block_size, 0, stream >>>(
        pos_in, keys_tmp, size
    );

    // 2) query temp storage bytes
    cub::DoubleBuffer<PosId<dim, tint>> keys(keys_tmp, keys_tmp2);
    size_t required_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, required_storage_bytes, keys, (int64_t)size, PosIdDecomposer<dim, tint>{}, stream
    );

    if (tmp_bytes < required_storage_bytes) {
        return std::string("CUB temp buffer too small. Have: ")
        + std::to_string(tmp_bytes) + " Required: " + std::to_string(required_storage_bytes);
    }

    keys.selector = 0;
    cub::DeviceRadixSort::SortKeys(
        tmp_buffer, required_storage_bytes, keys, static_cast<int64_t>(size), 
        PosIdDecomposer<dim, tint>{}, stream
    );

    // If the sorted data ended up in the alternate buffer, copy back.
    if (keys.selector == 0)
        DeinterleaveKernel<dim,tvec,tint><<< div_ceil(size, block_size), block_size, 0, stream >>>(
            keys_tmp, pos_id_out, size);
    else
        DeinterleaveKernel<dim,tvec,tint><<< div_ceil(size, block_size), block_size, 0, stream >>>(
            keys_tmp2, pos_id_out, size);

    return std::string();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          SearchSortedZ                                         */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
__global__ void SearchSortedZ(
    const Vec<dim,tvec>* posz_have,
    const Vec<dim,tvec>* posz_query,
    int32_t* indices,
    size_t n_have,
    size_t n_query,
    bool leaf_search
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_query)
        return;

    Vec<dim,tvec> xquery = posz_query[idx];

    // Binary search for the indices between which xquery would need to be inserted to 
    // maintain order
    int imin = 0, imax = n_have;
    while (imin+1 < imax) {
        int itest = (imin + imax) >> 1;
        if (z_pos_less<dim,tvec>(posz_have[itest], xquery)) {
            imin = itest;
        } else {
            imax = itest;
        }
    }

    int iout;
    if(leaf_search) {
        // In this scenario, we need to learn whether the particle belongs to the left or right leaf
        // it always belongs to the one with the smaller difference level
        int lv1 = msb_diff_level<dim,tvec>(posz_have[imin], xquery);
        int lv2 = (imax < n_have) ? msb_diff_level<dim,tvec>(posz_have[imax], xquery) : 2000;
        if(lv1 <= lv2)
            iout = imin;
        else
            iout = imax;
    }
    else {
        // If we are doing a normal binary search, the index is in general imin + 1
        // and we only need to take care of the boundary cases
        if(imin == 0)
            iout = z_pos_less<dim,tvec>(posz_have[0], xquery) ? 1 : 0;
        else if(imin == n_have - 1)
            iout = z_pos_less<dim,tvec>(posz_have[n_have - 1], xquery) ? n_have : n_have - 1;
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