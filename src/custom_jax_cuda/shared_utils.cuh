#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

#define INFTY  INFINITY //__int_as_float(0x7f800000)

// A wrapper to encapsulate an FFI call
template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}

// A function calculating the ceil of integer division on CPU
inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

template <int NLOAD, bool SINGLE = false>
struct SegmentManager {
    // Manages a pointer based segment list so that it can be traversed with threads
    // almost as if it was a linear structure
    // each segment goes from isplits[i] to isplits[i+1]
    // inodes contains the indices of the segments to be processed
    // istart, iend define the range of inodes to be processed
    // if SINGLE is set, only particles in one segment are loaded at a time

    int istart, iend;                     // indices into inodes[]
    const int* __restrict__ inodes;       // segment indices to visit (defaults to 0,1,... if nullptr)
    const int* __restrict__ isplits;      // global segment offsets
    int2* __restrict__ segments;          // -> shared buffer [NLOAD], allocated externally
    int seg_loaded;                       // how many segments resident in shared

    // internal state:
    int next_seg = 0;
    int seg_offset = 0;
    int num_loaded = 0;
    
    __device__ __forceinline__ SegmentManager(const int* inodes_, const int* isplits_, int2* segments_, int istart_, int iend_) {
        inodes = inodes_;
        isplits = isplits_;
        segments = segments_;

        istart = istart_;
        iend = iend_;
        
        loadSegments();
    }

    __device__ __forceinline__ void loadSegments() {
        seg_loaded = min(NLOAD, iend-istart);
        if(threadIdx.x < seg_loaded) {
            int segment_idx = inodes ? inodes[istart + threadIdx.x] : istart + threadIdx.x;
            int i0 = isplits[segment_idx];
            int i1 = isplits[segment_idx + 1];
            int2 segment = {i0, i1 - i0};
            segments[threadIdx.x] = segment;
        }
        istart += seg_loaded;
        next_seg = 0;
        __syncthreads();
    }
    
    __device__ __forceinline__ int2 next() {
        // If required, load more segments
        if(next_seg >= seg_loaded) {
            loadSegments();
        }

        int2 id = {-1, istart - seg_loaded + next_seg};
        // Go through the segments until we are sure that every thread has something
        num_loaded = 0;
        while (next_seg < seg_loaded)
        {
            int2 seg = segments[next_seg]; // {start, length}
            int nadd = min(seg.y - seg_offset, blockDim.x - num_loaded);
            if((threadIdx.x >= num_loaded ) && (threadIdx.x < num_loaded + nadd)) {
                id.x = seg.x + seg_offset + (threadIdx.x - num_loaded);
                id.y = istart - seg_loaded + next_seg;
            }
            num_loaded += nadd;
            if(num_loaded >= blockDim.x) {
                seg_offset += nadd;
                break;
            }
            seg_offset = 0;
            next_seg += 1;

            if(SINGLE) 
                break; // only load at most one segment at a time
        }

        return id;
    }

    __device__ __forceinline__ bool finished() const {
        return (next_seg >= seg_loaded) && (istart >= iend);
    }

    __device__ __forceinline__ int nids_loaded() const {
        return num_loaded;
    }
};

template <typename T>
struct PrefetchList {
    T const* __restrict__ data;
    T local;
    int icur, iend;
    int loff;

    __device__ __forceinline__
    PrefetchList(const T* __restrict__ data_, int istart_, int iend_) : data(data_)
    {
        restart(istart_, iend_);
    }

    __device__ __forceinline__ void restart(int istart_, int iend_) {
        icur = istart_;
        iend = iend_;
        loff = 0;

        int idx = icur + (threadIdx.x & (warpSize - 1));;
        if (idx < iend) {
            local = data[idx];
        }
    }

    __device__ __forceinline__ T next() {
        if(loff >= warpSize) {
            icur += warpSize;
            int idx = icur + (threadIdx.x & (warpSize - 1));
            if(idx < iend) {
                local = data[idx];
            }
            loff = 0;
        }
        
        return __shfl_sync(0xFFFFFFFF, local, loff++);
    }

    __device__ __forceinline__ bool finished() const {
        return icur + loff >= iend;
    }
};

template<class A, class B>
struct Pair { A first; B second; };

// ---- 2-array specialization ----
template <typename T0, typename T1>
struct PrefetchList2 {
    T0 const* __restrict__ data0;
    T1 const* __restrict__ data1;
    T0 local0;
    T1 local1;
    int icur, iend;
    int loff;

    __device__ __forceinline__
    PrefetchList2(const T0* __restrict__ d0,
                 const T1* __restrict__ d1,
                 int istart_, int iend_)
        : data0(d0), data1(d1)
    {
        restart(istart_, iend_);
    }

    __device__ __forceinline__ void restart(int istart_, int iend_) {
        icur = istart_;
        iend = iend_;
        loff = 0;

        int idx = icur + (threadIdx.x & (warpSize - 1));
        if (idx < iend) {
            local0 = data0[idx];
            local1 = data1[idx];
        }
    }

    __device__ __forceinline__ Pair<T0,T1> next() {
        if (loff >= warpSize) {
            icur += warpSize;
            int idx = icur + (threadIdx.x & (warpSize - 1));
            if (idx < iend) {
                local0 = data0[idx];
                local1 = data1[idx];
            }
            loff = 0;
        }
        unsigned mask = __activemask();
        int src = loff++;
        T0 a = __shfl_sync(mask, local0, src);
        T1 b = __shfl_sync(mask, local1, src);
        return {a, b};
    }

    __device__ __forceinline__ bool finished() const {
        return icur + loff >= iend;
    }
};

template <class T>
__device__ inline T warp_broadcast(const T& x, int src_lane, unsigned mask = __activemask()) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "T must be trivially copyable (POD-like).");

    // Fast paths for common sizes
    if constexpr (sizeof(T) == 4) {
        uint32_t w;
        memcpy(&w, &x, 4);
        w = __shfl_sync(mask, w, src_lane);
        T out;
        memcpy(&out, &w, 4);
        return out;
    } else if constexpr (sizeof(T) == 8) {
        uint32_t w0, w1;
        memcpy(&w0, &x, 4);
        memcpy(&w1, (const char*)&x + 4, 4);
        w0 = __shfl_sync(mask, w0, src_lane);
        w1 = __shfl_sync(mask, w1, src_lane);
        T out;
        memcpy(&out, &w0, 4);
        memcpy((char*)&out + 4, &w1, 4);
        return out;
    } else {
        constexpr int W = (sizeof(T) + 3) / 4;  // number of 32-bit words
        uint32_t in_words[W];
        memcpy(in_words, &x, sizeof(T));
        uint32_t out_words[W];
        #pragma unroll
        for (int i = 0; i < W; ++i)
            out_words[i] = __shfl_sync(mask, in_words[i], src_lane);
        T out;
        memcpy(&out, out_words, sizeof(T));
        return out;
    }
}

template <typename T>
struct PointedPrefetchList {
    const T* __restrict__ data;
    const int* __restrict__ ptrs;

    T local;
    int icur, iend;
    int loff;
    int plocal;
    int pcur;

    __device__ __forceinline__ PointedPrefetchList(const int* __restrict__ ptrs_, const T* __restrict__ data_, const int istart_, const int iend_) {
        data = data_;
        ptrs = ptrs_;
        icur = istart_;
        iend = iend_;
        loff = 0;

        if(icur + threadIdx.x < iend) {
            plocal = ptrs[icur + threadIdx.x];
            local = data[plocal];
        }
    }

    __device__ __forceinline__ T next() {
        if(loff >= warpSize) {
            icur += warpSize;
            if(icur + threadIdx.x < iend) {
                plocal = ptrs[icur + threadIdx.x];
                local = data[plocal];
            }
            loff = 0;
        }
        
        pcur = __shfl_sync(0xFFFFFFFF, plocal, loff);
        // return __shfl_sync(0xFFFFFFFF, local, loff++);
        return warp_broadcast<T>(local, loff++);
    }
    __device__ __forceinline__ int current_ptr() const {
        return pcur;
    }

    __device__ __forceinline__ bool finished() const {
        return icur + loff >= iend;
    }
};


// Utility: next power-of-two >= n (cap at MAX_THREADS)
// Smallest power of two >= x (32-bit)
__device__ __forceinline__ unsigned int next_pow2_u32(unsigned int x) {
    if (x == 0) return 1u;                 // define as 1 for x=0
    if (x > 0x80000000u) return 0u;        // overflow (no 33rd bit in u32)
    return 1u << (32 - __clz(x - 1));      // __clz: count leading zeros
}

struct KV {
    float k;   // key (float32)
    int32_t v; // value (int32)
};

// Compare-and-swap: ascending on key; swap value with it
__device__ __forceinline__
void cas(float &ak, int32_t &av, float &bk, int32_t &bv, bool dir /*true=>ascending*/) {
    // Treat NaN as +inf so they go to the end in ascending
    float ak_eff = isnan(ak) ? INFTY : ak;
    float bk_eff = isnan(bk) ? INFTY : bk;

    bool greater = (ak_eff > bk_eff);
    if (greater == dir) {
        float tk = ak; ak = bk; bk = tk;
        int32_t tv = av; av = bv; bv = tv;
    }
}

template <typename T>
void comparator(std::vector<T> &a, std::size_t i, std::size_t j) {
  if (i < j && j < a.size() && a[j] < a[i])
    std::swap(a[i], a[j]);
}

template <typename T> void impBitonicSort(std::vector<T> &a) {
  // Iterate k as if the array size were rounded up to the nearest power of two.
  for (std::size_t k = 2; (k >> 1) < a.size(); k <<= 1) {
    for (std::size_t i = 0; i < a.size(); i++)
      comparator(a, i, i ^ (k - 1));
    for (std::size_t j = k >> 1; 0 < j; j >>= 1)
      for (std::size_t i = 0; i < a.size(); i++)
        comparator(a, i, i ^ j);
  }
}

int main() {
  for (int n = 2; n <= 8; n++) {
    std::vector<int> unsorted(n);
    std::iota(unsorted.begin(), unsorted.end(), 0);
    do {
      auto sorted = unsorted;
      impBitonicSort(sorted);
      if (!std::is_sorted(sorted.begin(), sorted.end())) {
        for (int i : unsorted)
          std::printf(" %d", i);
        std::printf("\n");
        return 1;
      }
    } while (std::next_permutation(unsorted.begin(), unsorted.end()));
  }
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
