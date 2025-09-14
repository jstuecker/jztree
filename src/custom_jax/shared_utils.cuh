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
__device__ struct SegmentManager {
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
__device__ struct PrefetchList {
    T const* __restrict__ data;
    T local;
    int icur, iend;
    int loff;

    __device__ __forceinline__ PrefetchList(const T* __restrict__ data_, const int istart_, const int iend_) {
        data = data_;
        icur = istart_;
        iend = iend_;
        loff = 0;

        if(icur + threadIdx.x < iend) {
            local = data[icur + threadIdx.x];
        }
    }

    __device__ __forceinline__ T next() {
        if(loff >= blockDim.x) {
            icur += blockDim.x;
            if(icur + threadIdx.x < iend) {
                local = data[icur + threadIdx.x];
            }
            loff = 0;
        }
        
        return __shfl_sync(0xFFFFFFFF, local, loff++);
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
__device__ struct PointedPrefetchList {
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
        if(loff >= blockDim.x) {
            icur += blockDim.x;
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
void cas(KV &a, KV &b, bool dir /*true=>ascending*/) {
    // If dir==true we want smaller key first
    // Handle NaNs: treat NaN as +inf so they float to the end in ascending
    float ak = a.k, bk = b.k;
    bool a_nan = isnan(ak);
    bool b_nan = isnan(bk);
    float ak_eff = a_nan ? INFTY : ak;
    float bk_eff = b_nan ? INFTY : bk;

    bool greater = (ak_eff > bk_eff);
    // If equal keys and you prefer stability, you could compare original index carried in v or a separate index.
    if (greater == dir) {
        KV tmp = a; a = b; b = tmp;
    }
}

// Bitonic sort network over shared memory buffer of length nPow2
__device__ void bitonic_sort_shared(KV* sdata, int nPow2) {
    // nPow2 should be a power of two (pad beforehand).
    for (int k = 2; k <= nPow2; k <<= 1) {
        // ascending overall
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();  // ensure previous stage is complete

            // Stride over all indices in this block's shared segment
            for (int i = threadIdx.x; i < nPow2; i += blockDim.x) {
                int ixj = i ^ j;
                // Only one side does the compare-and-swap to avoid double work
                if (ixj > i) {
                    bool up = ((i & k) == 0); // true => ascending for this pair
                    cas(sdata[i], sdata[ixj], up);
                }
            }
        }
    }
    __syncthreads();
}

// -------- Kernel: segmented sort (keys + values) --------
__global__ void segmented_bitonic_sort_kv(
    float* __restrict__ keys,             // in/out float32 keys
    int32_t* __restrict__ values,         // in/out int32 values (permute with keys)
    const int32_t* __restrict__ offsets,  // len = num_segments + 1
    int32_t num_segments,
    int32_t max_sort 
)
{
    int seg = blockIdx.x;
    if (seg >= num_segments) return;

    int32_t start = offsets[seg];
    int32_t end   = offsets[seg + 1];
    int32_t len   = max(end - start, 0);
    if (len <= 1) return;

    extern __shared__ unsigned char smem[];
    KV* sdata = reinterpret_cast<KV*>(smem);

    int tid = threadIdx.x;
    // int nPow2 = next_pow2_cap(len);
    int nPow2 = min(next_pow2_u32(len), max_sort);

    // Padding: +inf key so padded items sink to the end (ascending).
    // Value can be anything (kept with the pad).
    const KV PAD = {1e8, 0};

    // Load keys/values into shared memory (with padding)
    for (int i = tid; i < nPow2; i += blockDim.x) {
        KV kv = PAD;
        if (i < len) {
            kv.k = keys[start + i];
            kv.v = values[start + i];
        }
        sdata[i] = kv;
    }
    __syncthreads();

    // Sort
    bitonic_sort_shared(sdata, nPow2);

    // Store back only the real elements
    for (int i = tid; i < min(len, nPow2); i += blockDim.x) {
        keys[start + i]   = sdata[i].k;
        values[start + i] = sdata[i].v;
    }
}