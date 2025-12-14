#ifndef COMMON_ITERATORS_CUH
#define COMMON_ITERATORS_CUH

struct SegmentManager {
    // Manages a pointer based segment list so that it can be traversed with threads
    // almost as if it was a linear structure
    // each segment goes from isplits[i] to isplits[i+1]
    // inodes contains the indices of the segments to be processed
    // istart, iend define the range of inodes to be processed

    int istart, iend;                     // indices into inodes[]
    const int* __restrict__ inodes;       // segment indices to visit (defaults to 0,1,... if nullptr)
    const int* __restrict__ isplits;      // global segment offsets
    int2* __restrict__ segments;          // -> shared buffer [NLOAD], allocated externally
    int seg_loaded;                       // how many segments resident in shared

    // internal state:
    int next_seg = 0;
    int seg_offset = 0;
    int num_loaded = 0;
    int nload_max = 0;
    
    __device__ __forceinline__ SegmentManager(const int* inodes_, const int* isplits_, int2* segments_, int istart_, int iend_, int nload_max_) {
        inodes = inodes_;
        isplits = isplits_;
        segments = segments_;

        istart = istart_;
        iend = iend_;
        nload_max = nload_max_;
        
        loadSegments();
    }

    __device__ __forceinline__ void loadSegments() {
        __syncthreads();
        seg_loaded = min(nload_max, iend-istart);
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
    
    __device__ __forceinline__ int next() {
        int id = -1;
        // Go through the segments until we are sure that every thread has something
        num_loaded = 0;
        while (!finished())
        {
            // If required, load more segments
            if(next_seg >= seg_loaded) {
                loadSegments();
            }

            int2 seg = segments[next_seg];
            int nadd = min(seg.y - seg_offset, blockDim.x - num_loaded);
            if((threadIdx.x >= num_loaded ) && (threadIdx.x < num_loaded + nadd)) {
                id = seg.x + seg_offset + (threadIdx.x - num_loaded);
            }
            num_loaded += nadd;
            if(num_loaded >= blockDim.x) {
                seg_offset += nadd;
                break;
            }
            seg_offset = 0;
            next_seg += 1;
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

#endif