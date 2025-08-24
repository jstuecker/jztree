#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

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

template <int nload>
__device__ struct SegmentManager {
    // Manages a pointer based segment list so that it can be traversed with threads
    // almost as if it was a linear structure
    // each segment goes from isplits[i] to isplits[i+1]
    // inodes contains the indices of the segments to be processed
    // istart, iend define the range of inodes to be processed

    int istart, iend;                     // indices into inodes[]
    const int* __restrict__ inodes;       // segment indices to visit
    const int* __restrict__ isplits;      // global segment offsets
    int2* __restrict__ segments;          // -> shared buffer [nload], allocated externally
    int seg_loaded;                       // how many segments resident in shared

    // internal state:
    int next_seg = 0;
    int seg_offset = 0;
    int num_loaded = 0;
    
    __device__ __forceinline__ void init(const int* inodes_, const int* isplits_, int2* segments_, int istart_, int iend_) {
        inodes = inodes_;
        isplits = isplits_;
        segments = segments_;

        istart = istart_;
        iend = iend_;
        
        loadSegments();
    }

    __device__ __forceinline__ void loadSegments() {
        seg_loaded = min(nload, iend-istart);
        if(threadIdx.x < seg_loaded) {
            int segment_idx = inodes[istart + threadIdx.x];
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
        // If required, load more segments
        if(next_seg >= seg_loaded) {
            loadSegments();
        }

        int id = -1;
        // Go through the segments until we are sure that every thread has something
        num_loaded = 0;
        while (next_seg < seg_loaded)
        {
            int2 seg = segments[next_seg]; // {start, length}
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