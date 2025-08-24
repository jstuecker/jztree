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
    int istart, iend;
    int* inodes;
    int* isplits;
    __shared__ int2 segments[nload];
    int linear_idx;

    int next_seg = 0;
    int seg_loaded = 0;

    bool done = false;
    
    __device__ inline void init(int* inodes_, int* isplits_, int istart, int iend) {
        inodes = inodes_;
        isplits = isplits_;

        istart = istart;
        iend = iend;
        linear_idx = threadIdx.x;
        
        loadSegments();
    }

    __device__ inline void loadSegments() {
        done = (istart >= iend);

        __syncthreads();
        seg_loaded = min(nload, iend-istart);
        if(threadIdx.x < seg_loaded) {
            int segment_idx = inodes[istart + threadIdx.x];
            int istart = isplits[segment_idx];
            int iend = isplits[segment_idx + 1];
            int2 segment = {istart, iend - istart};
            segments[threadIdx.x] = segment;
            // atomicAdd(&num_loaded, segment.y);
        }
        istart += seg_loaded;
        next_seg = 0;
        __syncthreads();
    }
    
    int seg_offset = 0;
    // int num_ids_loaded = 0;
    __device__ inline int next() {
        int id = -1;
        // Go through the segments until we are sure that every thread has something
        int num_ids_loaded = 0;
        while (next_seg < seg_loaded)
        {
            int2 seg = segments[next_seg]; // {start, length}
            int nadd = min(seg.y - seg_offset, blockDim.x - num_ids_loaded);
            if((threadIdx.x >= num_ids_loaded ) && (threadIdx.x < num_ids_loaded + nadd)) {
                id = seg.x + threadIdx.x - seg_offset;
            }
            num_ids_loaded += nadd;
            if(num_ids_loaded >= blockDim.x) {
                seg_offset += nadd;
                break;
            }
            seg_offset = 0;
            next_seg += 1;
        }
        // If required, load more segments
        if(next_seg >= seg_loaded) {
            loadSegments();
        }

        return id;
    }
};