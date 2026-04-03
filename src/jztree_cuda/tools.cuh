#ifndef TOOLS_CUH
#define TOOLS_CUH

#include "common/data.cuh"
#include "common/math.cuh"

#define INFTY  INFINITY //__int_as_float(0x7f800000)

/* ---------------------------------------------------------------------------------------------- */
/*                                        Ragged Transpose                                        */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ int find_segment_binary(const int64_t* seg_spl_out, int size_seg, int64_t idx)
{
    // seg_spl_out has length size_seg+1, monotonically increasing
    int lo = 0, hi = size_seg - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int64_t a = seg_spl_out[mid];
        int64_t b = seg_spl_out[mid + 1];
        if (idx < a) hi = mid - 1;
        else if (idx >= b) lo = mid + 1;
        else return mid;
    }
    return -1;
}


__global__ void RearangeSegments(
    const uint8_t* data_in,
    const int64_t* seg_spl_out,
    const int64_t* seg_offset_in,
    uint8_t* data_out,
    int64_t size,
    int64_t size_seg,
    int64_t dtype_bytes
) {
    int64_t idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    int seg_idx = find_segment_binary(seg_spl_out, size_seg, idx);
    
    int64_t ifrom;
    if(seg_idx >= 0)
        ifrom = seg_offset_in[seg_idx] + (idx - seg_spl_out[seg_idx]);
    else
        ifrom = idx; // for out of segment range indices we simply copy from the input
    
    if (ifrom < 0 || ifrom >= size) return;

    int64_t ifrom_bytes = ifrom * dtype_bytes;
    int64_t ito_bytes = idx * dtype_bytes;
    for(int64_t ibyte=0; ibyte<dtype_bytes; ibyte++) {
        // Could improve coalescence here by switching adaptively to larger data-types
        data_out[ito_bytes + ibyte] = data_in[ifrom_bytes + ibyte];
    }
}

__global__ void MapInRange(
    const int* range,
    const int* input,
    const int* map,
    int* output
) {
    int imin = range[0], imax = range[1];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if((idx >= range[0]) && (idx < range[1])) {
        output[idx] = map[input[idx]];
    }
}

#endif