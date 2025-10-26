#ifndef CUSTOM_JAX_TREE_H
#define CUSTOM_JAX_TREE_H
#include <type_traits>

// #include "xla/ffi/api/ffi.h"

// #include "shared_utils.cuh"
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <math_constants.h>

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ int32_t float_xor_msb(float a, float b) {
    // Finds the most significant bit that differs between x and y
    // For floating point numbers we need to treat the exponent and the mantissa differently:
    // If the exponent differs, then the (power of two) of the difference is given by the larger
    // exponent.
    // If the exponent is the same, then we need to compare the mantissas. The (power of two) of the
    // difference is then given by the differing bit in the mantissa, offset by the exponent

    if (signbit(a) != signbit(b)) {
        return 128;  // The sign is the highest significant bit
    }
    int32_t a_bits = __float_as_int(fabsf(a));
    int32_t b_bits = __float_as_int(fabsf(b));

    int32_t a_exp = (a_bits >> 23) - 127;
    int32_t b_exp = (b_bits >> 23) - 127;

    if (a_exp == b_exp) { // If both floats have the same exponent, we need to compare mantissas
        // clz counts bit-zeros from the left. There will be always 8 leading zeros due to the
        // exponent
        return a_exp + (8 - __clz(a_bits ^ b_bits)); 
    }
    else { // If exponents differ, return the larger exponent
        return max(a_exp, b_exp);
    }
}

struct PosId {
    float3 pos;
    int32_t id;
};

__device__ __forceinline__ bool z_pos_less(float3 pos1, float3 pos2)
{
    int msb_x = float_xor_msb(pos1.x, pos2.x);
    int msb_y = float_xor_msb(pos1.y, pos2.y);
    int msb_z = float_xor_msb(pos1.z, pos2.z);

    int ms_dim = (msb_x >= msb_y && msb_x >= msb_z) ? 0 : ((msb_y >= msb_z) ? 1 : 2);

    if (ms_dim == 0) return pos1.x < pos2.x;
    if (ms_dim == 1) return pos1.y < pos2.y;
    return pos1.z < pos2.z;
}

struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId &a, const PosId &b) {
        return z_pos_less(a.pos, b.pos);
    }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           Zorder Sort                                          */
/* ---------------------------------------------------------------------------------------------- */

__global__ void PosKeyArangeKernel(const float3* pos_in, PosId *keyid_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keyid_out[idx].pos = pos_in[idx];
        keyid_out[idx].id = idx;
    }
}

std::string PosZorderSort(
    cudaStream_t stream, 
    const float3* pos_in, 
    PosId* pos_id_out,
    int* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
) {
    // Initialize indices 0, 1, 2, ..., size-1
    PosKeyArangeKernel<<< div_ceil(size, block_size), block_size, 0, stream>>>(pos_in, pos_id_out, size);

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
    size_t required_storage_bytes;
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(nullptr, required_storage_bytes, pos_id_out, size, PosIdLess());
    
    // Check if the provided buffer is large enough
    if (tmp_bytes < required_storage_bytes) {
        return std::string(
            "The buffer in ZorderSort is too small. Please contact me if this check fails.") +
            std::string(" Have: ") + std::to_string(tmp_bytes) +
            std::string(". Required: ") + std::to_string(required_storage_bytes) +
            std::string(". Diff: ") + std::to_string((long long)required_storage_bytes - (long long)tmp_bytes);
    }
    
    // This is how we would allocate if we could. (Note that doing this breaks jit in some cases!)
    // cudaMallocAsync(&d_temp_storage, required_storage_bytes, stream); 

    // Run the sort
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(tmp_buffer, required_storage_bytes, pos_id_out, size, PosIdLess(), stream);
    
    return std::string();
}

__device__ __forceinline__ int32_t msb_diff_level(const float3 &p1, const float3 &p2) {
    int msb_x = float_xor_msb(p1.x, p2.x);
    int msb_y = float_xor_msb(p1.y, p2.y);
    int msb_z = float_xor_msb(p1.z, p2.z);

    // The level is given by the most significant differing bit
    // but offset according to the dimension
    return max(3*msb_x+3, max(3*msb_y+2, 3*msb_z+1));
}

/* ---------------------------------------------------------------------------------------------- */
/*                            Tree building (may be depreacated later)                            */
/* ---------------------------------------------------------------------------------------------- */

struct NodePointers {
    int32_t* levels;
    int32_t* lbound;
    int32_t* rbound;
    int32_t* lchild;
    int32_t* rchild;
};


__global__ void KernelBinarySearchLeftParent(const float3* pos_in, NodePointers nodes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid_thread = (idx < n-1);

    // Node indices are offset by 1, because we put a fake node at the beginning
    // but we don't launch the kernel for it
    int node = idx + 1; 

    int target_level, lvl_left, lvl_right;
    int lbound, rbound;
    float3 p1, p2;

    if (valid_thread) {
        // Calculate the level difference of our considered set of two points (=node)
        p1 = pos_in[idx];
        p2 = pos_in[idx + 1];

        target_level = msb_diff_level(p1, p2);

        // We do a binary search, trying to find the closest point to the left
        // that has a level difference of at least `level`
        int imin = -1, imax = idx+1;
        lvl_left = 388;
        while (imin+1 < imax) {
            int itest = (imin + imax) / 2;
            lvl_left = msb_diff_level(pos_in[itest], p2);
            if (lvl_left > target_level) {
                imin = itest;
            } else {
                imax = itest;
            }
        }

        // Our array has two fake nodes at the beginning and end
        // that's why we have to offset the indices by 1
        lbound = imin+1;

        if(imin >= 0)
            lvl_left = msb_diff_level(p1, pos_in[imin]);
        else
            lvl_left = 388;
    }
    
    __syncthreads(); // Synchronize to reduce thread divergence

    if (valid_thread) {
        // Now find the right side parent
        int imin = idx, imax = n;
        lvl_right = 388;
        while (imin+1 < imax) {
            int itest = (imin + imax) / 2;
            lvl_right = msb_diff_level(p1, pos_in[itest]);
            if (lvl_right > target_level) {
                imax = itest;
            } else {
                imin = itest;
            }
        }

        rbound = imin+1;

        if(rbound < n)
            lvl_right = msb_diff_level(p1, pos_in[rbound]);
        else
            lvl_right = 388;
    }

    __syncthreads();

    if (valid_thread) {
        nodes.levels[node] = target_level;
        nodes.lbound[node] = lbound;
        nodes.rbound[node] = rbound;
        
        // The parent of each node is the lower one of the two boundary nodes
        if(lvl_left <= lvl_right) {
            nodes.rchild[lbound] = node;
        } else {
            nodes.lchild[rbound] = node;
        }
    }
}
__global__ void KernelInitialize(NodePointers nodes, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nnodes = n + 1;

    if (idx >= Nnodes)
        return;

    // We have fake nodes at the beginning and end of the array
    // to simplify walking the tree
    if((idx == 0) || (idx == Nnodes - 1)) {
        nodes.lbound[idx] = 0;
        nodes.rbound[idx] = Nnodes-1;
        nodes.levels[idx] = 388;
        nodes.lchild[idx] = idx; // Point to itself, may be overwritten later
        nodes.rchild[idx] = idx; // Point to itself, may be overwritten later
    }
    else {
        // indices <= 0 correspond to leafs (=particles)
        // by defaults node's children point to particles,
        // but about half of them will be overwritten by nodes later
        nodes.lchild[idx] = -idx + 1;
        nodes.rchild[idx] = -idx;
    }
}

void BuildZTree(
    cudaStream_t stream, 
    const float3* pos_in,
    int* outputs,
    size_t size,
    size_t block_size
) {
    // size_t n = pos_in.element_count()/3;
    size_t Nnodes = size + 1;

    // Output will be (5, Nnodes) array with different types of information in the first axis
    // Create some easier readable pointers that start at offset locations in the output
    NodePointers nodes;
    nodes.levels = outputs;
    nodes.lbound = outputs + Nnodes;
    nodes.rbound = outputs + 2 * Nnodes;
    nodes.lchild = outputs + 3 * Nnodes;
    nodes.rchild = outputs + 4 * Nnodes;
    
    KernelInitialize<<< div_ceil(Nnodes, block_size), block_size, 0, stream>>>(nodes, size);

    KernelBinarySearchLeftParent<<< div_ceil(size-1, block_size), block_size, 0, stream>>>(pos_in, nodes, size);
}

struct PosN {
    float3 pos;
    int32_t n;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         SummarizeLeaves                                        */
/* ---------------------------------------------------------------------------------------------- */

__global__ void SummarizeLeaves(
    const PosN* xnleaf,
    const int* nleaves_filled,
    int32_t* split_flags,
    int max_size,
    int n_leaves,
    int scan_size
) {
    int nfilled = nleaves_filled[0];

    // Finds splitting points where the group of particles between each splitting point
    // can be summarized into a single leaf node that represents <= max_size particles
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data preceding and following our block into shared memory
    int nload = blockDim.x + 2*scan_size + 1;
    extern __shared__ unsigned char smem[];
    PosN*   xn = reinterpret_cast<PosN*>(smem);
    int32_t* level = reinterpret_cast<int32_t*>(xn + nload);

    int ioff = blockIdx.x * blockDim.x - scan_size - 1;
    
    // Note: we may load some points duplicate at the boundary, but that is ok (they will have 
    // level 0). Keeping it this way simplifies the indexing logic later
    for(int i = threadIdx.x; i < nload; i += blockDim.x) {
        int ifrom = ioff + i;
        if(ifrom < 0)
            xn[i] = {make_float3(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F), 0};
        else if(ifrom >= nfilled)
            xn[i] = {make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F), 0};
        else
            xn[i] = xnleaf[ifrom];
    }

    __syncthreads();
    for(int i = threadIdx.x; i < nload-1; i += blockDim.x) {
        level[i] = msb_diff_level(xn[i].pos, xn[i + 1].pos);
    }
    __syncthreads();

    // Find the boundaries of each node
    int idx = threadIdx.x + scan_size;
    int mylevel = level[idx];
    int lsize = 0;
    for(int i = idx - scan_size; i < idx; i++) {
        lsize = xn[i+1].n + (level[i] >= mylevel ? 0 : lsize);
    }
    int rsize = 0;
    for(int i = idx + scan_size; i > idx; i--) {
        rsize = xn[i].n + (level[i] >= mylevel ? 0 : rsize);
    }

    // Each maximum size node that is <= max_size is bounded by nodes that are > max_size
    // Therefore, we can find their splitting points by simply flagging all nodes that are > max_size
    bool is_split = lsize + rsize > max_size;

    // Additionally we set the beginning and end points to be splits
    is_split |= node_idx == 0; 
    is_split |= node_idx == nfilled;
    is_split &= node_idx <= nfilled;

    if (node_idx <= n_leaves) {
        split_flags[node_idx] = is_split ? mylevel : -1000;
    }
}

// ffi::Error HostSummarizeLeaves(
//     cudaStream_t stream, 
//     ffi::Buffer<ffi::F32> xnleaf,
//     ffi::Buffer<ffi::S32> nleaves_filled,
//     ffi::ResultBuffer<ffi::S32> flags_split,
//     size_t max_size,
//     size_t block_size,
//     size_t scan_size
// ) {
//     size_t n_leaves = xnleaf.element_count()/4;

//     size_t alloc_bytes = (block_size + 2*scan_size + 1) * (sizeof(PosN) + sizeof(int32_t));

//     KernelSummarizeLeaves<<< div_ceil(n_leaves+1, block_size), block_size, alloc_bytes, stream>>>(
//         reinterpret_cast<const PosN*>(xnleaf.typed_data()),
//         nleaves_filled.typed_data(),
//         flags_split->typed_data(),
//         max_size,
//         n_leaves,
//         scan_size
//     );

//     cudaError_t last_error = cudaGetLastError();
//     if (last_error != cudaSuccess) {
//         return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
//     }
//     return ffi::Error::Success();
// }

/* ---------------------------------------------------------------------------------------------- */
/*                                         Search Sorted Z                                        */
/* ---------------------------------------------------------------------------------------------- */

__global__ void KernelSearchSortedZ(
    const float3* posz_have,
    size_t n_have,
    const float3* posz_query,
    size_t n_query,
    int32_t* indices,
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
        if (z_pos_less(posz_have[itest], xquery)) {
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
            iout = z_pos_less(posz_have[0], xquery) ? 1 : 0;
        else if(imin == n_have - 1)
            iout = z_pos_less(posz_have[n_have - 1], xquery) ? n_have : n_have - 1;
        else
            iout = imin + 1;
    }

    indices[idx] = iout;
}

// ffi::Error SearchSortedZHost(
//     cudaStream_t stream, 
//     ffi::Buffer<ffi::F32> posz_have,
//     ffi::Buffer<ffi::F32> posz_query,
//     ffi::ResultBuffer<ffi::S32> indices,
//     bool leaf_search,
//     size_t block_size
// ) {
//     size_t n_have = posz_have.element_count()/3;
//     size_t n_query = posz_query.element_count()/3;
//     float3* posz_ptr = reinterpret_cast<float3*>(posz_have.typed_data());
//     float3* posz_query_ptr = reinterpret_cast<float3*>(posz_query.typed_data());

//     KernelSearchSortedZ<<< div_ceil(n_query, block_size), block_size, 0, stream>>>(
//         posz_ptr,
//         n_have,
//         posz_query_ptr,
//         n_query,
//         indices->typed_data(),
//         leaf_search
//     );

//     cudaError_t last_error = cudaGetLastError();
//     if (last_error != cudaSuccess) {
//         return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
//     }
//     return ffi::Error::Success();
// }

#endif // CUSTOM_JAX_TREE_H