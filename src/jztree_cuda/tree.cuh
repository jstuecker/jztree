#ifndef TREE_H
#define TREE_H

#include <cub/cub.cuh>
#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                           Zorder Sort                                          */
/* ---------------------------------------------------------------------------------------------- */

// Wrapper of the z-order comparison function to use with CUB
struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId &a, const PosId &b) {
        return z_pos_less(a.pos, b.pos);
    }
};

// Prepare keys and ids for sorting
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
    size_t required_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(nullptr, required_storage_bytes, pos_id_out, size, PosIdLess());
    
    // Check if the provided buffer is large enough
    if (tmp_bytes < required_storage_bytes) {
        return std::string(
            "The buffer in ZorderSort is too small. Please contact me if this check fails.") +
            std::string(" Have: ") + std::to_string(tmp_bytes) +
            std::string(". Required: ") + std::to_string(required_storage_bytes) +
            std::string(". Diff: ") + std::to_string((long long)required_storage_bytes - (long long)tmp_bytes);
    }

    // Run the sort
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(tmp_buffer, required_storage_bytes, pos_id_out, size, PosIdLess(), stream);
    
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

/* ---------------------------------------------------------------------------------------------- */
/*                                      DetectLeafBoundaries                                      */
/* ---------------------------------------------------------------------------------------------- */

__global__ void FlagLeafBoundaries(
    const float3* posz,
    const int* lvl_bound,
    const int* npart,
    int8_t* split_flags,
    int max_size,
    int size_part,
    int scan_size
) {
    // Finds splitting points where the group of particles between each splitting point
    // can be summarized into a single leaf node that represents <= max_size particles
    // The node_idx splitting point represents the boundary between x[node_idx-1] and x[node_idx]
    int nump = npart[0];

    int lbound_l = lvl_bound[0], lbound_r = lvl_bound[1];

    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data preceding and following our block into shared memory
    int nload = blockDim.x + 2*scan_size + 1;
    extern __shared__ int32_t level[];

    int ioff = blockIdx.x * blockDim.x - scan_size - 1;
    
    __syncthreads();
    for(int i = threadIdx.x; i < nload-1; i += blockDim.x) {
        int ipart = ioff + i;
        if(ipart < 0)
            level[i] = lbound_l;
        else if(ipart + 1 >= nump)
            level[i] = lbound_r;
        else
            level[i] = msb_diff_level(posz[ipart], posz[ipart + 1]);
    }
    __syncthreads();

    // Find the boundaries of each node
    int idx = threadIdx.x + scan_size;
    int mylevel = level[idx];
    int lsize = 0;
    for(int i = idx - scan_size; i < idx; i++) {
        lsize = (level[i] >= mylevel ? 0 : lsize) + 1;
    }
    int rsize = 0;
    for(int i = idx + scan_size; i > idx; i--) {
        rsize = (level[i] >= mylevel ? 0 : rsize) + 1;
    }

    // Each maximum size node that is <= max_size is bounded by nodes that are > max_size
    // Therefore, we can find their splitting points by simply flagging all nodes that are > max_size
    bool is_split = lsize + rsize > max_size;

    // If nodes hit a domain boundary and are larger than the boundary level, we always flag them
    int max_lvl_left = lvl_bound[0], max_lvl_right = lvl_bound[1];
    is_split = is_split | ((mylevel > lvl_bound[0]) && (node_idx - lsize <= 0));
    is_split = is_split | ((mylevel > lvl_bound[1]) && (node_idx + rsize >= nump));

    // Additionally we set the beginning and end points to be splits
    is_split &= node_idx <= nump;
    is_split |= node_idx == 0; 
    is_split |= node_idx == nump;

    if (node_idx < size_part + 1) {
        split_flags[node_idx] = is_split;
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Tree Building                                         */
/* ---------------------------------------------------------------------------------------------- */

__global__ void FindNodeBoundaries(
    const float3* pos_in,
    const float3* pos_boundary,
    const int *nleaves,
    int32_t* nodes_levels,
    int32_t* nodes_lbound,
    int32_t* nodes_rbound,
    const int size_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nleaves[0];
    int nnodes = n + 1;

    if (idx >= size_nodes)
        return;
    if (idx >= nnodes) {
        // Set values for undefined nodes
        nodes_levels[idx] = -1000;
        nodes_lbound[idx] = n;
        nodes_rbound[idx] = n;

        return;
    }

    int target_level, lvl_left, lvl_right;
    int lbound, rbound;
    float3 p1, p2;

    // Calculate the level difference of our considered set of two points (=node)
    if(idx == 0)
        p1 = pos_boundary[0];
    else
        p1 = pos_in[idx - 1];
    if(idx == nnodes-1)
        p2 = pos_boundary[1];
    else
        p2 = pos_in[idx];

    // Our node includes at least [idx-1, idx] and it goes up to the left until a
    // point has a level difference that is larger han target_level:
    target_level = msb_diff_level(p1, p2);

    if(msb_diff_level(pos_in[0], p2) <= target_level) {
        // Node goes until left-domain boundary... We have no left parent
        lbound = 0;
        lvl_left = 388; // larger than any possible level
    } else {
        // ibefore is the an index (left) outside of the node, iinside is an index inside
        // We do a binary search until they lie next to each other:
        int ibefore = 0, iinside = idx;
        lvl_left = 388;
        while (ibefore+1 < iinside) {
            int itest = (ibefore + iinside) / 2;
            lvl_left = msb_diff_level(pos_in[itest], p2);
            if (lvl_left > target_level) {
                ibefore = itest;
            } else {
                iinside = itest;
            }
        }
        lvl_left = msb_diff_level(p1, pos_in[ibefore]);
        lbound = iinside;
    }


    // Now find the right side parent
    if(msb_diff_level(p1, pos_in[n-1]) <= target_level)
    {
        rbound = n;
        lvl_right = 388;
    }
    else
    {
        int iinside = idx-1, iafter = n-1;
        lvl_right = 388;
        while (iinside+1 < iafter) {
            int itest = (iinside + iafter) / 2;
            lvl_right = msb_diff_level(p1, pos_in[itest]);
            if (lvl_right > target_level) {
                iafter = itest;
            } else {
                iinside = itest;
            }
        }

        rbound = iafter;
        lvl_right = msb_diff_level(p1, pos_in[rbound]);
    }

    nodes_levels[idx] = target_level;
    nodes_lbound[idx] = lbound;
    nodes_rbound[idx] = rbound;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Patching Nodes                                         */
/* ---------------------------------------------------------------------------------------------- */

template<bool left>
__global__ void KernelInitLevels(
    const int* irange,
    int32_t* index_of_lvl,
    const int lvl_min,
    const int lvl_max
) {
    int ilvl = blockDim.x * blockIdx.x + threadIdx.x;
    if(left && (ilvl <= lvl_max - lvl_min))
        index_of_lvl[ilvl] = irange[1];
    else if(!left && (ilvl <= lvl_max - lvl_min))
        index_of_lvl[ilvl] = irange[0];
}

template<bool left>
__global__ void KernelGetBoundaryExtendPerLevel(
    const float3* pos_ref,
    const int* irange,
    const float3* posz,
    int32_t* index_of_lvl,
    const int lvl_min,
    const int lvl_max
) {
    // Finds for each level the first point that has a higher bit difference
    // to a reference point than the considered level.
    // This method is useful for patching together domains.

    int idx = irange[0] + blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= irange[1])
        return;

    float3 x0 = pos_ref[0];
    float3 x1 = posz[idx];

    int lvl = msb_diff_level(x0, x1);

    int ilvl = max(min(lvl, lvl_max) - lvl_min, 0);

    if(left && (index_of_lvl[ilvl] > idx))
        atomicMin(&index_of_lvl[ilvl], idx);
    else if(!left && (index_of_lvl[ilvl] < idx+1))
        atomicMax(&index_of_lvl[ilvl], idx+1);
}

template<bool left>
__global__ void KernelPostProcessLevels(
    const int* irange,
    int32_t* index_of_lvl,
    const int lvl_min,
    const int lvl_max
) {
    int ilvl = blockDim.x * blockIdx.x + threadIdx.x;
    int nlevels = lvl_max - lvl_min + 1;

    if(ilvl > lvl_max - lvl_min)
        return;

    int idx = index_of_lvl[ilvl];
    
    for(int i=nlevels-1; i>=ilvl; i--) {
        if(left)
            idx = min(idx, index_of_lvl[i]);
        else
            idx = max(idx, index_of_lvl[i]);
    }

    index_of_lvl[ilvl] = idx;
}

template<bool left>
std::string GetBoundaryExtendPerLevel(
    cudaStream_t stream, 
    const float3* pos_ref,
    const int* irange,
    const float3* posz,
    int32_t* index_of_lvl,
    const int size,
    const size_t block_size
) {
    int lvl_min = -450;
    int lvl_max = 388;

    // Initialize indices 0, 1, 2, ..., size-1
    int nlevels = lvl_max - lvl_min + 1;

    KernelInitLevels<left><<< div_ceil(nlevels, block_size), block_size, 0, stream>>>(
        irange, index_of_lvl, lvl_min, lvl_max
    );

    KernelGetBoundaryExtendPerLevel<left><<< div_ceil(size, block_size), block_size, 0, stream>>>(
        pos_ref, irange, posz, index_of_lvl, lvl_min, lvl_max
    );

    KernelPostProcessLevels<left><<< div_ceil(nlevels, block_size), block_size, 0, stream>>>(
        irange, index_of_lvl, lvl_min, lvl_max
    );

    return std::string();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Node Properties                                        */
/* ---------------------------------------------------------------------------------------------- */

__device__ int clip(int a, int imin, int imax) {
    return min(max(a, imin), imax);
}

__global__ void GetNodeGeometry(
    const float3* pos,
    const int* lbound,
    const int* rbound,
    const int *nnodes,
    int32_t* level,
    float3* center,
    float3* extent,
    const int size_nodes,
    const int size_part
) {
    // Gets the properties of the smallest node that contains pos[lbound[idx]] and pos[rbound[idx]-1]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size_nodes)
        return;
    if(idx >= nnodes[0]) {
        level[idx] = -1000;
        center[idx] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
        extent[idx] = make_float3(0, 0, 0);
        return;
    }
    
    float3 x0 = pos[clip(lbound[idx], 0, size_part-1)];
    float3 x1 = pos[clip(rbound[idx]-1, 0, size_part-1)];

    int lvl = msb_diff_level(x0, x1);
    NodeWithExt node_ext = get_common_node(x0, x1);

    level[idx] = msb_diff_level(x0, x1);
    center[idx] = node_ext.center;
    extent[idx] = node_ext.extent;
}

#endif // TREE_H