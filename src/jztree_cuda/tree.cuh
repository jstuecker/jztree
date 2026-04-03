#ifndef TREE_H
#define TREE_H

#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                      DetectLeafBoundaries                                      */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
__global__ void FlagLeafBoundaries(
    const Vec<dim,tvec>* posz,
    const uint8_t* ptype,
    const int* lvl_bound,
    const int* npart,
    int8_t* split_flags,
    int* lvl,
    int max_size,
    int size_part,
    int scan_size,
    int num_types
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
    uint8_t* types = reinterpret_cast<uint8_t*>(level + nload);

    int ioff = blockIdx.x * blockDim.x - scan_size - 1;
    
    __syncthreads();
    for(int i = threadIdx.x; i < nload-1; i += blockDim.x) {
        int ipart = ioff + i;
        if(ipart < 0) {
            level[i] = lbound_l;
            types[i] = 0;
        }
        else if(ipart + 1 >= nump) {
            level[i] = lbound_r;
            types[i] = 0;
        }
        else {
            level[i] = msb_diff_level<dim,tvec>(posz[ipart], posz[ipart + 1]);
            if(num_types > 1)
                types[i] = ptype[ipart];
            else
                types[i] = 0;
        }
    }
    __syncthreads();

    int idx = threadIdx.x + scan_size;
    int mylevel = level[idx];
    bool is_split = false;

    for(int type=0; type<num_types; type++) {
        // Find the boundaries of each node
        int lsize = 0;
        for(int i = idx - scan_size; i < idx; i++) {
            lsize = (level[i] < mylevel ? lsize : 0) + 1*(types[i] == type);
        }
        int rsize = 0;
        for(int i = idx + scan_size; i > idx; i--) {
            rsize = (level[i] < mylevel ? rsize : 0) + 1*(types[i] == type);
        }

        // Each maximum size node that is <= max_size is bounded by nodes that are > max_size
        // Therefore, we can find their splitting points by simply flagging all nodes that are > max_size
        is_split = is_split | (lsize + rsize > max_size);
        
        // If nodes hit a domain boundary and are larger than the boundary level, we always flag them
        // int max_lvl_left = lvl_bound[0], max_lvl_right = lvl_bound[1];
        is_split = is_split | ((mylevel > lvl_bound[0]) && (node_idx - lsize <= 0));
        is_split = is_split | ((mylevel > lvl_bound[1]) && (node_idx + rsize >= nump));
    }


    // Additionally we set the beginning and end points to be splits
    is_split &= node_idx <= nump;
    is_split |= node_idx == 0; 
    is_split |= node_idx == nump;

    if (node_idx < size_part + 1) {
        split_flags[node_idx] = is_split;
        lvl[node_idx] = mylevel;
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Tree Building                                         */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
__global__ void FindNodeBoundaries(
    const Vec<dim,tvec>* pos_in,
    const Vec<dim,tvec>* pos_boundary,
    const int *nleaves,
    int32_t* nodes_levels,
    int32_t* nodes_lbound,
    int32_t* nodes_rbound,
    const int size_nodes,
    const int lvl_max,
    const int lvl_invalid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nleaves[0];
    int nnodes = n + 1;

    if (idx >= size_nodes)
        return;
    if (idx >= nnodes) {
        // Set values for undefined nodes
        nodes_levels[idx] = lvl_invalid;
        nodes_lbound[idx] = n;
        nodes_rbound[idx] = n;

        return;
    }

    int target_level, lvl_left, lvl_right;
    int lbound, rbound;
    Vec<dim,tvec> p1, p2;

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
    target_level = msb_diff_level<dim,tvec>(p1, p2);

    if(msb_diff_level<dim,tvec>(pos_in[0], p2) <= target_level) {
        // Node goes until left-domain boundary... We have no left parent
        lbound = 0;
        lvl_left = lvl_max; // larger than any possible level
    } else {
        // ibefore is the an index (left) outside of the node, iinside is an index inside
        // We do a binary search until they lie next to each other:
        int ibefore = 0, iinside = idx;
        lvl_left = lvl_max;
        while (ibefore+1 < iinside) {
            int itest = (ibefore + iinside) / 2;
            lvl_left = msb_diff_level<dim,tvec>(pos_in[itest], p2);
            if (lvl_left > target_level) {
                ibefore = itest;
            } else {
                iinside = itest;
            }
        }
        lvl_left = msb_diff_level<dim,tvec>(p1, pos_in[ibefore]);
        lbound = iinside;
    }


    // Now find the right side parent
    if(msb_diff_level<dim,tvec>(p1, pos_in[n-1]) <= target_level)
    {
        rbound = n;
        lvl_right = lvl_max;
    }
    else
    {
        int iinside = idx-1, iafter = n-1;
        lvl_right = lvl_max;
        while (iinside+1 < iafter) {
            int itest = (iinside + iafter) / 2;
            lvl_right = msb_diff_level<dim,tvec>(p1, pos_in[itest]);
            if (lvl_right > target_level) {
                iafter = itest;
            } else {
                iinside = itest;
            }
        }

        rbound = iafter;
        lvl_right = msb_diff_level<dim,tvec>(p1, pos_in[rbound]);
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

template<bool left, int dim, typename tvec>
__global__ void KernelGetBoundaryExtendPerLevel(
    const Vec<dim,tvec>* pos_ref,
    const int* irange,
    const Vec<dim,tvec>* posz,
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

    Vec<dim,tvec> x0 = pos_ref[0];
    Vec<dim,tvec> x1 = posz[idx];

    int lvl = msb_diff_level<dim,tvec>(x0, x1);

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

template<bool left, int dim, typename tvec>
std::string GetBoundaryExtendPerLevel(
    cudaStream_t stream, 
    const Vec<dim,tvec>* pos_ref,
    const int* irange,
    const Vec<dim,tvec>* posz,
    int32_t* index_of_lvl,
    const int size,
    const size_t block_size,
    const int lvl_min,
    const int lvl_max
) {
    // Initialize indices 0, 1, 2, ..., size-1
    int nlevels = lvl_max - lvl_min + 1;

    KernelInitLevels<left><<< div_ceil(nlevels, block_size), block_size, 0, stream>>>(
        irange, index_of_lvl, lvl_min, lvl_max
    );

    KernelGetBoundaryExtendPerLevel<left,dim,tvec><<< div_ceil(size, block_size), block_size, 0, stream>>>(
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

template<int dim, typename tvec>
__global__ void GetNodeGeometry(
    const Vec<dim,tvec>* pos,
    const int* lbound,
    const int* rbound,
    const int *nnodes,
    int32_t* level,
    Vec<dim,tvec>* center,
    Vec<dim,tvec>* extent,
    const int size_nodes,
    const int size_part,
    const int lvl_invalid,
    const uint32_t mode_flags,
    const bool upper_extent
) {
    // Gets the properties of the smallest node that contains pos[lbound[idx]] and pos[rbound[idx]-1]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int lb=lbound[idx], rb=rbound[idx];

    if(idx >= size_nodes)
        return;
    if(idx >= nnodes[0]) {
        if(mode_flags & 1u) level[idx] = lvl_invalid;
        if(mode_flags & 2u) center[idx] = Vec<dim,tvec>::constant(invalid_val<tvec>()); // !!! Fix later
        if(mode_flags & 4u) extent[idx] = Vec<dim,tvec>::constant(0);
        return;
    }

    Vec<dim,tvec> x0b = pos[clip(lb, 0, size_part-1)];
    Vec<dim,tvec> x1a = pos[clip(rb-1, 0, size_part-1)];

    int lvl;
    if(!upper_extent) {
        // in lower extent mode we use the minimal level that guarantees to include
        // all particles that we know about
        lvl = msb_diff_level<dim,tvec>(x0b, x1a);
    }
    else {
        // in upper extent mode we use our (lowest) parent's level - 1
        // this guarantees that our node includes all the space that exists in our parents
        // basically also including hypothetical particles
        Vec<dim,tvec> x0a = pos[clip(lb-1, 0, size_part-1)];
        if(lb-1 < 0) x0a = Vec<dim,tvec>::constant(-INFINITY);

        Vec<dim,tvec> x1b = pos[clip(rb, 0, size_part-1)];
        if(rb >= size_part) x1b = Vec<dim,tvec>::constant(INFINITY);

        int lvl_left = msb_diff_level<dim,tvec>(x0a, x0b);
        int lvl_right = msb_diff_level<dim,tvec>(x1a, x1b);

        lvl = min(lvl_left, lvl_right) - 1;
    }
    
    if(mode_flags & 1u) 
        level[idx] = lvl;
    if(mode_flags & 2u)
        center[idx] = LvlToCenter<dim,tvec>(x0b, lvl);
    if(mode_flags & 4u)
        extent[idx] = LvlToExt<dim,tvec>(lvl);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Center of Mass                                         */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
__global__ void CenterOfMass(
    const int* __restrict__ isplit,
    const Vec<dim,tvec>* __restrict__ pos,
    const tvec* __restrict__ mass,
    PosMass<dim,tvec>* __restrict__ com_out,
    int nnodes,
    bool kahan
) {
    int inode = blockIdx.x * blockDim.x + threadIdx.x;

    if (inode >= nnodes)
        return;

    int istart = isplit[inode], iend = isplit[inode + 1];

    if(istart >= iend) {
        com_out[inode] = {Vec<dim,tvec>::constant(invalid_val<tvec>()), invalid_val<tvec>()};
        return;
    }
    
    Vec<dim+1,tvec> mp_sum = Vec<dim+1,tvec>::constant(static_cast<tvec>(0));
    Vec<dim+1,tvec> mp_kahan = Vec<dim+1,tvec>::constant(static_cast<tvec>(0));

    for(int ip = istart; ip < iend; ip++) {
        tvec m = mass[ip];
        Vec<dim+1,tvec> mp_new;
        #pragma unroll
        for(int i=0; i<dim; i++)
            mp_new[i] = m * pos[ip][i];
        mp_new[dim] = m;

        if(kahan)
            kahan_add_vec<dim+1,tvec>(mp_sum, mp_new, mp_kahan);
        else
            mp_sum = mp_sum + mp_new;
    }

    if (mp_sum[dim] > static_cast<tvec>(0)) {
        PosMass<dim,tvec> out;
        out.mass = mp_sum[dim];
        tvec invmass = static_cast<tvec>(1) / out.mass;
        #pragma unroll
        for(int i=0; i<dim; i++) {
            out.pos[i] = invmass * mp_sum[i];
        }
        com_out[inode] = out;
    }
    else {
        com_out[inode] = PosMass<dim,tvec>{Vec<dim,tvec>::constant(invalid_val<tvec>()), invalid_val<tvec>()};
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   Interaction List Reduction                                   */
/* ---------------------------------------------------------------------------------------------- */

__global__ void FlagInteractingNodes(
    const int* __restrict__ isplit,
    const int* __restrict__ isrc,
    bool* flag,
    const int size_nodes,
    const int size_ilist
) {
    // Flags all nodes that appear as source or receiving index in the interaction list

    int inode = blockIdx.x;
    
    int ilow = isplit[inode], iup = min(isplit[inode + 1], size_ilist);
    if(ilow >= iup)
        return;

    if(threadIdx.x == 0)
        flag[inode] = true;

    for(int i=ilow + threadIdx.x; i<iup; i += blockDim.x) {
        int idx = isrc[i];
        if((idx >= 0) && (idx < size_nodes))
            flag[idx] = true;
    }
}

#endif // TREE_H