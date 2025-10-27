#ifndef CUSTOM_JAX_FMM_H
#define CUSTOM_JAX_FMM_H

/* ---------------------------------------------------------------------------------------------- */
/*                                   Evaluate Tree Plane Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */

#define BLOCKSIZE 32

struct NodeInfo {
    float3 center;
    int level;
};

struct NodeWithExt {
    float3 center;
    float3 extent;
};

__device__ __forceinline__ float3 float3sum(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 float3diff(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float norm2(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ __forceinline__ float3 LvlToExt(int level) {
    // Converts a node's or leaf's binary level to its extend per dimension

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 3000) / 3 - 1000;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2);
    int lz = olvl + (omod >= 1);
    
    return make_float3(ldexpf(1.0f, lx), ldexpf(1.0f, ly), ldexpf(1.0f, lz));
}

__device__ __forceinline__ bool OpeningCriterion(
    NodeWithExt nodeA,
    NodeWithExt nodeB,
    float opening_angle
) {
    float r2 = norm2(float3diff(nodeA.center, nodeB.center));
    float L2 = norm2(float3sum(nodeA.extent, nodeB.extent));

    bool need_open = L2 > opening_angle * opening_angle * r2;
    // also open if L2 had an overflow (and r2 is valid)
    need_open = need_open || ((isnan(L2) || isinf(L2)) && !isnan(r2));
    return need_open;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(__activemask(), v, offset);
    return v;
}

// template<int p>
__global__ void CountInteractions(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const NodeInfo* children,
    // const float* mp_values,
    // outputs:
    // float* loc_out,
    int* child_count_out,
    // int* child_ilist_out,
    // attributes:
    float softening,
    float opening_angle
) {
    // Node A info:
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    // Child A info:
    int2 child_range = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};
    int num_childrenA = min(child_range.y - child_range.x, blockDim.x); // handle larger nodes later

    __shared__ NodeWithExt childA[BLOCKSIZE];
    if(child_range.x + threadIdx.x < child_range.y) {
        NodeInfo child = children[child_range.x + threadIdx.x];
        childA[threadIdx.x] = {child.center, LvlToExt(child.level)};
    }
    else
        childA[threadIdx.x] = {NAN, NAN, NAN, 1e10, 1e10, 1e10};

    __shared__ int num_open[BLOCKSIZE];
    if(threadIdx.x < BLOCKSIZE) {
        num_open[threadIdx.x] = 0;
    }
    
    // Interaction list info:
    int2 ilist_range = {spl_ilist[nodeid], spl_ilist[nodeid + 1]};
    __syncthreads();

    __shared__ int2 segments[BLOCKSIZE];
    SegmentManager<BLOCKSIZE> seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        ilist_range.x,
        ilist_range.y
    );

    while(!seg_mgr.finished()) {
        int2 id = seg_mgr.next();

        NodeWithExt childB_ext;
        if(id.x >= 0) {
            NodeInfo childB = children[id.x];
            childB_ext = {childB.center, LvlToExt(childB.level)};
        }
        else {
            childB_ext = {NAN, NAN, NAN, 1e10, 1e10, 1e10};
        }

        for(int i = 0; i < num_childrenA; i++) {
            bool need_open = OpeningCriterion(childA[i], childB_ext, opening_angle);

            int num = warp_reduce_sum((need_open && id.x >= 0) ? 1 : 0);

            if((threadIdx.x & 0x1f) == 0)
                atomicAdd(&num_open[i], num);
        }
    }
    __syncthreads();
    if(child_range.x + threadIdx.x < child_range.y) {
        child_count_out[child_range.x + threadIdx.x] = num_open[threadIdx.x];
    }
}

#endif