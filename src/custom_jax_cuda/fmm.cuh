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
    float epsilon
) {
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 child_range = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    __shared__ NodeInfo childA[BLOCKSIZE];
    
    if(threadIdx.x < (child_range.y - child_range.x))
        childA[threadIdx.x] = children[child_range.x + threadIdx.x];
    else
        childA[threadIdx.x] = {0.f, 0.f, 0.f, 0};
    
    // int ipart = prange.x + threadIdx.x;
    // bool valid = ipart < prange.y;

    // if(valid) {
    //     loc_out[ipart] = threadIdx.x; // placeholder write
    // }
    if(threadIdx.x < (child_range.y - child_range.x)) {
        child_count_out[child_range.x + threadIdx.x] = child_range.x + threadIdx.x;
    }
}

#endif