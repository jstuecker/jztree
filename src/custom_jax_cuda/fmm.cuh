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
    // Node A info:
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    // Child A info:
    int2 child_range = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};
    __shared__ NodeInfo childA[BLOCKSIZE];
    if(child_range.x + threadIdx.x < child_range.y)
        childA[threadIdx.x] = children[child_range.x + threadIdx.x];
    else
        childA[threadIdx.x] = {NAN, NAN, NAN, 10000};
    
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

    int iterartion = 0;
    while(!seg_mgr.finished()) {
        int2 id = seg_mgr.next();
        
        if((blockIdx.x == 0) && (id.y >= 0)) {
            child_count_out[child_range.x + threadIdx.x + iterartion*blockDim.x] = seg_mgr.nids_loaded();
        }
        iterartion += 1;
    }
}

#endif