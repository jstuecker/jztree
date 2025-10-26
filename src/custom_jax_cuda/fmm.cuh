#ifndef CUSTOM_JAX_FMM_H
#define CUSTOM_JAX_FMM_H

#include "common.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                   Evaluate Tree Plane Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */

template<int p>
__global__ void EvaluateTreePlane(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const float3* xchild,
    const float* mp_values,
    // outputs:
    float* loc_out,
    int* spl_child_ilist_out,
    int* child_ilist_out,
    // attributes:
    float epsilon
) {
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 prange = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    int ipart = prange.x + threadIdx.x;
    bool valid = ipart < prange.y;

    if(valid) {
        loc_out[ipart] = threadIdx.x; // placeholder write
    }
}

#endif