#ifndef CUSTOM_JAX_FMM_H
#define CUSTOM_JAX_FMM_H

#include "common.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                   Evaluate Tree Plane Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */

struct EvaluateTreePlaneInputs {
    const int2* node_range;
    const int* spl_nodes;
    const int* spl_ilist;
    const int* ilist_nodes;
    const float3* xchild;
    const float* mp_values;
};

struct EvaluateTreePlaneOutputs {
    float* loc_out;
    int* spl_child_ilist_out;
    int* child_ilist_out;
};

struct EvaluateTreePlaneAttrs {
    float epsilon;
};

template<int p>
__global__ void EvaluateTreePlaneKernel(
    const EvaluateTreePlaneInputs inputs,
    const EvaluateTreePlaneOutputs outputs,
    const EvaluateTreePlaneAttrs attrs
) {
    int2 nrange = inputs.node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 prange = {inputs.spl_nodes[nodeid], inputs.spl_nodes[nodeid + 1]};

    int ipart = prange.x + threadIdx.x;
    bool valid = ipart < prange.y;

    if(valid) {
        outputs.loc_out[ipart] = threadIdx.x; // placeholder write
    }
}

void launch_EvaluateTreePlaneKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const EvaluateTreePlaneInputs inputs,
    const EvaluateTreePlaneOutputs outputs,
    const EvaluateTreePlaneAttrs attrs) {
    LAUNCH_KERNEL_SWITCH(p, EvaluateTreePlaneKernel, grid_size, block_size, stream, 
        inputs, outputs, attrs);
}

template<int p>
__global__ void TestPositions(
    const int* indices,
    float3* positions,
    int num,
    float boxsize
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
        int idx = indices[i];
        positions[i] = make_float3(idx * boxsize, idx * boxsize, idx * boxsize);
    }
}

__global__ void SimpleArange(
    int* output,
    int size
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if( i < size )
            output[i] = i;
    }
}

#endif