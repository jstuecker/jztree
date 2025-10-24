#ifndef CUSTOM_JAX_MULTIPOLES_H
#define CUSTOM_JAX_MULTIPOLES_H

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Simple POD for particle position+mass used by the multipoles kernels
struct PosMass {
    float x, y, z, mass;
};

// Bundle all device pointers and scalars into a single view to simplify signatures.
struct EvaluateTreePlaneView {
    const int2* node_range;
    const int* spl_nodes;
    const int* spl_ilist;
    const int* ilist_nodes;
    const float3* xchild;
    const float* mp_values;
    float* loc_out;
    int* spl_child_ilist_out;
    int* child_ilist_out;
    float epsilon;
};

// Launchers compiled in multipoles.cu (defined there). These are used by the FFI
// wrappers in ffi_multipoles.cu. Keep signatures stable.
void launch_IlistM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
        const float3 *x, const float *mp, int2 *interactions, int *iminmax, float *Lout,
        size_t interactions_per_block, float epsilon);

void launch_IlistLeaf2NodeM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const float3 *xnodes, const float4 *xm, int32_t *isplit, int2 *interactions, int *iminmax,
    float *Lout, size_t interactions_per_block, float epsilon);

void launch_MultipolesFromParticlesKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const int *isplit, const PosMass *part_posm, float *mp_out, float3 *xcom_out);

// Launcher for coarsening multipoles: reduces child multipoles (mp_center, mp_values)
// into coarser-level outputs (out_mp, out_xcent). Implemented in multipoles.cu.
void launch_CoarsenMultipolesKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const int *isplit, const float *mp_values, const float3 *mp_center, float *out_mp, float3 *out_xcent);

// Launcher for evaluate_tree_plane: evaluates M2L for a tree plane and generates child interaction list
void launch_EvaluateTreePlaneKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream,
    const EvaluateTreePlaneView view);

#endif // CUSTOM_JAX_MULTIPOLES_H