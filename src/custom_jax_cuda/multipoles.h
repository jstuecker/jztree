#ifndef CUSTOM_JAX_MULTIPOLES_H
#define CUSTOM_JAX_MULTIPOLES_H

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Simple POD for particle position+mass used by the multipoles kernels
struct PosMass {
    float x, y, z, mass;
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

#endif // CUSTOM_JAX_MULTIPOLES_H
