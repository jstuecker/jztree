#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Each custom FFI handler has four parts:
// (1) the CUDA kernel
// (2) the host function that launches the kernel
// (3) the FFI handler registration that registers the handler with the XLA runtime
// And in Python:
// (4) A Python function that calls the FFI handler

__global__ void PotentialKernel(const float4 *xm, float *phi, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t blocksize = blockDim.x;

  const size_t steps = n / blocksize;

  float4 xmi = xm[blockIdx.x * blocksize + threadIdx.x];

  extern __shared__ float4 xmj_shared[];

  float phii = 0.f;

  for (size_t jblock = 0; jblock < steps; jblock += 1) {
    // Load the next block of x into shared memory
    // this avoids reading from global memory multiple times
    __syncthreads();
    xmj_shared[threadIdx.x] = xm[jblock * blocksize + threadIdx.x];
    __syncthreads();

    for (size_t j = 0; j < blocksize; j++) {
      float4 xmj = xmj_shared[j];
      
      float dx = xmi.x - xmj.x;
      float dy = xmi.y - xmj.y;
      float dz = xmi.z - xmj.z;
      float m = xmj.w; // we packed mass into the w component of xm

      float r = sqrtf(dx*dx + dy*dy + dz*dz);
      float rinv = (r >= 1e-15f) ? 1.0f/r : 0.0f; // avoid division by zero

      phii += -m*rinv;
    }
  }

  phi[blockIdx.x * blocksize + threadIdx.x] = phii;
}

ffi::Error PotentialHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> phi, size_t block_size) {
  size_t n = x.element_count() / x.dimensions().back();

  const size_t grid_size = (n + (block_size - 1)) / block_size;
  
  auto* x_float4 = reinterpret_cast<const float4*>(x.typed_data()); // interprete xm as an array of float4. This makes the kernel easier to write.
  PotentialKernel<<<grid_size, block_size, /*shared_mem=*/block_size*sizeof(float4), stream>>>(x_float4, phi->typed_data(), n);

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Potential, PotentialHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()              // x
        .Ret<ffi::Buffer<ffi::F32>>()              // phi
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});
