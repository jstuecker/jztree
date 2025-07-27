#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Each custom FFI handler has four parts:
// (1) the CUDA kernel
// (2) the host function that launches the kernel
// (3) the FFI handler registration that registers the handler with the XLA runtime
// And in Python:
// (4) A Python function that calls the FFI handler

__global__ void PotentialKernel(const float3 *x, float *phi, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t blocksize = blockDim.x;

  const size_t steps = n / blocksize;

  float3 xi = x[blockIdx.x * blocksize + threadIdx.x];

  extern __shared__ float3 xj[];

  float phii = 0.f;

  for (size_t jblock = 0; jblock < steps; jblock += 1) {
    // Load the next block of x into shared memory
    // this avoids reading from global memory multiple times
    __syncthreads();
    xj[threadIdx.x] = x[jblock * blocksize + threadIdx.x];
    __syncthreads();

    for (size_t j = 0; j < blocksize; j++) {
      float3 xj_val = xj[j];
      
      float dx = xi.x - xj_val.x;
      float dy = xi.y - xj_val.y;
      float dz = xi.z - xj_val.z;

      phii += sqrtf(dx*dx + dy*dy + dz*dz);
    }
  }

  phi[blockIdx.x * blocksize + threadIdx.x] = phii;
}

ffi::Error PotentialHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> phi, size_t block_size) {
  size_t n = x.element_count() / x.dimensions().back();

  const size_t grid_size = (n + (block_size - 1)) / block_size;
  
  auto* x_float3 = reinterpret_cast<const float3*>(x.typed_data()); // interprete x as an array of float3. This makes the kernel easier to write.
  PotentialKernel<<<grid_size, block_size, /*shared_mem=*/block_size*sizeof(float3), stream>>>(x_float3, phi->typed_data(), n);

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
