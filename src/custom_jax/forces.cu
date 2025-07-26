#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Each custom FFI handler has four parts:
// (1) the CUDA kernel
// (2) the host function that launches the kernel
// (3) the FFI handler registration that registers the handler with the XLA runtime
// And in Python:
// (4) A Python function that calls the FFI handler

__global__ void PotentialKernel(const float *x, float *phi, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    phi[i] =  x[3*i]*x[3*i] + x[3*i+1]*x[3*i+1] + x[3*i+2]*x[3*i+2];
  }
}

ffi::Error PotentialHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> phi, size_t block_size) {
  size_t n = x.element_count() / x.dimensions().back();

  const size_t grid_size = (n + (block_size - 1)) / block_size;
  
  PotentialKernel<<<grid_size, block_size, /*shared_mem=*/0, stream>>>(x.typed_data(), phi->typed_data(), n);

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
