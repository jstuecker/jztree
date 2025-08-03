#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

namespace nb = nanobind;
namespace ffi = xla::ffi;

// A wrapper to encapsulate an FFI call
template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}


__device__ __forceinline__ int32_t float_xor_msb(float a, float b) {
    // Finds the most significant bit that differs between x and y
    // For floating point numbers we need to treat the exponent and the mantissa differently:
    // If the exponent differs, then the (power of two) of the difference is given by the larger
    // exponent.
    // If the exponent is the same, then we need to compare the mantissas. The (power of two) of the
    // difference is then given by the differing bit in the mantissa, offset by the exponent

    if (signbit(a) != signbit(b)) {
        return 128;  // The sign is the highest significant bit
    }
    int32_t a_bits = __float_as_int(fabsf(a));
    int32_t b_bits = __float_as_int(fabsf(b));

    int32_t a_exp = (a_bits >> 23) - 127;
    int32_t b_exp = (b_bits >> 23) - 127;

    if (a_exp == b_exp) { // If both floats have the same exponent, we need to compare mantissas
        // clz counts bit-zeros from the left. There will be always 8 leading zeros due to the
        // exponent
        return a_exp + (8 - __clz(a_bits ^ b_bits)); 
    }
    else { // If exponents differ, return the larger exponent
        return max(a_exp, b_exp);
    }
}

struct PosId {
    float3 pos;
    int32_t id;
};

struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId &a, const PosId &b) {
        int msb_x = float_xor_msb(a.pos.x, b.pos.x);
        int msb_y = float_xor_msb(a.pos.y, b.pos.y);
        int msb_z = float_xor_msb(a.pos.z, b.pos.z);

        // Find dimension with least clz → most significant differing bit
        int ms_dim = (msb_x >= msb_y && msb_x >= msb_z) ? 0 : ((msb_y >= msb_z) ? 1 : 2);

        // Perform the comparison on the most significant dimension
        if (ms_dim == 0) return a.pos.x < b.pos.x;
        if (ms_dim == 1) return a.pos.y < b.pos.y;
        return a.pos.z < b.pos.z;
    }
};

__global__ void PosKeyArangeKernel(const float3* pos_in, PosId *keyid_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keyid_out[idx].pos = pos_in[idx];
        keyid_out[idx].id = idx;
    }
}

ffi::Error HostPosZorderSort(cudaStream_t stream, ffi::Buffer<ffi::F32> pos_in, ffi::ResultBuffer<ffi::S32> id_out, size_t block_size) {
    size_t n = pos_in.element_count()/3;

    float3* keys_in = reinterpret_cast<float3*>(pos_in.typed_data());
    PosId* keyids = reinterpret_cast<PosId*>(id_out->typed_data());
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    PosKeyArangeKernel<<<blocks, block_size, 0, stream>>>(keys_in, keyids, n);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(nullptr, temp_storage_bytes, keyids, n, PosIdLess(), stream);

    if (temp_storage_bytes > 0) {
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(d_temp_storage, temp_storage_bytes, keyids, n, PosIdLess(), stream);
        cudaFree(d_temp_storage);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


__global__ void GetMsbDiffLevel(const float3* pos_in, int *level_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n-1) {
        float3 p1 = pos_in[idx];
        float3 p2 = pos_in[idx + 1];

        int msb_x = float_xor_msb(p1.x, p2.x);
        int msb_y = float_xor_msb(p1.y, p2.y);
        int msb_z = float_xor_msb(p1.z, p2.z);

        // The most significant differing bit
        // int msb_diff = max(3*msb_x+2, max(3*msb_y+1, 3*msb_z));

        level_out[idx] = msb_x;
        level_out[idx+(n-1)] = msb_y;
        level_out[idx+2*(n-1)] = msb_z;
        level_out[idx+3*(n-1)] = max(3*msb_x+3, max(3*msb_y+2, 3*msb_z+1));
    }
}

ffi::Error HostBuildZTree(cudaStream_t stream, ffi::Buffer<ffi::F32> pos_in, ffi::ResultBuffer<ffi::S32> level_out, size_t block_size) {
    size_t n = pos_in.element_count()/3;

    float3* keys_in = reinterpret_cast<float3*>(pos_in.typed_data());
    
    int blocks = (n + block_size - 1) / block_size;
    GetMsbDiffLevel<<<blocks, block_size, 0, stream>>>(keys_in, level_out->typed_data(), n);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PosZorderSort, HostPosZorderSort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()        // pos
        .Ret<ffi::Buffer<ffi::S32>>()        // output ids / pos
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BuildZTree, HostBuildZTree,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()        // pos
        .Ret<ffi::Buffer<ffi::S32>>()        // output level
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});


// Include deprecated functions
// This module includes a bunch of functions that we do not need anymore, but we keep
// them temporarily for comparison and test purposes
#include "tree_deprecated.cu"


NB_MODULE(nb_tree, m) {
    m.def("PosZorderSort", []() { return EncapsulateFfiCall(PosZorderSort); });
    m.def("BuildZTree", []() { return EncapsulateFfiCall(BuildZTree); });

    // A bunc of deprecated functions
    m.def("OldArgsort", []() { return EncapsulateFfiCall(OldArgsort); });
    m.def("OldI3zsort", []() { return EncapsulateFfiCall(OldI3zsort); });
    m.def("OldF3zsort", []() { return EncapsulateFfiCall(OldF3zsort); });
    m.def("OldI3Argsort", []() { return EncapsulateFfiCall(OldI3Argsort); });
    m.def("OldI3zMergesort", []() { return EncapsulateFfiCall(OldI3zMergesort); });
}