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


__global__ void KernelMsbDiffLevel(const float3* pos_in, int *level_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n-1) {
        float3 p1 = pos_in[idx];
        float3 p2 = pos_in[idx + 1];

        int msb_x = float_xor_msb(p1.x, p2.x);
        int msb_y = float_xor_msb(p1.y, p2.y);
        int msb_z = float_xor_msb(p1.z, p2.z);

        // The level is given by the most significant differing bit
        // but offset according to the dimension
        // This is so that L = 2**(level//3) would correspond to the size 
        // of the octree-node that contains both points
        // The minimal possible difference is -(127+23)*3 = -450
        // the maximal possible difference is (128+1)*3 = 387
        level_out[idx] = max(3*msb_x+3, max(3*msb_y+2, 3*msb_z+1));
    }
}

__device__ __forceinline__ int32_t msb_diff_level(const float3 &p1, const float3 &p2) {
    int msb_x = float_xor_msb(p1.x, p2.x);
    int msb_y = float_xor_msb(p1.y, p2.y);
    int msb_z = float_xor_msb(p1.z, p2.z);

    // The level is given by the most significant differing bit
    // but offset according to the dimension
    return max(3*msb_x+3, max(3*msb_y+2, 3*msb_z+1));
}

__global__ void KernelBinarySearchLeftParent(const float3* pos_in, int *tree_info, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int target_level, lvl_right, lvl_left;
    size_t lbound, rbound;
    float3 p1, p2;
    bool valid_thread = (idx < n-1);

    if (valid_thread) {
        // Calculate the level difference of our considered set of two points (=node)
        p1 = pos_in[idx];
        p2 = pos_in[idx + 1];

        target_level = msb_diff_level(p1, p2);

        // We do a binary search, trying to find the closest point to the left
        // that has a level difference of at least `level`
        size_t imin = -1, imax = idx+1;
        lvl_left = 387;
        while (imin+1 < imax) {
            size_t itest = (imin + imax) / 2;
            lvl_left = msb_diff_level(pos_in[itest], p2);
            if (lvl_left > target_level) {
                imin = itest;
            } else {
                imax = itest;
            }
        }

        // Our array has two fake nodes at the beginning and end
        // that's why we have to offset the indices by 1
        lbound = imin+1;

        lvl_left = msb_diff_level(p1, pos_in[lbound-1]);
    }
    
    __syncthreads(); // Synchronize to reduce thread divergence

    if (valid_thread) {
        // Now find the right side parent
        size_t imin = idx, imax = n;
        lvl_right = 387;
        while (imin+1 < imax) {
            size_t itest = (imin + imax) / 2;
            lvl_right = msb_diff_level(p1, pos_in[itest]);
            if (lvl_right > target_level) {
                imax = itest;
            } else {
                imin = itest;
            }
        }

        rbound = imin+1;

        lvl_right = msb_diff_level(p1, pos_in[rbound]);
    }

    __syncthreads();

    if (valid_thread) {
        // Our array has two fake nodes at the beginning and end
        // that's why we have to offset the indices by 1
        tree_info[idx+1] = target_level;
        tree_info[idx+1 + (n+1)] = lbound;
        tree_info[idx+1 + 2*(n+1)] = rbound;

        
        // The parent of each node is the lower one of the two boundary nodes
        if(lvl_left <= lvl_right) {
            tree_info[lbound + 4*(n+1)] = idx+1; // we are the parent's right child
        } else {
            tree_info[rbound + 3*(n+1)] = idx+1; // we are the parent's left child
        }
    }
}

__global__ void KernelInitialize(int *tree_info, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n+1)
        return;

    tree_info[idx + 3*(n+1)] = -idx + 1;
    tree_info[idx + 4*(n+1)] = -idx;


}

// __global__ void KernelAssignChildren(int *tree_info, size_t n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= n-1)
//         return;
    
//     int lvl = tree_info[idx + 1];
//     int lbound = tree_info[idx + 1 + (n+1)];
//     int rbound = tree_info[idx + 1 + 2*(n+1)];

//     int lvl_lbound = tree_info[lbound];
//     int lvl_rbound = tree_info[rbound];

//     if(lvl_lbound <= lvl_rbound) {
//         tree_info[lbound + 4*(n+1)] = idx+1; // we are the parent's right child
//     } else {
//         tree_info[rbound + 3*(n+1)] = idx+1; // we are the parent's left child
//     }
// }

ffi::Error HostBuildZTree(cudaStream_t stream, ffi::Buffer<ffi::F32> pos_in, ffi::ResultBuffer<ffi::S32> level_out, size_t block_size) {
    size_t n = pos_in.element_count()/3;

    float3* keys_in = reinterpret_cast<float3*>(pos_in.typed_data());
    
    // int blocks = (n + block_size - 1) / block_size;
    // KernelMsbDiffLevel<<<blocks, block_size, 0, stream>>>(keys_in, level_out->typed_data(), n);

    KernelInitialize<<<(n+1 + block_size-1) / block_size, block_size, 0, stream>>>(level_out->typed_data(), n);

    KernelBinarySearchLeftParent<<<(n-1 + block_size-1) / block_size, block_size, 0, stream>>>(keys_in, level_out->typed_data(), n);

    // KernelAssignChildren<<<(n-1 + block_size-1) / block_size, block_size, 0, stream>>>(level_out->typed_data(), n);


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
// #include "tree_deprecated.cu"


NB_MODULE(nb_tree, m) {
    m.def("PosZorderSort", []() { return EncapsulateFfiCall(PosZorderSort); });
    m.def("BuildZTree", []() { return EncapsulateFfiCall(BuildZTree); });

    // A bunch of deprecated functions
    // m.def("OldArgsort", []() { return EncapsulateFfiCall(OldArgsort); });
    // m.def("OldI3zsort", []() { return EncapsulateFfiCall(OldI3zsort); });
    // m.def("OldF3zsort", []() { return EncapsulateFfiCall(OldF3zsort); });
    // m.def("OldI3Argsort", []() { return EncapsulateFfiCall(OldI3Argsort); });
    // m.def("OldI3zMergesort", []() { return EncapsulateFfiCall(OldI3zMergesort); });
}