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

ffi::Error HostPosZorderSort(cudaStream_t stream, ffi::Buffer<ffi::F32> pos_in, ffi::ResultBuffer<ffi::S32> id_out, ffi::ResultBuffer<ffi::S32> tmp_buffer, size_t block_size) {
    size_t n = pos_in.element_count()/3;

    float3* keys_in = reinterpret_cast<float3*>(pos_in.typed_data());
    PosId* keyids = reinterpret_cast<PosId*>(id_out->typed_data());
    
    // Initialize indices 0, 1, 2, ..., n-1
    int blocks = (n + block_size - 1) / block_size;
    PosKeyArangeKernel<<<blocks, block_size, 0, stream>>>(keys_in, keyids, n);

    // We have an annoying problem here:
    // CUB requires a temporary storage buffer and it will usually tell us dynamically what the
    // size of it is (On first call with zero pointer).
    // Unfortunately, we may not allocate storage dynamically in an FFI call
    // Therefore, we have to estimate the storage requirements in advance in python/jax and pass
    // a sufficiently large buffer to the function (via "tmp_buffer").
    // Empirically I have found that the storage tends to be a bit larger than n * sizeof(PosId)
    // that is why we will pre-allocate something that is a few percent larger than that.
    // However, below we throw an error if our assumption ever turns out wrong.

    // find out the required storage size
    size_t required_storage_bytes;
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(nullptr, required_storage_bytes, keyids, n, PosIdLess());
    
    // Check if the provided buffer is large enough
    if (tmp_buffer->size_bytes() < required_storage_bytes) {
        return ffi::Error::Internal(std::string(
            "The buffer in ZorderSort is too small. Please contact me if this check fails.") +
            std::string(" Have: ") + std::to_string(tmp_buffer->size_bytes()) +
            std::string(". Required: ") + std::to_string(required_storage_bytes) +
            std::string(". Diff: ") + std::to_string((long long)required_storage_bytes - (long long)tmp_buffer->size_bytes())
        );
    }
    
    // This is how we ould allocate if we could. (Note that doing this breaks jit in some cases!)
    // cudaMallocAsync(&d_temp_storage, required_storage_bytes, stream); 

    // Run the sort
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(tmp_buffer->untyped_data(), required_storage_bytes, keyids, n, PosIdLess(), stream);
    
    // cudaFreeAsync(d_temp_storage, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

__device__ __forceinline__ int32_t msb_diff_level(const float3 &p1, const float3 &p2) {
    int msb_x = float_xor_msb(p1.x, p2.x);
    int msb_y = float_xor_msb(p1.y, p2.y);
    int msb_z = float_xor_msb(p1.z, p2.z);

    // The level is given by the most significant differing bit
    // but offset according to the dimension
    return max(3*msb_x+3, max(3*msb_y+2, 3*msb_z+1));
}


struct NodePointers {
    int32_t* levels;
    int32_t* lbound;
    int32_t* rbound;
    int32_t* lchild;
    int32_t* rchild;
};


__global__ void KernelBinarySearchLeftParent(const float3* pos_in, NodePointers nodes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int node = idx + 1; // Node indices are offset by 1, because we put a fake node at the beginning
    int Nnodes = n + 1; 

    int target_level, lvl_right, lvl_left;
    int lbound, rbound;
    float3 p1, p2;
    bool valid_thread = (idx < n-1);

    if (valid_thread) {
        // Calculate the level difference of our considered set of two points (=node)
        p1 = pos_in[idx];
        p2 = pos_in[idx + 1];

        target_level = msb_diff_level(p1, p2);

        // We do a binary search, trying to find the closest point to the left
        // that has a level difference of at least `level`
        int imin = -1, imax = idx+1;
        lvl_left = 388;
        while (imin+1 < imax) {
            int itest = (imin + imax) / 2;
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

        if(imin >= 0)
            lvl_left = msb_diff_level(p1, pos_in[imin]);
        else
            lvl_left = 388;
    }
    
    __syncthreads(); // Synchronize to reduce thread divergence

    if (valid_thread) {
        // Now find the right side parent
        int imin = idx, imax = n;
        lvl_right = 388;
        while (imin+1 < imax) {
            int itest = (imin + imax) / 2;
            lvl_right = msb_diff_level(p1, pos_in[itest]);
            if (lvl_right > target_level) {
                imax = itest;
            } else {
                imin = itest;
            }
        }

        rbound = imin+1;

        if(rbound < Nnodes)
            lvl_right = msb_diff_level(p1, pos_in[rbound]);
        else
            lvl_right = 388;
    }

    __syncthreads();

    if (valid_thread) {
        nodes.levels[node] = target_level;
        nodes.lbound[node] = lbound;
        nodes.rbound[node] = rbound;
        
        // The parent of each node is the lower one of the two boundary nodes
        if(lvl_left <= lvl_right) {
            nodes.rchild[lbound] = node;
        } else {
            nodes.lchild[rbound] = node;
        }
    }
}
__global__ void KernelInitialize(NodePointers nodes, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nnodes = n + 1;

    if (idx >= Nnodes)
        return;

    // We have fake nodes at the beginning and end of the array
    // to simplify walking the tree
    if((idx == 0) || (idx == Nnodes - 1)) {
        nodes.lbound[idx] = 0;
        nodes.rbound[idx] = Nnodes-1;
        nodes.levels[idx] = 388;
        nodes.lchild[idx] = idx; // Point to itself, may be overwritten later
        nodes.rchild[idx] = idx; // Point to itself, may be overwritten later
    }
    else {
        // indices <= 0 correspond to leafs (=particles)
        // by defaults node's children point to particles,
        // but about half of them will be overwritten by nodes later
        nodes.lchild[idx] = -idx + 1;
        nodes.rchild[idx] = -idx;
    }
}

ffi::Error HostBuildZTree(cudaStream_t stream, ffi::Buffer<ffi::F32> pos_in, ffi::ResultBuffer<ffi::S32> outputs, size_t block_size) {
    size_t n = pos_in.element_count()/3;
    size_t Nnodes = n + 1;

    float3* keys_in = reinterpret_cast<float3*>(pos_in.typed_data());

    // Output will be (5, Nnodes) array with different types of information in the first axis
    // Create some easier readable pointers that start at offset locations in the output
    int *out_ptr = outputs->typed_data();
    NodePointers nodes;
    nodes.levels = out_ptr;
    nodes.lbound = out_ptr + Nnodes;
    nodes.rbound = out_ptr + 2 * Nnodes;
    nodes.lchild = out_ptr + 3 * Nnodes;
    nodes.rchild = out_ptr + 4 * Nnodes;
    
    KernelInitialize<<<(n+1 + block_size-1) / block_size, block_size, 0, stream>>>(nodes, n);

    KernelBinarySearchLeftParent<<<(n-1 + block_size-1) / block_size, block_size, 0, stream>>>(keys_in, nodes, n);

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
        .Ret<ffi::Buffer<ffi::S32>>()        // temporary buffer
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