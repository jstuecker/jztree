#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "shared_utils.cuh"
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

__device__ __forceinline__ bool z_pos_less(float3 pos1, float3 pos2)
{
    int msb_x = float_xor_msb(pos1.x, pos2.x);
    int msb_y = float_xor_msb(pos1.y, pos2.y);
    int msb_z = float_xor_msb(pos1.z, pos2.z);

    int ms_dim = (msb_x >= msb_y && msb_x >= msb_z) ? 0 : ((msb_y >= msb_z) ? 1 : 2);

    if (ms_dim == 0) return pos1.x < pos2.x;
    if (ms_dim == 1) return pos1.y < pos2.y;
    return pos1.z < pos2.z;
}

struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId &a, const PosId &b) {
        return z_pos_less(a.pos, b.pos);
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
    PosKeyArangeKernel<<< div_ceil(n, block_size), block_size, 0, stream>>>(keys_in, keyids, n);

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
    bool valid_thread = (idx < n-1);
    int Nnodes = n + 1; 

    // Node indices are offset by 1, because we put a fake node at the beginning
    // but we don't launch the kernel for it
    int node = idx + 1; 

    int target_level, lvl_left, lvl_right;
    int lbound, rbound;
    float3 p1, p2;

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

        if(rbound < n)
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
    
    KernelInitialize<<< div_ceil(Nnodes, block_size), block_size, 0, stream>>>(nodes, n);

    KernelBinarySearchLeftParent<<< div_ceil(n-1, block_size), block_size, 0, stream>>>(keys_in, nodes, n);

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


__device__ float distance_squared(const float3 &a, const float3 &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

struct Neighbor {
    float r2;
    int id;
};

template<int k>
struct NearestK {
    float r2s[k];
    int ids[k];

    int max_idx;
    float  max_r2;

    __device__ inline void init(float r2=1e10, int index=-1) {
        #pragma unroll
        for (int i = 0; i < k; i++) {
            r2s[i] = r2;
            ids[i] = index;
        }
        max_idx = 0;
        max_r2 = r2;
    }

    __device__ inline void set_without_update(const int i, const float r2, int index) {
        r2s[i] = r2;
        ids[i] = index;
    }

    __device__ inline void rebuild_max() {
        max_idx = 0; 
        max_r2 = r2s[0];
        #pragma unroll
        for (int i = 1; i < k; i++) {
            if (r2s[i] > max_r2) { 
                max_r2 = r2s[i]; 
                max_idx = i; 
            }
        }
    }

    __device__ inline void consider(float r2, int id) {
        // If this particle is closer than the furthest one we have, replace it
        if (r2 <= max_r2) {
            r2s[max_idx] = r2;
            ids[max_idx] = id;
        }
        else {
            // Check later whether this return improves things or makes them slower
            // It is a thread divergence vs. unnecessary work trade-off
            return; 
        }
        
        rebuild_max(); 
    }
    
    __device__ inline void final_sort() {
        // simple bubble sort (works well in registers)
        // optimize later!
        #pragma unroll
        for (int i = 0; i < k; i++) {
            for (int j = i; j < k; j++) {
                if (r2s[i] > r2s[j]) {
                    float tmp_r2 = r2s[i];
                    int tmp_id = ids[i];
                    r2s[i] = r2s[j];
                    ids[i] = ids[j];
                    r2s[j] = tmp_r2;
                    ids[j] = tmp_id;
                }
            }
        }
    }
};


template <int k>
__global__ void KernelIlistKNN(
    const float4* xT,           // input positions
    const float4* xQ,           // query positions
    const int* isplitT,         // leaf-ranges in A
    const int* isplitQ,         // leaf-ranges in B
    const int* lvlT,            // binary levels of A
    const int* ilist,           // interaction list
    const int* ilist_splitsQ,   // B leaf-ranges in ilist
    Neighbor* knn,              // output knn list
    int interactions_per_block, // 1 for now
    float boxsize,              // ignored for now
    int nleavesQ                // number of leaves in B
) {
    // for now we assume interactions_per_block == 1
    int ileafQ = blockIdx.x;
    if (ileafQ >= nleavesQ) {
        return; // No work to do
    }
    
    int ileafQ_start = isplitQ[ileafQ], ileafQ_end = isplitQ[ileafQ + 1];
    int npartQ = ileafQ_end - ileafQ_start;

    if (npartQ <= 0) {
        return; // No particles in this leaf
    }

    int ipart = ileafQ_start + min(threadIdx.x, npartQ - 1);

    float4 posQf4 = xQ[ipart];
    float3 posQ = make_float3(posQf4.x, posQf4.y, posQf4.z);
    // float rmin = posQf4.w; // Will use this later, but not for now

    NearestK<k> nearestK;
    nearestK.init();

    int ilist_start = ilist_splitsQ[ileafQ], ilist_end = ilist_splitsQ[ileafQ + 1];
    int ninteractions = ilist_end - ilist_start;

    __shared__ float3 xT_shared[32];

    for(int i = 0; i < ninteractions; i++) {
        int ileafT = ilist[ilist_start + i];

        // Load the positions of particles in leaf A into shared memory
        int ileafT_start = isplitT[ileafT], ileafT_end = isplitT[ileafT + 1];
        int npartT = ileafT_end - ileafT_start;

        __syncthreads();
        if (threadIdx.x < npartT) {
            float4 xload = xT[ileafT_start + threadIdx.x];
            xT_shared[threadIdx.x] = make_float3(xload.x, xload.y, xload.z);
        }
        __syncthreads();

        // Now search for the nearest neighbors in A
        for (int j = 0; j < npartT; j++) {
            float3 posT = xT_shared[j];
            float r2 = distance_squared(posT, posQ);
            nearestK.consider(r2, ileafT_start + j);
        }
    }

    nearestK.final_sort();
    
    if(threadIdx.x < npartQ) {
        for(int i = 0; i < k; ++i) {
            knn[ipart * k + i] = {sqrtf(nearestK.r2s[i]), nearestK.ids[i]};
        }
    }
}

void launch_KernelIlistKNN(int k, cudaStream_t stream, 
    int nleavesQ, int interactions_per_block,
    const float4* xT, const float4* xQ,
    const int* isplitT, const int* isplitQ, const int* lvlT,
    const int* ilist, const int* ilist_splitsQ,
    Neighbor* knn, float boxsize) {

    switch(k) {
        case 4: KernelIlistKNN<4><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, lvlT, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 8: KernelIlistKNN<8><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, lvlT, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 16: KernelIlistKNN<16><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, lvlT, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 32: KernelIlistKNN<32><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, lvlT, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        default:
            throw std::runtime_error("Unsupported k value in launch_KernelIlistKNN");
    }
}

ffi::Error HostIlistKNNSearch(
        cudaStream_t stream, 
        ffi::Buffer<ffi::F32> xT, 
        ffi::Buffer<ffi::F32> xQ, 
        ffi::Buffer<ffi::S32> isplitT,
        ffi::Buffer<ffi::S32> isplitQ,
        ffi::Buffer<ffi::S32> lvlT,
        ffi::Buffer<ffi::S32> ilist,
        ffi::Buffer<ffi::S32> ilist_splitsQ,
        ffi::ResultBuffer<ffi::S32> knn,
        size_t interactions_per_block,
        float boxsize
    ) {
    int k = knn->dimensions()[knn->dimensions().size() - 2];
    size_t nleavesQ = isplitQ.element_count() - 1;

    float4* xAf4 = reinterpret_cast<float4*>(xT.typed_data());
    float4* xBf4 = reinterpret_cast<float4*>(xQ.typed_data());
    Neighbor* knn_ptr = reinterpret_cast<Neighbor*>(knn->typed_data());

    size_t block_size = 32;
    
    // KernelIlistKNN<32><<< div_ceil(nleavesQ, interactions_per_block), block_size, 0, stream>>>(
    //     xAf4, xBf4,
    //     isplitT.typed_data(), isplitQ.typed_data(),
    //     lvlT.typed_data(), ilist.typed_data(), ilist_splitsQ.typed_data(),
    //     knn_ptr, interactions_per_block, 
    //     boxsize, nleavesQ
    // );

    launch_KernelIlistKNN(k, stream, nleavesQ, interactions_per_block,
        xAf4, xBf4,
        isplitT.typed_data(), isplitQ.typed_data(),
        lvlT.typed_data(), ilist.typed_data(), ilist_splitsQ.typed_data(),
        knn_ptr, boxsize);
    
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistKNNSearch, HostIlistKNNSearch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // xT : input positions
        .Arg<ffi::Buffer<ffi::F32>>() // xQ : eval positions
        .Arg<ffi::Buffer<ffi::S32>>() // isplitT : leaf-ranges in A
        .Arg<ffi::Buffer<ffi::S32>>() // isplitQ : leaf-ranges in B
        .Arg<ffi::Buffer<ffi::S32>>() // lvlT : levels of A, used to determine the extend
        .Arg<ffi::Buffer<ffi::S32>>() // ilist : interaction list
        .Arg<ffi::Buffer<ffi::S32>>() // ilist_splitsQ : leaf-ranges in ilist
        .Ret<ffi::Buffer<ffi::S32>>() // knn : output knn list
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible});

// Include deprecated functions
// This module includes a bunch of functions that we do not need anymore, but we keep
// them temporarily for comparison and test purposes
// #include "tree_deprecated.cu"


NB_MODULE(nb_tree, m) {
    m.def("PosZorderSort", []() { return EncapsulateFfiCall(PosZorderSort); });
    m.def("BuildZTree", []() { return EncapsulateFfiCall(BuildZTree); });
    m.def("IlistKNNSearch", []() { return EncapsulateFfiCall(IlistKNNSearch); });

    // A bunch of deprecated functions
    // m.def("OldArgsort", []() { return EncapsulateFfiCall(OldArgsort); });
    // m.def("OldI3zsort", []() { return EncapsulateFfiCall(OldI3zsort); });
    // m.def("OldF3zsort", []() { return EncapsulateFfiCall(OldF3zsort); });
    // m.def("OldI3Argsort", []() { return EncapsulateFfiCall(OldI3Argsort); });
    // m.def("OldI3zMergesort", []() { return EncapsulateFfiCall(OldI3zMergesort); });
}