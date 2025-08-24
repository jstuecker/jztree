#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "shared_utils.cuh"


namespace nb = nanobind;
namespace ffi = xla::ffi;

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

struct Particle {
    float3 pos;
    int id;
};

struct Node {
    float3 center;
    int level;
};

template <int k>
__global__ void KernelIlistKNN(
    const float4* xT,           // input positions
    const float4* xQ,           // query positions
    const int* isplitT,         // leaf-ranges in A
    const int* isplitQ,         // leaf-ranges in B
    const Node* leaves,         // binary levels of A
    const int* ilist,           // interaction list
    const int* ilist_splitsQ,   // B leaf-ranges in ilist
    Neighbor* knn,              // output knn list
    int interactions_per_block, // 1 for now
    float boxsize,              // ignored for now
    int nleavesQ                // number of leaves in B
) {
    int istart = blockIdx.x * interactions_per_block;
    int iend = min(istart + interactions_per_block, nleavesQ);

    __shared__ int2 _seg_spaceQ[8];
    SegmentManager<8, true> segmentsQ(nullptr, isplitQ, _seg_spaceQ, istart, iend);

    while(!segmentsQ.finished()) {
        int2 ipartQ = segmentsQ.next();
        int ileafQ = ipartQ.y;

        float4 posQf4 = ipartQ.x >= 0 ? xQ[ipartQ.x] : make_float4(0.f,0.f,0.f,0.f);
        float3 posQ = {posQf4.x, posQf4.y, posQf4.z};

        NearestK<k> nearestK;
        nearestK.init();

        __shared__ int2 _seg_spaceT[16];
        SegmentManager<16, false> segmentsT(ilist, isplitT, _seg_spaceT, ilist_splitsQ[ileafQ], ilist_splitsQ[ileafQ + 1]);

        __shared__ Particle particles[32];

        while(!segmentsT.finished()) {
            __syncthreads();
            int ipartT = segmentsT.next().x;

            if (ipartT >= 0) {
                float4 xload = xT[ipartT];
                particles[threadIdx.x] = {make_float3(xload.x, xload.y, xload.z), ipartT};
            }
            __syncthreads();

            if(ipartQ.x < 0)
                continue; // skip if no valid query point

            // Now search for the nearest neighbors in A
            for (int j = 0; j < segmentsT.nids_loaded(); j++) {
                Particle p = particles[j];
                float r2 = distance_squared(p.pos, posQ);
                nearestK.consider(r2, p.id);
            }
        }

        nearestK.final_sort();
        
        if(ipartQ.x >= 0) {
            for(int i = 0; i < k; i++) {
                knn[ipartQ.x * k + i] = {sqrtf(nearestK.r2s[i]), nearestK.ids[i]};
            }
        }
    }
}

void launch_KernelIlistKNN(int k, cudaStream_t stream, 
    int nleavesQ, int interactions_per_block,
    const float4* xT, const float4* xQ,
    const int* isplitT, const int* isplitQ, const Node* leaves,
    const int* ilist, const int* ilist_splitsQ,
    Neighbor* knn, float boxsize) {

    switch(k) {
        case 4: KernelIlistKNN<4><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 8: KernelIlistKNN<8><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 16: KernelIlistKNN<16><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 32: KernelIlistKNN<32><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
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
        ffi::Buffer<ffi::F32> leaves,
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
    Node* leaves_ptr = reinterpret_cast<Node*>(leaves.typed_data());

    size_t block_size = 32;

    launch_KernelIlistKNN(k, stream, nleavesQ, interactions_per_block,
        xAf4, xBf4,
        isplitT.typed_data(), isplitQ.typed_data(),
        leaves_ptr, ilist.typed_data(), ilist_splitsQ.typed_data(),
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
        .Arg<ffi::Buffer<ffi::F32>>() // leaves : pos and levels of leaves
        .Arg<ffi::Buffer<ffi::S32>>() // ilist : interaction list
        .Arg<ffi::Buffer<ffi::S32>>() // ilist_splitsQ : leaf-ranges in ilist
        .Ret<ffi::Buffer<ffi::S32>>() // knn : output knn list
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_knn, m) {
    m.def("IlistKNNSearch", []() { return EncapsulateFfiCall(IlistKNNSearch); });
}