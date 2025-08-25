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

template<int k>
struct SortedNearestK {
    float r2s[k];
    int ids[k];

    __device__ inline void init(float r2=1e10, int index=-1) {
        #pragma unroll
        for (int i = 0; i < k; i++) {
            r2s[i] = r2;
            ids[i] = index;
        }
    }

    // __device__ inline void set_without_update(const int i, const float r2, int index) {
    //     r2s[i] = r2;
    //     ids[i] = index;
    // }

    __device__ inline float max_r2() const {
        return r2s[k-1];
    }


    // Insert (r2,id) if it belongs; keeps array ascending and evicts the largest.
    __device__ __forceinline__ void consider(float r2, int id){
        if (r2 > r2s[k-1]) return; // fast reject, ok

        float carry_r2 = r2; int carry_id = id;

        #pragma unroll
        for (int i = k-2; i >= 0; --i) {
            float ai = r2s[i]; int bi = ids[i];
            bool shift = (carry_r2 < ai);          // need to shift ai right?

            // Write position i+1
            r2s[i+1] = shift ? ai  : carry_r2;
            ids[i+1] = shift ? bi  : carry_id;

            // Update carry (if we placed carry at i+1, we now carry ai toward earlier slots)
            carry_r2 = shift ? carry_r2 : ai;
            carry_id = shift ? carry_id : bi;
        }
        // Finally place remaining carry at slot 0
        r2s[0] = carry_r2;
        ids[0] = carry_id;
    }

};

template<int K>
struct SortedNearestKMasked {
    float r2s[K];
    int   ids[K];

    __device__ __forceinline__ void init(float r2=1e10f, int idx=-1){
        #pragma unroll
        for (int i=0;i<K;++i){ 
            r2s[i]=r2;
            ids[i]=idx;
        }
    }

    __device__ __forceinline__ float max_r2() const {
        return r2s[K-1]; 
    }

    // Insert (r2,id) into ascending r2s[], evicting the largest.
    // Early-exit effect via an "active" mask; no dynamic indexing, no returns in the loop.
    __device__ __forceinline__ void consider(float r2, int id) {
        if (r2 > r2s[K-1]) return;

        float carry_r2 = r2; 
        int   carry_id = id;
        bool  active   = true;  // true until we've placed the item

        #pragma unroll
        for (int i = K-2; i >= 0; --i) {
            // only meaningful while active
            bool take = active && (carry_r2 < r2s[i]);

            // write i+1 only while active
            float out_r2 = take ? (float)r2s[i] : carry_r2;
            int   out_id = take ? (int)  ids[i] : carry_id;

            // predicated stores keep it register-only
            r2s[i+1] = active ? out_r2 : r2s[i+1];
            ids[i+1] = active ? out_id : ids[i+1];

            // if we didn't take (and were active), we've placed; turn off further work
            active = active && take;

            // update carry only while still active (otherwise irrelevant)
            carry_r2 = active ? carry_r2 : carry_r2; // no-op when inactive
            carry_id = active ? carry_id : carry_id; // (kept for symmetry)
        }

        // If we never placed, item belongs at slot 0
        r2s[0] = active ? carry_r2 : r2s[0];
        ids[0] = active ? carry_id : ids[0];
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

struct PosR {
    float3 pos;
    float r;
};

__device__ __forceinline__ float3 LvlToHalfExt(int level) {
    int olvl = level / 3;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2 ? 1 : 0);
    int lz = olvl + (omod >= 1 ? 1 : 0);
    
    return make_float3(ldexpf(1.0f, lx-1), ldexpf(1.0f, ly-1), ldexpf(1.0f, lz-1));
}

__device__ __forceinline__ float mindist(float x1, float x2, float width_half) {
    return max(fabsf(x1-x2) - width_half, 0.0f);
}

__device__ __forceinline__ float NodePartMinDist2(const Node& node, const float3& part) {
    float3 half_ext = LvlToHalfExt(node.level);

    float dx = mindist(part.x, node.center.x, half_ext.x);
    float dy = mindist(part.y, node.center.y, half_ext.y);
    float dz = mindist(part.z, node.center.z, half_ext.z);

    return dx*dx + dy*dy + dz*dz;
}

__device__ __forceinline__ int get_next_leaf(
    const Node* __restrict__ leaves,
    const int* __restrict__ ilist,
    int &ileaf_out,
    float3 xpart, 
    float r2max,
    int &icur,
    int iend
) {
    while(icur < iend) {
        ileaf_out = ilist[icur];
        Node leaf = leaves[ileaf_out];
        
        float r2leaf = NodePartMinDist2(leaf, xpart);
        
        bool accept = (r2leaf <= r2max);

        bool any_accept = syncthreads_or(accept);

        icur += 1;
        if(any_accept) {
            return true;
        }
    }
    return false;
}

template <int k>
__global__ void KernelIlistKNN(
    const PosR* xT,             // input positions
    const PosR* xQ,             // query positions
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

    for(int ileafQ = istart; ileafQ < iend; ileafQ += 1) {
        int iqstart = isplitQ[ileafQ], iqend = isplitQ[ileafQ + 1];
        int ipartQ = min(iqstart + threadIdx.x, iqend - 1);
        bool validQ = iqstart + threadIdx.x < iqend;

        float3 posQ = xQ[ipartQ].pos;

        SortedNearestKMasked<k> nearestK;
        nearestK.init();

        __shared__ Particle particles[32];

        int iqmin = ilist_splitsQ[ileafQ], iqmax = ilist_splitsQ[ileafQ + 1];
        int ileafT = 0;

        while(get_next_leaf(leaves, ilist, ileafT, posQ, nearestK.max_r2(), iqmin, iqmax)) {
            int ipartT = isplitT[ileafT] + threadIdx.x;
            int npartT = isplitT[ileafT + 1] - isplitT[ileafT];

            if (threadIdx.x < npartT) {
                particles[threadIdx.x] = {xT[ipartT].pos, ipartT};
            }
            __syncthreads();

            // Now search for the nearest neighbors in A
            for (int j = 0; j < npartT; j++) {
                Particle p = particles[j];
                float r2 = distance_squared(p.pos, posQ);
                nearestK.consider(r2, p.id);
            }
        }

        // nearestK.final_sort();
        
        if(validQ) {
            for(int i = 0; i < k; i++) {
                knn[ipartQ * k + i] = {sqrtf(nearestK.r2s[i]), nearestK.ids[i]};
            }
        }
    }
}

void launch_KernelIlistKNN(int k, cudaStream_t stream, 
    int nleavesQ, int interactions_per_block,
    const PosR* xT, const PosR* xQ,
    const int* isplitT, const int* isplitQ, const Node* leaves,
    const int* ilist, const int* ilist_splitsQ,
    Neighbor* knn, float boxsize) {

    switch(k) {
        case 4: KernelIlistKNN<4><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 8: KernelIlistKNN<8><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 12: KernelIlistKNN<12><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 16: KernelIlistKNN<16><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 32: KernelIlistKNN<32><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
        case 64: KernelIlistKNN<64><<< div_ceil(nleavesQ, interactions_per_block), 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, interactions_per_block, boxsize, nleavesQ); break;
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

    PosR* xAf4 = reinterpret_cast<PosR*>(xT.typed_data());
    PosR* xBf4 = reinterpret_cast<PosR*>(xQ.typed_data());
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