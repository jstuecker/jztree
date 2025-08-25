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

template<int K>
struct SortedNearestK {
    // Keeps track of the nearest K neighbors that we have seen so far in ascending order of r2.
    // Offers efficient insertion of new candidates

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
    // I have tried many different variants of this function and this turned out to be the fastest.
    // Important aspects:
    // * no dynamic indexing
    // * no breaks/returns in the loop (stops the compiler from unrolling)
    // * It seems the pattern with the mask helps the compiler to skip unnecessary work
    //   -- effectively achieving an early exit (still not totally sure how it works though...)
    __device__ __forceinline__ void consider(float r2, int id) {
        if (r2 > r2s[K-1]) return;

        bool  active   = true;  // true until we've placed the item

        #pragma unroll
        for (int i = K-2; i >= 0; --i) {
            bool take = active && (r2 < r2s[i]);

            r2s[i+1] = active ? (take ? r2s[i] : r2) : r2s[i+1];
            ids[i+1] = active ? (take ? ids[i]  : id) : ids[i+1];

            active = take;
        }

        // If we never placed, item belongs at slot 0
        r2s[0] = active ? r2 : r2s[0];
        ids[0] = active ? id : ids[0];
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
    float boxsize               // ignored for now
) {
    int ileafQ = blockIdx.x;

    int iqstart = isplitQ[ileafQ], iqend = isplitQ[ileafQ + 1];
    int ipartQ = min(iqstart + threadIdx.x, iqend - 1);
    bool validQ = iqstart + threadIdx.x < iqend;

    float3 posQ = xQ[ipartQ].pos;

    SortedNearestK<k> nearestK;
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
    
    if(validQ) {
        #pragma unroll
        for(int i = 0; i < k; i++) {
            knn[ipartQ * k + i] = {sqrtf(nearestK.r2s[i]), nearestK.ids[i]};
        }
    }
}

void launch_KernelIlistKNN(
    int k, cudaStream_t stream, int nleavesQ, 
    const PosR* xT, const PosR* xQ,
    const int* isplitT, const int* isplitQ, const Node* leaves,
    const int* ilist, const int* ilist_splitsQ,
    Neighbor* knn, float boxsize) {

    switch(k) {
        case 4: KernelIlistKNN<4><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, boxsize); break;
        case 8: KernelIlistKNN<8><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, boxsize); break;
        case 12: KernelIlistKNN<12><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, boxsize); break;
        case 16: KernelIlistKNN<16><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, boxsize); break;
        case 32: KernelIlistKNN<32><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, boxsize); break;
        case 64: KernelIlistKNN<64><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ilist_splitsQ, knn, boxsize); break;
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
        float boxsize
    ) {
    int k = knn->dimensions()[knn->dimensions().size() - 2];
    size_t nleavesQ = isplitQ.element_count() - 1;

    PosR* xAf4 = reinterpret_cast<PosR*>(xT.typed_data());
    PosR* xBf4 = reinterpret_cast<PosR*>(xQ.typed_data());
    Neighbor* knn_ptr = reinterpret_cast<Neighbor*>(knn->typed_data());
    Node* leaves_ptr = reinterpret_cast<Node*>(leaves.typed_data());

    size_t block_size = 32;

    launch_KernelIlistKNN(k, stream, nleavesQ,
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
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_knn, m) {
    m.def("IlistKNNSearch", []() { return EncapsulateFfiCall(IlistKNNSearch); });
}