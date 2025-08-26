#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "shared_utils.cuh"


namespace nb = nanobind;
namespace ffi = xla::ffi;

struct Neighbor {
    float r2;
    int id;
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


__device__ __forceinline__ float wrap(float dx, const float boxsize) {
    // wraps a coordinate difference into the interval [-boxsize/2, boxsize/2)
    // Note: in principle this code would be slightly more optimal if we decided at compile time
    // whether we are periodic. However, my tests suggest that this would only be 3% or so.
    if(boxsize > 0.f) {
        float bh = 0.5f * boxsize;
        dx = dx < -bh ? dx + boxsize : dx;
        dx = dx >= bh ? dx - boxsize : dx;
    }

    return dx;
}

__device__ __forceinline__ float distance_squared(const float3 &a, const float3 &b, const float boxsize) {
    float dx = wrap(a.x - b.x, boxsize);
    float dy = wrap(a.y - b.y, boxsize);
    float dz = wrap(a.z - b.z, boxsize);

    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float3 LvlToHalfExt(int level) {
    // Converts a node's or leaf's binary level to its half of its extend per dimension
    int olvl = level / 3;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2 ? 1 : 0);
    int lz = olvl + (omod >= 1 ? 1 : 0);
    
    return make_float3(ldexpf(1.0f, lx-1), ldexpf(1.0f, ly-1), ldexpf(1.0f, lz-1));
}

__device__ __forceinline__ float mindist(float x1, float x2, float width_half, float boxsize=0.f) {
    // Minimum distance per dimension between a point and a box
    return max(fabsf(wrap(x1-x2, boxsize)) - width_half, 0.0f);
}

__device__ __forceinline__ float NodePartMinDist2(const Node& node, const float3& part, float boxsize=0.f) {
    // Minimum squared distance between a node and a particle
    float3 half_ext = LvlToHalfExt(node.level);

    float dx = mindist(part.x, node.center.x, half_ext.x, boxsize);
    float dy = mindist(part.y, node.center.y, half_ext.y, boxsize);
    float dz = mindist(part.z, node.center.z, half_ext.z, boxsize);

    return dx*dx + dy*dy + dz*dz;
}

template<int K>
struct SortedNearestK {
    // Keeps track of the nearest K neighbors that we have seen so far in ascending order of r2.
    // Offers efficient insertion of new candidates

    float r2s[K];
    int   ids[K];

    __device__ __forceinline__ SortedNearestK(float r2=1e10f, int idx=-1){
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

    float3 posQ = xQ[ipartQ].pos;

    SortedNearestK<k> nearestK(1e38, -1);

    __shared__ Particle particles[32];

    PrefetchList<int> pf_ilist(ilist, ilist_splitsQ[ileafQ], ilist_splitsQ[ileafQ + 1]);

    while(!pf_ilist.finished()) {
        int ileafT = pf_ilist.next();

        /* First check whether any particle needs to interact with the leaf*/
        Node leaf = leaves[ileafT];
        float r2leaf = NodePartMinDist2(leaf, posQ, boxsize);
        bool accept = (r2leaf <= nearestK.max_r2());
        bool any_accept = syncthreads_or(accept);
        if(!any_accept) 
            continue;

        /* Now load the leaf */
        int ipartT = isplitT[ileafT] + threadIdx.x;
        int npartT = isplitT[ileafT + 1] - isplitT[ileafT];

        if (threadIdx.x < npartT)
            particles[threadIdx.x] = {xT[ipartT].pos, ipartT};
        __syncthreads();

        // Now search for the nearest neighbors in A
        for (int j = 0; j < npartT; j++) {
            Particle p = particles[j];
            float r2 = distance_squared(p.pos, posQ, boxsize);
            nearestK.consider(r2, p.id);
        }
    }
    
    if(iqstart + threadIdx.x < iqend) {
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

/* A histogram that can live in registers. As usual with complex data structures in registers,
   we have to do some extra work when updating, so that we can work with static indices only.
*/
template<int BINS>
struct CumHist {
    int c[BINS];
    __device__ CumHist() {
        #pragma unroll
        for (int i=0; i<BINS; i++) 
            c[i] = 0;
    }
    __device__ inline void add(int b) {
        // Updates all counters with predication; stays in registers when unrolled.
        #pragma unroll
        for (int i=0; i<BINS; i++)
            c[i] += (int)(b >= i);
    }
};


// template <int RBINS>
__global__ void KernelCountInteractions(
    const Node* leaves,
    const int* leaves_npart,
    const int* isplit,
    const int* node_ilist,
    const int* node_ilist_splits,
    int* interaction_count,
    float* rmax,
    int k,
    float boxsize
) {
    int nodeQ = blockIdx.x;
    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    int ileafQ = min(ileafQ_start + threadIdx.x, ileafQ_end - 1);

    constexpr int RBINS = 16;

    CumHist<RBINS> rhist;

    PrefetchList<int> pf_ilist(node_ilist, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

    __shared__ int counts[RBINS];

    while(!pf_ilist.finished()) {
        int nodeT = pf_ilist.next();
        int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

        rhist.add(1);
    }

    // interaction_count[ileafQ] = rhist.c[2];
    interaction_count[threadIdx.x] = threadIdx.x;
}

ffi::Error HostConstructIlist(
        cudaStream_t stream,
        ffi::Buffer<ffi::F32> leaves,
        ffi::Buffer<ffi::S32> leaves_npart,
        ffi::Buffer<ffi::S32> isplit,
        ffi::Buffer<ffi::S32> node_ilist,
        ffi::Buffer<ffi::S32> node_ilist_splits,
        ffi::ResultBuffer<ffi::S32> temp_buffer,
        ffi::ResultBuffer<ffi::S32> leaf_ilist,
        ffi::ResultBuffer<ffi::S32> leaf_ilist_splits,
        int k,
        float boxsize
    )
{
    Node* leaves_ptr = reinterpret_cast<Node*>(leaves.typed_data());
    int nnodes = isplit.element_count() - 1;
    int nleaves = leaves_npart.element_count();

    int* counts_ptr = temp_buffer->typed_data();
    float* rmax_ptr = reinterpret_cast<float*>(&counts_ptr[nleaves]);

    KernelCountInteractions<<< nnodes, 32, 0, stream>>>(
        leaves_ptr,
        leaves_npart.typed_data(),
        isplit.typed_data(),
        node_ilist.typed_data(),
        node_ilist_splits.typed_data(),
        counts_ptr,
        rmax_ptr,
        k,
        boxsize
    );
    
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ConstructIlist, HostConstructIlist,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() 
        .Arg<ffi::Buffer<ffi::F32>>() 
        .Arg<ffi::Buffer<ffi::S32>>() 
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<int>("k")
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible});


NB_MODULE(nb_knn, m) {
    m.def("IlistKNNSearch", []() { return EncapsulateFfiCall(IlistKNNSearch); });
    m.def("ConstructIlist", []() { return EncapsulateFfiCall(ConstructIlist); });
}