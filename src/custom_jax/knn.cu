#include <type_traits>

#include <cmath>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "shared_utils.cuh"
#include <cub/cub.cuh>

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

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 3000) / 3 - 1000;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2);
    int lz = olvl + (omod >= 1);
    
    return make_float3(ldexpf(1.0f, lx-1), ldexpf(1.0f, ly-1), ldexpf(1.0f, lz-1));
}

__device__ __forceinline__ float mindist2(float3 x1, float3 x2, float3 width_half, float boxsize=0.f) {
    float dx =  max(fabsf(wrap(x1.x-x2.x, boxsize)) - width_half.x, 0.0f);
    float dy =  max(fabsf(wrap(x1.y-x2.y, boxsize)) - width_half.y, 0.0f);
    float dz =  max(fabsf(wrap(x1.z-x2.z, boxsize)) - width_half.z, 0.0f);

    return dx*dx + dy*dy + dz*dz;
}

__device__ __forceinline__ float maxdist2(float3 x1, float3 x2, float3 width_half, float boxsize=0.f) {
    float dx =  fabsf(wrap(x1.x-x2.x, boxsize)) + width_half.x;
    float dy =  fabsf(wrap(x1.y-x2.y, boxsize)) + width_half.y;
    float dz =  fabsf(wrap(x1.z-x2.z, boxsize)) + width_half.z;

    return dx*dx + dy*dy + dz*dz;
}

__device__ __forceinline__ float NodePartMinDist2(const Node& node, const float3& part, float boxsize=0.f) {
    // Minimum squared distance between a node and a particle
    float3 half_ext = LvlToHalfExt(node.level);

    return mindist2(part, node.center, half_ext, boxsize);
}

__device__ __forceinline__ float NodeNodeMinDist2(const Node& nodeA, const Node& nodeB, float boxsize=0.f) {
    // The distance between the closest points inside two nodes
    float3 hA = LvlToHalfExt(nodeA.level);
    float3 hB = LvlToHalfExt(nodeB.level);
    float3 half_ext = make_float3(hA.x + hB.x, hA.y + hB.y, hA.z + hB.z);

    return mindist2(nodeA.center, nodeB.center, half_ext, boxsize);
}

__device__ __forceinline__ float NodeNodeMaxDist2(const Node& nodeA, const Node& nodeB, float boxsize=0.f) {
    // The distance between the closest points inside two nodes
    float3 hA = LvlToHalfExt(nodeA.level);
    float3 hB = LvlToHalfExt(nodeB.level);
    float3 half_ext = make_float3(hA.x + hB.x, hA.y + hB.y, hA.z + hB.z);

    return maxdist2(nodeA.center, nodeB.center, half_ext, boxsize);
}

__device__ __forceinline__ float3 sumf3(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float dotf3(const float3 &a, const float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


template<int K>
struct SortedNearestK {
    // Keeps track of the nearest K neighbors that we have seen so far in ascending order of r2.
    // Offers efficient insertion of new candidates

    float r2s[K];
    int   ids[K];

    __device__ __forceinline__ SortedNearestK(float r2=INFTY, int idx=-1){
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
    const float* ir2list,       // (lower) interaction radii
    const int* ilist_splitsQ,   // B leaf-ranges in ilist
    Neighbor* knn,              // output knn list
    float boxsize               // ignored for now
) {
    int ileafQ = blockIdx.x;

    int iqstart = isplitQ[ileafQ], iqend = isplitQ[ileafQ + 1];
    int ipartQ = min(iqstart + threadIdx.x, iqend - 1);

    float3 posQ = xQ[ipartQ].pos;

    SortedNearestK<k> nearestK(INFTY, -1);

    __shared__ Particle particles[32];

    PrefetchList2<int,float> pf_ilist(ilist, ir2list, ilist_splitsQ[ileafQ], ilist_splitsQ[ileafQ + 1]);

    while(!pf_ilist.finished()) {
        Pair<int,float> interaction = pf_ilist.next();
        int ileafT = interaction.first;
        float r2T = interaction.second;

        // r2T are the lower leaf-leaf distances and the interactions are sorted by these radii
        // Once we encounter an interaction that is farther away than any of the current nearest 
        // neighbors, we can skip all subsequent interactions. Since the interaction lists are 
        // build on worst case assumptions, this saves a lot of time in practice!
        bool any_accept = syncthreads_or(r2T <= nearestK.max_r2());
        if(!any_accept) break;

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
    const int* ilist, const float* ir2list, const int* ilist_splitsQ,
    Neighbor* knn, float boxsize) {

    switch(k) {
        case 4: KernelIlistKNN<4><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ir2list, ilist_splitsQ, knn, boxsize); break;
        case 8: KernelIlistKNN<8><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ir2list, ilist_splitsQ, knn, boxsize); break;
        case 12: KernelIlistKNN<12><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ir2list, ilist_splitsQ, knn, boxsize); break;
        case 16: KernelIlistKNN<16><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ir2list, ilist_splitsQ, knn, boxsize); break;
        case 32: KernelIlistKNN<32><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ir2list, ilist_splitsQ, knn, boxsize); break;
        case 64: KernelIlistKNN<64><<< nleavesQ, 32, 0, stream>>>(xT, xQ, isplitT, isplitQ, leaves, ilist, ir2list, ilist_splitsQ, knn, boxsize); break;
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
        ffi::Buffer<ffi::F32> ir2list,
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
        leaves_ptr, ilist.typed_data(), ir2list.typed_data(), ilist_splitsQ.typed_data(),
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
        .Arg<ffi::Buffer<ffi::F32>>() // ir2list : (lower) interaction radii
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
    __device__ __forceinline__ CumHist() {
        #pragma unroll
        for (int i=0; i<BINS; i++) 
            c[i] = 0;
    }
    __device__ __forceinline__ void insert(int b, int num=1) {
        #pragma unroll
        for (int i=0; i<BINS; i++) {
            c[i] += num*(i >= b);
        }
    }

    __device__ __forceinline__  int find(int b) const {
        // Find the smallest bin that is >= b
        int ibin = BINS;
        #pragma unroll
        for(int i = BINS-1; i >= 0; i--) {
            ibin = (c[i] >= b) ? i : ibin;
        }
        return ibin;
    }

    __device__ __forceinline__  int get(int b) const {
        // get the value, using static indexing only
        int val = 0;
        #pragma unroll
        for(int i = BINS-1; i >= 0; i--) {
            val = (i == b) ? c[i] : val;
        }
        return val;
    }
};

template <int BINS>
struct LogBinMap {
    float logrbase2;
    float bins_per_log2;
    #define OFFSET 0.99f
    // We use the OFFSET to make the first bin almost exclusively contain the rbase2 case
    // This helps to not include unnecessary leaves when the immediate neighboord is sufficient

    __device__ __forceinline__ LogBinMap(float rbase2, float bins_per_log2) {
        this->logrbase2 = __log2f(rbase2);
        this->bins_per_log2 = bins_per_log2;
    }

    __device__ __forceinline__ int r2_to_bin(float r2)
    {
        float logr2 = __log2f(r2);
        return __float2int_rd((logr2 - logrbase2) * 0.5f * bins_per_log2 + OFFSET);
    }

    __device__ __forceinline__ float bin_end(int ibin)
    {
        if(ibin < BINS)
            return __powf(2.0f, logrbase2 + float(ibin+1.0f-OFFSET)*(2.0f/bins_per_log2));
        else
            return INFTY;
    }
};

__global__ void KernelCountInteractions(
    const Node* leaves,
    const int* leaves_npart,
    const int* isplit,
    const int* node_ilist,
    const float* node_ir2list,
    const int* node_ilist_splits,
    int* interaction_count,
    float* rmax_out,
    int k,
    float boxsize
) {

    // Finds the minimal radius at which we are guaranteed to have at least k neighbors for any
    // point of a leaf and counts the number of other leaves that are needed to interact at that 
    //  distance

    // This is done in 3 steps:
    // (1) We make a histogram of the number of particles that are guaranteed to be included at
    //     a distance r. (Note that this depends on the upper bound of the distance between leaves)
    // (2) We find the radius at which we have at least k particles
    // (3) We go through the leaves again and count how many we need to check at that distance
    //     (Note that here we need to use the lower bound of the distance between leaves)

    int nodeQ = blockIdx.x;

    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    int ileafQ = min(ileafQ_start + threadIdx.x, ileafQ_end - 1);
    Node leafQ = leaves[ileafQ];
    float3 xQ = leafQ.center;
    float3 extQ = LvlToHalfExt(leafQ.level);

    // We will define bins in units of the diagonal size of leafQ
    // This is the radius at which every point in Q would include every other point in Q
    float rbase2 = 4.0f*dotf3(extQ, extQ); // factor 4, since ext is half the node size
    
    LogBinMap<20> binmap(rbase2, 4.0f);
    CumHist<20> rhist;

    PrefetchList2<int,float> pf_ilist(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

    float rmax2 = INFTY;

    while(!pf_ilist.finished()) {
        Pair<int,float> interaction = pf_ilist.next();
        int nodeT = interaction.first;
        float r2T = interaction.second;

        bool any_accept = syncthreads_or(r2T <= rmax2);
        if(!any_accept) break;

        int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];
        int nleavesT = ileafT_end - ileafT_start;

        __shared__ float3 xT[32];
        __shared__ float3 extT[32];
        __shared__ int npartT[32];
        if(threadIdx.x < nleavesT) {
            Node leafT = leaves[ileafT_start + threadIdx.x];
            xT[threadIdx.x] = leafT.center;
            extT[threadIdx.x] = LvlToHalfExt(leafT.level);
            npartT[threadIdx.x] = leaves_npart[ileafT_start + threadIdx.x];
        }
        __syncthreads();
        
        for(int j = 0; j < nleavesT; j++) {
            float r2 = maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

            int bin = binmap.r2_to_bin(r2);
            rhist.insert(bin, npartT[j]);
        }
        __syncthreads();

        int ibin = rhist.find(k);
        rmax2 = binmap.bin_end(ibin);
    }

    // // Now we have to go again through all leaves and count how many we need to interact with
    // // Note that here we need to use the minimal distance, not the maximal one
    // // since we need to include any leaf that may contain particles within rmax
    int ncount = 0;

    PrefetchList2<int,float> pf_ilist2(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

    while(!pf_ilist2.finished()) {
        Pair<int,float> interaction = pf_ilist2.next();
        int nodeT = interaction.first;
        float r2T = interaction.second;

        bool any_accept = syncthreads_or(r2T <= rmax2);
        if(!any_accept) break;

        int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];
        int nleavesT = ileafT_end - ileafT_start;

        __shared__ float3 xT[32];
        __shared__ float3 extT[32];
        if(threadIdx.x < nleavesT) {
            Node leafT = leaves[ileafT_start + threadIdx.x];
            xT[threadIdx.x] = leafT.center;
            extT[threadIdx.x] = LvlToHalfExt(leafT.level);
        }
        __syncthreads();
        
        for(int j = 0; j < nleavesT; j++) {
            float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

            ncount += (r2 <= rmax2);
        }
        __syncthreads();
    }

    // Output our results:
    if(threadIdx.x <= ileafQ_end - ileafQ_start) {
        rmax_out[ileafQ] = sqrtf(rmax2);
        interaction_count[ileafQ] = ncount;
    }
}

__global__ void KernelInsertInteractions(
    const Node* leaves,
    const int* isplit,
    const int* node_ilist,
    const float* node_ir2list,
    const int* node_ilist_splits,
    const int* out_splits,
    const float* rmax,
    int* ilist_out,
    float* ilist_radii,
    int nmax,
    float boxsize,
    bool get_radii=false
) {
    int nodeQ = blockIdx.x;
    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    int ileafQ = min(ileafQ_start + threadIdx.x, ileafQ_end - 1);
    Node leafQ = leaves[ileafQ];
    float3 xQ = leafQ.center;
    float3 extQ = LvlToHalfExt(leafQ.level);
    float rmaxQ2 = rmax[ileafQ]*rmax[ileafQ];

    int ninserted = 0;

    //PrefetchList<int> pf_ilist(node_ilist, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);
    PrefetchList2<int,float> pf_ilist(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);
    
    while(!pf_ilist.finished()) {
        Pair<int,float> interaction = pf_ilist.next();
        int nodeT = interaction.first;
        float r2T = interaction.second;

        bool any_accept = syncthreads_or(r2T <= rmaxQ2);
        if(!any_accept) break;

        int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];
        int nleavesT = ileafT_end - ileafT_start;

        __shared__ float3 xT[32];
        __shared__ float3 extT[32];

        if(threadIdx.x < nleavesT) {
            Node leafT = leaves[ileafT_start + threadIdx.x];
            xT[threadIdx.x] = leafT.center;
            extT[threadIdx.x] = LvlToHalfExt(leafT.level);
        }
        __syncthreads();
        for(int j = 0; j < nleavesT; j++) {
            float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

            if(r2 <= rmaxQ2){
                int offset = out_splits[ileafQ] + ninserted;

                if(offset >= nmax)
                    return; // We have run out of space, just return

                if(get_radii){
                    if(r2 == 0.f) {
                        // For the direct neighbourhood we add a tiny contribution of the maximum
                        // distance so that sorting guarantees that we start with the leaf itself
                        r2 = 1e-10f*maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);
                    }
                    ilist_radii[offset] = r2;
                }
                ilist_out[offset] = ileafT_start + j;
                ninserted += 1;
            }
        }
    }
}

ffi::Error HostConstructIlist(
        cudaStream_t stream,
        ffi::Buffer<ffi::F32> leaves,
        ffi::Buffer<ffi::S32> leaves_npart,
        ffi::Buffer<ffi::S32> isplit,
        ffi::Buffer<ffi::S32> node_ilist,
        ffi::Buffer<ffi::F32> node_ir2list,
        ffi::Buffer<ffi::S32> node_ilist_splits,
        ffi::ResultBuffer<ffi::F32> radii,
        ffi::ResultBuffer<ffi::S32> leaf_ilist,
        ffi::ResultBuffer<ffi::F32> leaf_ilist_rad,
        ffi::ResultBuffer<ffi::S32> leaf_ilist_splits,
        int k,
        float boxsize,
        bool sort
    )
{
    Node* leaves_ptr = reinterpret_cast<Node*>(leaves.typed_data());
    int nnodes = isplit.element_count() - 1;
    int nleaves = leaves_npart.element_count();

    float* rmax_ptr = radii->typed_data();

    int* lsplits_ptr = leaf_ilist_splits->typed_data();
    cudaMemsetAsync(lsplits_ptr, 0, sizeof(int)*(nleaves+1), stream);

    KernelCountInteractions<<< nnodes, 32, 0, stream>>>(
        leaves_ptr,
        leaves_npart.typed_data(),
        isplit.typed_data(),
        node_ilist.typed_data(),
        node_ir2list.typed_data(),
        node_ilist_splits.typed_data(),
        lsplits_ptr + 1,
        rmax_ptr,
        k,
        boxsize
    );
    
    // Get the prefix sum with CUB
    // We can use the interaction list array as a temporary stoarge
    // This should easily fit in general, but better check that it actually does:
    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, lsplits_ptr + 1, lsplits_ptr + 1, 
        nleaves, stream); // determine the needed allocation size for CUB:

    if (tmp_bytes > leaf_ilist->size_bytes()) {
        return ffi::Error(ffi::ErrorCode::kOutOfRange,
            "Scan allocation too small!  Needed: " +  std::to_string(tmp_bytes) + " bytes." + 
            "Have:" + std::to_string(leaf_ilist->size_bytes()) + " bytes. ");
    }
    
    cub::DeviceScan::InclusiveSum(leaf_ilist->untyped_data(), tmp_bytes, 
        lsplits_ptr + 1, lsplits_ptr + 1, nleaves, stream);

    // Now insert the interactions
    KernelInsertInteractions<<< nnodes, 32, 0, stream>>>(
        leaves_ptr,
        isplit.typed_data(),
        node_ilist.typed_data(),
        node_ir2list.typed_data(),
        node_ilist_splits.typed_data(),
        lsplits_ptr,
        rmax_ptr,
        leaf_ilist->typed_data(),
        leaf_ilist_rad->typed_data(),
        leaf_ilist->element_count(),
        boxsize,
        sort
    );
    
    if(sort) {
        int block_size = 64;
        int smem_size = 512;
        size_t smem_bytes = smem_size * sizeof(KV);
        int nsegs = leaf_ilist_splits->element_count() - 1;
        segmented_bitonic_sort_kv<<< nsegs, block_size, smem_bytes, stream>>>(
            leaf_ilist_rad->typed_data(), leaf_ilist->typed_data(), 
            leaf_ilist_splits->typed_data(), nsegs, smem_size);
    }
    
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
        .Arg<ffi::Buffer<ffi::F32>>() 
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<int>("k")
        .Attr<float>("boxsize")
        .Attr<bool>("sort"),
    {xla::ffi::Traits::kCmdBufferCompatible});

ffi::Error HostSegmentSort(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> key,
    ffi::Buffer<ffi::S32> val,
    ffi::Buffer<ffi::S32> isplit,
    ffi::ResultBuffer<ffi::F32> key_out,
    ffi::ResultBuffer<ffi::S32> val_out,
    int smem_size
)
{
    int nsegs = isplit.element_count() - 1;
    int blocksize = 64;
    size_t smem_bytes = smem_size * sizeof(KV);

    // copy data to output buffers
    cudaMemcpyAsync(key_out->typed_data(), key.typed_data(), key.size_bytes(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(val_out->typed_data(), val.typed_data(), val.size_bytes(), cudaMemcpyDeviceToDevice, stream);

    segmented_bitonic_sort_kv<<< nsegs, blocksize, smem_bytes, stream>>>(
        key_out->typed_data(), val_out->typed_data(), isplit.typed_data(), nsegs, smem_size);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SegmentSort, HostSegmentSort,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() 
        .Arg<ffi::Buffer<ffi::F32>>() // key
        .Arg<ffi::Buffer<ffi::S32>>() // val
        .Arg<ffi::Buffer<ffi::S32>>() // isplit
        .Ret<ffi::Buffer<ffi::F32>>() // key_out
        .Ret<ffi::Buffer<ffi::S32>>() // val_out
        .Attr<int>("smem_size"),
    {xla::ffi::Traits::kCmdBufferCompatible});


NB_MODULE(nb_knn, m) {
    m.def("IlistKNNSearch", []() { return EncapsulateFfiCall(IlistKNNSearch); });
    m.def("ConstructIlist", []() { return EncapsulateFfiCall(ConstructIlist); });
    m.def("SegmentSort", []() { return EncapsulateFfiCall(SegmentSort); });
}