#include <type_traits>

#include <cmath>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "common/segment_sort.cuh"
#include "common/data.cuh"
#include "common/math.cuh"
#include "common/knn_math.cuh"
#include "common/iterators.cuh"
#include <cub/cub.cuh>

#define INTERACTION_BINS 16

namespace nb = nanobind;
namespace ffi = xla::ffi;

template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Histogram Data Structures                                   */
/* ---------------------------------------------------------------------------------------------- */

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
        if(isnan(r2) || (r2 >= INFTY)) return BINS; // overflow bin

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

/* ---------------------------------------------------------------------------------------------- */
/*                                   Interaction Counting Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */

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
    float bins_per_log2,
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

    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;
    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    for(int iqoff = ileafQ_start; iqoff < ileafQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last leaf to avoid adding many conditionals
        int ileafQ = min(iqoff + threadIdx.x, ileafQ_end - 1); 
        Node leafQ = leaves[ileafQ];
        float3 xQ = leafQ.center;
        float3 extQ = LvlToHalfExt(leafQ.level);

        // We will define bins in units of the diagonal size of leafQ
        // This is the radius at which every point in Q would include every other point in Q
        float rbase2 = 4.0f*dotf3(extQ, extQ); // factor 4, since ext is half the node size
        
        LogBinMap<INTERACTION_BINS> binmap(rbase2, bins_per_log2);
        CumHist<INTERACTION_BINS> rhist;

        PrefetchList2<int,float> pf_ilist(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

        float rmax2 = INFTY;

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int nodeT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmax2);
            if(!any_accept) break;

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);
            int* npartT = reinterpret_cast<int*>(extT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < ileafT_end) {
                    Node leafT = leaves[ilT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                    npartT[threadIdx.x] = leaves_npart[ilT];
                }
                __syncthreads();
                
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    float r2 = maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    int bin = binmap.r2_to_bin(r2);
                    rhist.insert(bin, npartT[j]);
                }
                __syncthreads();

                int ibin = rhist.find(k);
                // add a small safety margin, since our floating point operations might not
                // be exactly invertible
                rmax2 = binmap.bin_end(ibin) * (1.f + 1e-6f); 
            }
        }

        // // Now we have to go again through all leaves and count how many we need to interact with
        // // Note that here we need to use the minimal distance, not the maximal one
        // // since we need to include any leaf that may contain particles within rmax
        int ncount = 0;

        pf_ilist.restart(node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int nodeT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmax2);
            if(!any_accept) break;

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < ileafT_end) {
                    Node leafT = leaves[ilT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                }
                __syncthreads();
                
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    ncount += (r2 <= rmax2);
                }
                __syncthreads();
            }
        }

        // Output our results:
        if(iqoff + threadIdx.x < ileafQ_end) {
            rmax_out[ileafQ] = rmax2;
            interaction_count[ileafQ] = ncount;
        }
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    KernelInsertInteractions                                    */
/* ---------------------------------------------------------------------------------------------- */

__global__ void KernelInsertInteractions(
    const Node* leaves,
    const int* isplit,
    const int* node_ilist,
    const float* node_ir2list,
    const int* node_ilist_splits,
    const int* out_splits,
    const float* rmax2,
    int* ilist_out,
    float* ilist_radii,
    int nmax,
    float boxsize
) {
    extern __shared__ unsigned char smem[];

    int nodeQ = blockIdx.x;

    int ileafQ_start = isplit[nodeQ], ileafQ_end = isplit[nodeQ + 1];
    for(int iqoff = ileafQ_start; iqoff < ileafQ_end; iqoff += blockDim.x) {
        int ileafQ = min(iqoff + threadIdx.x, ileafQ_end - 1);
        Node leafQ = leaves[ileafQ];
        float3 xQ = leafQ.center;
        float3 extQ = LvlToHalfExt(leafQ.level);
        float rmaxQ2 = rmax2[ileafQ];
        int out_offset = out_splits[ileafQ];

        int ninserted = 0;

        PrefetchList2<int,float> pf_ilist(node_ilist, node_ir2list, node_ilist_splits[nodeQ], node_ilist_splits[nodeQ + 1]);
        
        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int nodeT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmaxQ2);
            if(!any_accept) break;

            int ileafT_start = isplit[nodeT], ileafT_end = isplit[nodeT + 1];

            float3* xT = reinterpret_cast<float3*>(smem);
            float3* extT = reinterpret_cast<float3*>(xT + blockDim.x);

            for(int itoff=ileafT_start; itoff < ileafT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < ileafT_end) {
                    Node leafT = leaves[ilT];
                    xT[threadIdx.x] = leafT.center;
                    extT[threadIdx.x] = LvlToHalfExt(leafT.level);
                }
                __syncthreads();
                for(int j = 0; j < min(ileafT_end - itoff, blockDim.x); j++) {
                    float r2 = mindist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);

                    if((iqoff + threadIdx.x < ileafQ_end) && (r2 <= rmaxQ2)){
                        int offset = out_offset + ninserted;

                        if(offset >= nmax)
                            break; // We have run out of space, just return

                        if(r2 == 0.f) {
                            // For the direct neighbourhood we add a tiny contribution of the maximum
                            // distance so that sorting guarantees that we start with the leaf itself
                            r2 = 1e-10f*maxdist2(xT[j], xQ, sumf3(extQ, extT[j]), boxsize);
                        }

                        ilist_radii[offset] = r2;
                        ilist_out[offset] = itoff + j;
                        ninserted += 1;
                    }
                }
                __syncthreads();
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
        ffi::ResultBuffer<ffi::F32> rmax2,
        ffi::ResultBuffer<ffi::S32> leaf_ilist,
        ffi::ResultBuffer<ffi::F32> leaf_ilist_rad,
        ffi::ResultBuffer<ffi::S32> leaf_ilist_splits,
        int k,
        size_t blocksize_fill,
        size_t blocksize_sort,
        float rfac_maxbin,
        float boxsize
    )
{
    Node* leaves_ptr = reinterpret_cast<Node*>(leaves.typed_data());
    int nnodes = isplit.element_count() - 1;
    int nleaves = leaves_npart.element_count();
    int* lsplits_ptr = leaf_ilist_splits->typed_data();
    cudaMemsetAsync(lsplits_ptr, 0, sizeof(int)*(nleaves+1), stream);

    constexpr int BINS = INTERACTION_BINS;
    float bins_per_log2 = BINS / log2f(rfac_maxbin);

    size_t smem_alloc_size = blocksize_fill * (2*sizeof(float3) + sizeof(int));

    KernelCountInteractions<<< nnodes, blocksize_fill, smem_alloc_size, stream>>>(
        leaves_ptr,
        leaves_npart.typed_data(),
        isplit.typed_data(),
        node_ilist.typed_data(),
        node_ir2list.typed_data(),
        node_ilist_splits.typed_data(),
        lsplits_ptr + 1,
        rmax2->typed_data(),
        k,
        bins_per_log2,
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
    smem_alloc_size = blocksize_fill * 2*sizeof(float3);
    KernelInsertInteractions<<< nnodes, blocksize_fill, smem_alloc_size, stream>>>(
        leaves_ptr,
        isplit.typed_data(),
        node_ilist.typed_data(),
        node_ir2list.typed_data(),
        node_ilist_splits.typed_data(),
        lsplits_ptr,
        rmax2->typed_data(),
        leaf_ilist->typed_data(),
        leaf_ilist_rad->typed_data(),
        leaf_ilist->element_count(),
        boxsize
    );
    
    
    // Now sort the interaction list segments. (This will allow early exit in the knn search)
    // Note: We cannot use CUB here, since its segmented search is not jax-compatible. 
    // (Probably because it uses dynamic dispatches internally)
    // Since most segments are small enough to be sorted in shared memory, this adds a very
    // small overhead (~ O(2ms) for 1M particles). So it is well worth it.
    int smem_size = 512;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));
    int nsegs = leaf_ilist_splits->element_count() - 1;
    segmented_bitonic_sort_kv<<< nsegs, blocksize_sort, smem_bytes, stream>>>(
        leaf_ilist_rad->typed_data(), leaf_ilist->typed_data(), 
        leaf_ilist_splits->typed_data(), nsegs, smem_size);
    
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
        .Attr<size_t>("blocksize_fill")
        .Attr<size_t>("blocksize_sort")
        .Attr<float>("rfac_maxbin")
        .Attr<float>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible});


/* ---------------------------------------------------------------------------------------------- */
/*                                        Segment Sort FFI                                        */
/* ---------------------------------------------------------------------------------------------- */

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
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int));

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


NB_MODULE(ffi_knn, m) {
    m.def("ConstructIlist", []() { return EncapsulateFfiCall(ConstructIlist); });
    m.def("SegmentSort", []() { return EncapsulateFfiCall(SegmentSort); });
}