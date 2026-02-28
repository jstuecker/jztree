#ifndef KNN_H
#define KNN_H

#include <cub/cub.cuh>
#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"
#include "xla/ffi/api/ffi.h"
#include "sort.cuh"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

namespace ffi = xla::ffi;

struct PosR {
    float3 pos;
    float r;
};

struct ConstInteractionList {
    const int32_t* spl;
    const int32_t* iother;
    const float* rad2 = nullptr;
};

struct InteractionList {
    int32_t* spl;
    int32_t* iother;
    float* rad2 = nullptr;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         Nearest K Heap                                         */
/* ---------------------------------------------------------------------------------------------- */

template<int K, typename trad>
struct SortedNearestK {
    // Keeps track of the nearest K neighbors that we have seen so far in ascending order of r2.
    // Offers efficient insertion of new candidates

    trad r2s[K];
    int   ids[K];

    __device__ __forceinline__ SortedNearestK(trad r2=INFINITY, int idx=-1){
        #pragma unroll
        for (int i=0;i<K;++i){ 
            r2s[i]=r2;
            ids[i]=idx;
        }
    }

    __device__ __forceinline__ trad max_r2() const {
        return r2s[K-1];
    }

    __device__ __forceinline__ trad r2_at(int k) const {
        // Access with dynamic indexing that can stay in registers
        trad val = static_cast<trad>(0);
        #pragma unroll
        for(int i=0; i<K; i++) {
            val = i==k ? r2s[i] : val;
        }
        return val;
    }

    __device__ __forceinline__ int count_equal_up_to(int imax, trad r2) const {
        // Access with dynamic indexing that can stay in registers
        int count = 0;
        #pragma unroll
        for(int i=0; i<K; i++) {
            count += (r2s[i] == r2) && (i<=imax) ? 1 : 0;
        }
        return count;
    }

    // Insert (r2,id) into ascending r2s[], evicting the largest.
    // I have tried many different variants of this function and this turned out to be the fastest.
    // Important aspects:
    // * no dynamic indexing
    // * no breaks/returns in the loop (stops the compiler from unrolling)
    // * It seems the pattern with the mask helps the compiler to skip unnecessary work
    //   -- effectively achieving an early exit (still not totally sure how it works though...)
    __device__ __forceinline__ void consider(trad r2, int id) {
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


template<int kbins>
struct SortedNearestKWithCounts {
    // Keeps track of the nearest K neighbor radii in ascending order of r2.
    // Allows to insert multiple entries with the same radius

    float r2s[kbins];
    int cts[kbins];
    int total_count, target_count;

    __device__ __forceinline__ SortedNearestKWithCounts(int _target_count, float r2=INFINITY){
        #pragma unroll
        for (int i=0; i<kbins; i++){ 
            r2s[i] = r2;
            cts[i] = 0;
        }
        total_count = 0;
        target_count = _target_count;
    }

    __device__ __forceinline__ float max_r2(int count) const {
        bool active = true;
        int sum = 0;
        float r2 = INFINITY;
        #pragma unroll
        for(int i=0; i<kbins; i++) {
            sum += cts[i];
            r2 = (active && (sum >= count)) ? r2s[i] : r2;
            active = sum < count;
        }
        return r2;
    }

    __device__ __forceinline__ void insert_add(float r2, int num) {
        // Insertion by adding count to the smallest bin that is >= ours
        total_count += num;

        bool active = true;  // true until we've placed the item

        #pragma unroll
        for (int i = 0; i < kbins; i++) {
            bool add = active && (r2s[i] >= r2);

            cts[i] = add ? (cts[i] + num) : cts[i];

            active = active && !add;
        }
    }

    __device__ __forceinline__ void insert_shift(float r2, int num) {
        // Insertion by shifting and discarding the last element
        total_count = total_count + num - cts[kbins-1];

        bool active = true;  // true until we've placed the item

        #pragma unroll
        for (int i = kbins-2; i >= 0; i--) {
            bool take = active && (r2 < r2s[i]);

            r2s[i+1] = active ? (take ? r2s[i] : r2) : r2s[i+1];
            cts[i+1] = active ? (take ? cts[i] : num) : cts[i+1];

            active = take;
        }

        // If we never placed, item belongs at slot 0
        r2s[0] = active ? r2 : r2s[0];
        cts[0] = active ? num : cts[0];
    }

    __device__ __forceinline__ void consider_num(float r2, int num) {
        if (r2 > r2s[kbins-1]) return;

        // We have to insert in such a way that we have in total >= target_count at all times
        // after an initial filling phase.
        // We prefer to shift, but if it doesn't guarantee this constraint, we add instead 
        // (loosing a bit of accuracy)

        bool need_add = (cts[kbins-1] > 0) && (total_count + num - cts[kbins-1] < target_count);
        need_add |= r2 == r2s[kbins-1];

        if(need_add)
            insert_add(r2, num);
        else
            insert_shift(r2, num);
    }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                          Leaf2Leaf KNN                                         */
/* ---------------------------------------------------------------------------------------------- */

template <typename tvec>
struct RminInfo {
    tvec rmin2;
    int nskip_equals;
};

template <int kmax, int dim, typename tvec>
__global__ void KnnLeaf2LeafKernel(
    ConstInteractionList ilist,
    const int* splT,            // leaf-ranges in A
    const Vec<dim,tvec>* xT,    // input positions
    const int* splQ,            // leaf-ranges in B
    const Vec<dim,tvec>* xQ,    // query positions
    tvec* knn_rad,             // output knn radii
    int* knn_id,                // output knn ids
    RminInfo<tvec>* rmin_info,  // Optional I/O, allows to skip nodes that are <= a given radius
    int koffset,                // point where we start filling the output
    int ksize,                  // total k we want to fill in (possibly) multiple passes
    float boxsize               // if > 0, use for periodic wrapping
) {
    int ileafQ = blockIdx.x;
    int iqstart = splQ[ileafQ], iqend = splQ[ileafQ + 1];
    int max_part_smem = blockDim.x;

    bool use_rmin = rmin_info != nullptr;
    RminInfo<tvec> rinfo;

    for(int qoff=iqstart; qoff < iqend; qoff+=blockDim.x) {
        int ipartQ = min(qoff + threadIdx.x, iqend - 1);

        if(use_rmin)
            rinfo = rmin_info[ipartQ];

        Vec<dim,tvec> posQ = xQ[ipartQ];

        SortedNearestK<kmax,tvec> nearestK(INFINITY, -1);

        extern __shared__ char shared_mem[];
        PosId<dim, tvec>* particles = reinterpret_cast<PosId<dim, tvec>*>(shared_mem);

        PrefetchList2<int,float> pf_ilist(
            ilist.iother, ilist.rad2, ilist.spl[ileafQ], ilist.spl[ileafQ + 1]
        );

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int ileafT = interaction.first;
            float r2T = interaction.second;

            // r2T are the lower leaf-leaf distances and the interactions are sorted by these rmax2
            // Once we encounter an interaction that is farther away than any of the current nearest 
            // neighbors, we can skip all subsequent interactions. Since the interaction lists are 
            // build on worst case assumptions, this saves a lot of time in practice!
            bool any_accept = syncthreads_or(r2T <= nearestK.max_r2() * (1.0f + 1e-6f));
            if(!any_accept) break;

            /* Now load the leaf */
            int ipartTstart = splT[ileafT];
            int ipartTend = splT[ileafT + 1];
            for(int ioff = ipartTstart; ioff < ipartTend; ioff += max_part_smem) {
                int nload = min(max_part_smem, ipartTend - ioff);
                for(int i = threadIdx.x; i < nload; i += blockDim.x)
                    particles[i] = {xT[ioff + i], ioff + i};
                __syncthreads();

                // Now search for the nearest neighbors in A
                for (int j = 0; j < nload; j++) {
                    PosId<dim,tvec> p = particles[j];
                    tvec r2 = distance_squared<dim,tvec>(p.pos, posQ, boxsize);

                    if(!use_rmin || (r2 > rinfo.rmin2)) {
                        nearestK.consider(r2, p.id);
                    }
                    else if(r2 == rinfo.rmin2) {
                        if(rinfo.nskip_equals > 0)
                            rinfo.nskip_equals -= 1;
                        else
                            nearestK.consider(r2, p.id);
                    }
                }

                __syncthreads();
            }
        }
        
        if(qoff + threadIdx.x < iqend) {
            #pragma unroll
            for(size_t dk = 0; dk < min(kmax, ksize-koffset); dk++) {
                size_t iout = (size_t)ipartQ * (size_t)ksize + (size_t)koffset + dk;
                knn_rad[iout] = sqrt(nearestK.r2s[dk]);
                knn_id[iout] = nearestK.ids[dk];
            }

            if(use_rmin) {
                // Also write out information of up to which radius we need to skip in next pass
                // Additionally we need to keep track of the number of neighbours that are 
                // exactly equal to rmin2 so that we can skip them in the next pass
                int dk = min(kmax-1, ksize-koffset-1);
                tvec rmax2 = nearestK.r2_at(dk);
                RminInfo<tvec> new_info = {rmax2, nearestK.count_equal_up_to(dk, rmax2)};
                new_info.nskip_equals += rmax2 == rinfo.rmin2 ? rinfo.nskip_equals : 0;

                rmin_info[ipartQ] = new_info;
            }
        }
        __syncthreads();
    }
}


// template<int dim, typename tvec>
template <int dim, typename tvec>
std::string KnnLeaf2Leaf(
    cudaStream_t stream,
    const int* ilist_spl,       // leaf-ranges in ilist
    const int* ilist_iother,           // interaction list
    const float* ilist_r2,      // (lower) interaction rmax2
    const int* splT,            // leaf-ranges in A
    const Vec<dim,tvec>* xT,    // input positions
    const int* splQ,            // leaf-ranges in B
    const Vec<dim,tvec>* xQ,    // query positions
    tvec* knn_rad,             // output knn radii
    int* knn_id,                // output knn ids
    void* rinfo_buf,            // Optional temporary buffer to keep track of inter-pass-information
    int k,                      // Actual k, has to be <= kmax
    float boxsize,              // if > 0, use for periodic wrapping
    size_t size_leaves_query,
    size_t size_part_query
) {
    int block_size = 32;
    int smem_size = block_size*sizeof(PosId<dim, tvec>);
    ConstInteractionList ilist = {ilist_spl, ilist_iother, ilist_r2};

    constexpr int kmax = 32;

    RminInfo<tvec>* rmin_info = nullptr;
    if(k > kmax) { // if multiple passes are needed, we need some additional temporary buffer
        rmin_info = reinterpret_cast<RminInfo<tvec>*>(rinfo_buf);
        cudaMemsetAsync(rmin_info, 0, size_part_query * sizeof(RminInfo<tvec>), stream);
    }

    for(int koffset=0; koffset < k; koffset += kmax) {
        int knext = min(k-koffset, kmax);
        if(knext <= 8)
            KnnLeaf2LeafKernel<8,dim,tvec><<<size_leaves_query, block_size, smem_size, stream>>>(
                ilist, splT, xT, splQ, xQ, knn_rad, knn_id, rmin_info, koffset, k, boxsize
            );
        else if(knext <= 16)
            KnnLeaf2LeafKernel<16,dim,tvec><<<size_leaves_query, block_size, smem_size, stream>>>(
                ilist, splT, xT, splQ, xQ, knn_rad, knn_id, rmin_info, koffset, k, boxsize
            );
        else
            KnnLeaf2LeafKernel<kmax,dim,tvec><<<size_leaves_query, block_size, smem_size, stream>>>(
                ilist, splT, xT, splQ, xQ, knn_rad, knn_id, rmin_info, koffset, k, boxsize
            );
    }
    
    return std::string();
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
/*                                    Interaction list Building                                   */
/* ---------------------------------------------------------------------------------------------- */

template <int kbins, int dim, typename tvec>
__global__ void KnnNode2NodeFindRmax(
    ConstInteractionList par_ilist,
    const int* parent_spl,
    const Node<dim,tvec>* nodes,
    const int* nodes_npart,
    float* rmax_out,
    int k,
    float boxsize
) {
    // Finds the minimal radius at which we are guaranteed to have at least k neighbors for any
    // point inside of each node
    // This is done by going through all interactions and keep track of the maximal needed 
    // radius for each k
    extern __shared__ unsigned char smem[];

    int parentQ = blockIdx.x;
    int inodeQ_start = parent_spl[parentQ], inodeQ_end = parent_spl[parentQ + 1];
    for(int iqoff = inodeQ_start; iqoff < inodeQ_end; iqoff += blockDim.x) {
        // we set overhead threads to the last node to avoid adding many conditionals
        int inodeQ = min(iqoff + threadIdx.x, inodeQ_end - 1); 
        Node<dim,tvec> nodeQ = nodes[inodeQ];
        Vec<dim,tvec> xQ = nodeQ.center;
        Vec<dim,tvec> extQ = LvlToHalfExt<dim,tvec>((int)nodeQ.level);

        SortedNearestKWithCounts<kbins> nearestK(k, INFINITY);

        PrefetchList2<int,float> pf_ilist(
            par_ilist.iother, par_ilist.rad2, par_ilist.spl[parentQ], par_ilist.spl[parentQ + 1]
        );

        float rmax2 = INFINITY;

        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int parentT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmax2);
            if(!any_accept) break;

            int inodeT_start = parent_spl[parentT], inodeT_end = parent_spl[parentT + 1];

            Vec<dim,tvec>* xT = reinterpret_cast<Vec<dim,tvec>*>(smem);
            Vec<dim,tvec>* extT = reinterpret_cast<Vec<dim,tvec>*>(xT + blockDim.x);
            int* npartT = reinterpret_cast<int*>(extT + blockDim.x);

            for(int itoff=inodeT_start; itoff < inodeT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < inodeT_end) {
                    Node<dim,tvec> nodeT = nodes[ilT];
                    xT[threadIdx.x] = nodeT.center;
                    extT[threadIdx.x] = LvlToHalfExt<dim,tvec>((int)nodeT.level);
                    npartT[threadIdx.x] = nodes_npart[ilT];
                }
                __syncthreads();
                
                for(int j = 0; j < min(inodeT_end - itoff, blockDim.x); j++) {
                    float r2 = (float)maxdist2<dim,tvec>(xT[j], xQ, extQ+extT[j], boxsize);

                    nearestK.consider_num(r2, npartT[j]);
                }
                __syncthreads();
            }

            // Add
            rmax2 = nearestK.max_r2(k) * (1.f + 1e-6f);
        }

        // Output our results:
        if(iqoff + threadIdx.x < inodeQ_end) {
            rmax_out[inodeQ] = rmax2;
        }
    }
}

template<int pass, int dim, typename tvec>
__global__ void KnnNode2NodeCountInsert(
    ConstInteractionList par_ilist,
    const int* parent_spl,
    const Node<dim,tvec>* nodes,
    const float* node_rmax2,
    int* node_icount,
    InteractionList node_ilist,
    int nmax,
    float boxsize
) {
    // pass 0: Count node-node interactions that intersect node_rmax2
    //   in-between: calculate offsets
    // pass 1: Insert the interactions into the interaction list

    extern __shared__ unsigned char smem[];

    int parentQ = blockIdx.x;

    int inodeQ_start = parent_spl[parentQ], inodeQ_end = parent_spl[parentQ + 1];
    for(int iqoff = inodeQ_start; iqoff < inodeQ_end; iqoff += blockDim.x) {
        int inodeQ = min(iqoff + threadIdx.x, inodeQ_end - 1);
        Node<dim,tvec> nodeQ = nodes[inodeQ];
        Vec<dim,tvec> xQ = nodeQ.center;
        Vec<dim,tvec> extQ = LvlToHalfExt<dim,tvec>((int)nodeQ.level);
        float rmaxQ2 = node_rmax2[inodeQ];
        int out_offset = node_ilist.spl[inodeQ];

        int ncount = 0;

        PrefetchList2<int,float> pf_ilist(
            par_ilist.iother, par_ilist.rad2, par_ilist.spl[parentQ], par_ilist.spl[parentQ + 1]
        );
        
        while(!pf_ilist.finished()) {
            Pair<int,float> interaction = pf_ilist.next();
            int parentT = interaction.first;
            float r2T = interaction.second;

            bool any_accept = syncthreads_or(r2T <= rmaxQ2);
            if(!any_accept) break;

            int inodeT_start = parent_spl[parentT], inodeT_end = parent_spl[parentT + 1];

            Vec<dim,tvec>* xT = reinterpret_cast<Vec<dim,tvec>*>(smem);
            Vec<dim,tvec>* extT = reinterpret_cast<Vec<dim,tvec>*>(xT + blockDim.x);

            for(int itoff=inodeT_start; itoff < inodeT_end; itoff += blockDim.x) {
                int ilT = itoff + threadIdx.x;
                if(ilT < inodeT_end) {
                    Node<dim,tvec> nodeT = nodes[ilT];
                    xT[threadIdx.x] = nodeT.center;
                    extT[threadIdx.x] = LvlToHalfExt<dim,tvec>((int)nodeT.level);
                }
                __syncthreads();
                for(int j = 0; j < min(inodeT_end - itoff, blockDim.x); j++) {
                    float r2 = (float)mindist2<dim,tvec>(xT[j], xQ, extQ+extT[j], boxsize);

                    if((iqoff + threadIdx.x < inodeQ_end) && (r2 <= rmaxQ2)){
                        if(pass == 1) {
                            int offset = out_offset + ncount;

                            if(offset >= nmax)
                                break; // We have run out of space, just return

                            if(r2 == 0.f) {
                                // For the direct neighbourhood we add a tiny contribution of the maximum
                                // distance so that sorting guarantees that we start with the node itself
                                r2 = 1e-10f*(float)maxdist2<dim,tvec>(xT[j], xQ, extQ+extT[j], boxsize);
                            }

                            node_ilist.rad2[offset] = r2;
                            node_ilist.iother[offset] = itoff + j;
                        }
                        ncount += 1;
                    }
                }
                __syncthreads();
            }
        }

        if((pass == 0) && (iqoff + threadIdx.x < inodeQ_end)) {
            node_icount[inodeQ] = ncount;
        }
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Host function                                         */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
ffi::Error KnnNode2Node(
    cudaStream_t stream,
    const int32_t* parent_ilist_spl,
    const int32_t* parent_ilist_ioth,
    const float* parent_ilist_r2,
    const int32_t* parent_spl,
    const Node<dim,tvec>* nodes,
    const int32_t* nodes_npart,
    // outputs
    float* node_rmax2,
    int32_t* node_ilist_spl,
    int32_t* node_ilist_ioth,
    float* node_ilist_r2,
    // parameters
    const int k,
    const size_t blocksize_fill,
    const size_t blocksize_sort,
    float boxsize,
    // parameters that can be infered inside ffi:
    const int size_parents,
    const int size_nodes,
    const size_t node_ilist_size
) {
    ConstInteractionList par_ilist = {parent_ilist_spl, parent_ilist_ioth, parent_ilist_r2};
    InteractionList node_ilist = {node_ilist_spl, node_ilist_ioth, node_ilist_r2};

    cudaMemsetAsync(node_ilist.spl, 0, sizeof(int32_t)*(size_nodes+1), stream);

    size_t smem_alloc_size = blocksize_fill * (2*sizeof(Vec<dim,tvec>) + sizeof(int));

    if(k <= 16) {
        KnnNode2NodeFindRmax<8,dim,tvec><<<size_parents, blocksize_fill, smem_alloc_size, stream>>>(
            par_ilist, parent_spl, nodes, nodes_npart, node_rmax2, k, boxsize
        );
    }
    if(k <= 32) {
        KnnNode2NodeFindRmax<16,dim,tvec><<<size_parents, blocksize_fill, smem_alloc_size, stream>>>(
            par_ilist, parent_spl, nodes, nodes_npart, node_rmax2, k, boxsize
        );
    }
    else {
        KnnNode2NodeFindRmax<32,dim,tvec><<<size_parents, blocksize_fill, smem_alloc_size, stream>>>(
            par_ilist, parent_spl, nodes, nodes_npart, node_rmax2, k, boxsize
        );
    }

    smem_alloc_size = blocksize_fill * 2*sizeof(Vec<dim,tvec>);
    KnnNode2NodeCountInsert<0,dim,tvec><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
        par_ilist, parent_spl, nodes, node_rmax2,
        node_ilist.spl + 1, node_ilist, // output
        node_ilist_size, boxsize    // parameters
    );

    // Get the prefix sum with CUB
    // We can use the interaction list array as a temporary stoarge
    // This should easily fit in general, but better check that it actually does:
    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(
        nullptr, tmp_bytes, node_ilist.spl + 1, node_ilist.spl + 1,  size_nodes, stream
    ); // determine the needed allocation size for CUB:

    if (tmp_bytes > node_ilist_size * sizeof(int)) {
        return ffi::Error(ffi::ErrorCode::kOutOfRange,
            "Scan allocation too small!  Needed: " +  std::to_string(tmp_bytes) + " bytes." + 
            "Have:" + std::to_string(node_ilist_size * sizeof(int)) + " bytes. ");
    }
    cub::DeviceScan::InclusiveSum(
        node_ilist_ioth, tmp_bytes, node_ilist.spl + 1, node_ilist.spl + 1, size_nodes, stream
    );

    // Now insert the interactions
    smem_alloc_size = blocksize_fill * 2*sizeof(Vec<dim,tvec>);
    KnnNode2NodeCountInsert<1,dim,tvec><<< size_parents, blocksize_fill, smem_alloc_size, stream>>>(
        par_ilist, parent_spl, nodes, node_rmax2,
        nullptr, node_ilist, // output
        node_ilist_size, boxsize    // parameters
    );

    // Now sort the interaction list segments. (This will allow early exit in the knn search)
    // Note: We cannot use CUB here, since its segmented search is not jax-compatible. 
    // (Probably because it uses dynamic dispatches internally)
    // Since most segments are small enough to be sorted in shared memory, this adds a very
    // small overhead (~ O(2ms) for 1M particles). So it is well worth it.
    int smem_size = 512;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));
    segmented_bitonic_sort_kv<<<size_nodes, blocksize_sort, smem_bytes, stream>>>(
        node_ilist_r2, node_ilist_ioth, node_ilist.spl, size_nodes, smem_size
    );
    
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           SegmentSort                                          */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SegmentSort(
    cudaStream_t stream,
    const int32_t* spl,
    const float* key,
    const int32_t* val,
    float* key_out,
    int32_t* val_out,
    const int32_t size_segs,
    const int32_t size_keys,
    const size_t smem_size
) {
    int blocksize = 64;
    size_t smem_bytes = smem_size * (sizeof(float) + sizeof(int32_t));

    cudaMemcpyAsync(key_out, key, size_keys*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(val_out, val, size_keys*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    segmented_bitonic_sort_kv<<< size_segs, blocksize, smem_bytes, stream>>>(
        key_out, val_out, spl, size_segs, smem_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

#endif // KNN_H