#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// A wrapper to encapsulate an FFI call
template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

// =============================================================
// Multipole Translators
// =============================================================

#define MAXP 7
__constant__ int multi_index_to_flat[MAXP][MAXP][MAXP];
__constant__ int3 flat_to_multi_index[(MAXP*(MAXP+1)*(MAXP+2))/6];
bool index_tables_initialzied = false;

// Set's up the table needed for maping mullti indices (nx,ny,nz) to flat indices
void setup_index_tables() 
{
    if (index_tables_initialzied) {
        return;
    }

    int index_table[MAXP][MAXP][MAXP];
    int3 inverse_table[(MAXP*(MAXP+1)*(MAXP+2))/6];

    int idx = 0;
    for(int p=0; p < MAXP; p++) {
        for(int i=0; i <= p; i++) {
            for(int j=0; j <= p - i; j++) {
                int k = p - i - j;
                index_table[k][j][i] = idx;
                inverse_table[idx] = int3{k, j, i};
                idx += 1;
            }
        }
    }

    // copy to constant memory
    cudaMemcpyToSymbol(multi_index_to_flat, index_table, sizeof(index_table), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(flat_to_multi_index, inverse_table, sizeof(inverse_table));

    index_tables_initialzied = true;
}

template<int p>
__device__ void setupGn(float r2, float eps2, float* G)
{
    // The derivatives of (1/r d/dr)^n G_0  with G_0 = 1/r
    float rinv = rsqrtf(r2 + eps2); 
    float rinv2 = rinv*rinv;
    G[0] = 1.0f * rinv;

    #pragma unroll
    for (int n = 1; n <= p; n++) {
        G[n] = -(2*n-1) * G[n-1] * rinv2;
    }
}

#define NCOMB(p) (((p) + 1) * ((p) + 2) * ((p) + 3) / 6)

__device__ __forceinline__ float get_xk(const float3& x, int k) {
    // Get's the k-th component of a float3 vector
    // Note that we need to do it this way (rather than with an array type vector),
    // because it is not possible to dynamically index register arrays (and if you do
    // they will end up in local memory (=very slow))
    return (k == 0) ? x.x : (k == 1 ? x.y : x.z);
}

template<int p>
__device__ void setupDnG(float3 dx, float eps2, float *Dn) {
    // Set's up the cartesian derivatives of the Green's function
    // Using the method described in Tausch (2003):
    // "The fast multipole method for arbitrary Green’s functions"
    // This works by using a recurrence between the derivatives of the Green's function
    // We start at D0 G(q), go to D1 G(q+1), D2 G(q+2) ...

    // float *dxflat = reinterpret_cast<float*>(&dx);

    float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

    float G[p+1];
    setupGn<p>(r2, eps2, G);
    Dn[0] = G[p];

    for(int q=p-1; q >= 0; q--) {
        for(int iflat=NCOMB(p-q)-1; iflat >= 1; iflat--) { // we loop backwards to not overwrite needed values
            int3 n = flat_to_multi_index[iflat];
            // choose ek, the direction with the largest index
            int nk = n.x >= n.y ? n.x : n.y;
            nk = nk >= n.z ? nk : n.z;
            int k = n.x == nk ? 0 : (n.y == nk ? 1 : 2);
            int new_n[3] = {n.x, n.y, n.z};
            // Add contribution from n - ek
            new_n[k] = max(0, new_n[k] - 1);

            float Dnew = get_xk(dx, k)*Dn[multi_index_to_flat[new_n[0]][new_n[1]][new_n[2]]];
            // Add contribution from n - 2*ek
            new_n[k] = max(0, new_n[k] - 1);
            Dnew += (nk-1)*Dn[multi_index_to_flat[new_n[0]][new_n[1]][new_n[2]]];

            Dn[iflat] = Dnew;
        }
        Dn[0] = G[q];
    }
}

template<int p>
__global__ void IlistM2LKernel(const float3 *x, const float *mp, int2 *interactions, int *iminmax, float *Lout, size_t interactions_per_block, float epsilon) {
    const size_t blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    int nc = NCOMB(p);

    float invfact[MAXP] = {1.f, 1.f, 1./2.f, 1.f/6.f, 1.f/24.f, 1.f/120.f, 1.f/720.f};

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + blocksize * (blockIdx.x * interactions_per_block + iint) + threadIdx.x;
        if (int_id >= imax) {
            return;
        }

        int iA = interactions[int_id].x, iB = interactions[int_id].y;
        float3 xA = x[iA], xB = x[iB];
        float3 dx = {xB.x - xA.x, xB.y - xA.y, xB.z - xA.z};
        float Dn[NCOMB(p)];
        setupDnG<p>(dx, epsilon2, Dn);

        float Mp[NCOMB(p)];
        for (int iM=0; iM < nc; iM++) {
            Mp[iM] = mp[iB * nc + iM];
        }

        for (int iL = 0; iL < nc; iL++) {
            float Lnew = 0.f;
            int3 k = flat_to_multi_index[iL];
            int ksum = k.x + k.y + k.z;
            for (int iN = 0; iN < NCOMB(p-ksum); iN++) {
                int3 n = flat_to_multi_index[iN];
                int nsum = n.x + n.y + n.z;
                
                float Dnk = Dn[multi_index_to_flat[k.x+n.x][k.y+n.y][k.z+n.z]];
                float Mpn = Mp[iN];
                
                
                Lnew += Dnk * Mpn * (invfact[n.x]*invfact[n.y]*invfact[n.z]);
            }

            float sign = ksum % 2 == 0 ? -1.f : 1.f;
            atomicAdd(&Lout[iA * nc + iL], Lnew * sign * invfact[k.x] * invfact[k.y] * invfact[k.z]);
        }
    }
}

void launch_IlistM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream, const float3 *x, const float *mp, int2 *interactions, int *iminmax, float *Lout, size_t interactions_per_block, float epsilon) {
    // This launch mechanic is needed so that p can be treated as a compile time constant
    // I wish there was a simpler way...
    switch(p) {
        case 1: IlistM2LKernel<1><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 2: IlistM2LKernel<2><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 3: IlistM2LKernel<3><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 4: IlistM2LKernel<4><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 5: IlistM2LKernel<5><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 6: IlistM2LKernel<6><<<grid_size, block_size, 0, stream>>>(x, mp, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        default: throw std::runtime_error("Unsupported p value for IlistM2LKernel"); break;
    }
}

ffi::Error IlistM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> mp, ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax,  ffi::ResultBuffer<ffi::F32> loc, int p, size_t block_size, size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    const size_t grid_size = (ninteractions + block_size*interactions_per_block - 1) / (block_size*interactions_per_block);

    auto* xfloat3 = reinterpret_cast<const float3*>(x.typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());

    cudaMemsetAsync(loc->typed_data(), 0, mp.element_count() * sizeof(float), stream);

    setup_index_tables();

    launch_IlistM2LKernel(p, grid_size, block_size, stream, xfloat3, mp.typed_data(), interactions_i2, iminmax.typed_data(), loc->typed_data(), interactions_per_block, epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistM2L, IlistM2LHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()     // interactions
        .Arg<ffi::Buffer<ffi::S32>>()     // iminmax
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int>("p")
        .Attr<size_t>("block_size")
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});


template<int p>
__global__ void IlistLeaf2NodeM2LKernel(
        const float3 *xnodes, const float4 *xm, int32_t *isplit, int2 *interactions, int *iminmax, 
        float *Lout, size_t interactions_per_block, float epsilon
    )  {
    const int blocksize = blockDim.x;
    float epsilon2 = epsilon * epsilon;

    int imin = iminmax[0], imax = iminmax[1];

    float invfact[MAXP] = {1.f, 1.f, 1./2.f, 1.f/6.f, 1.f/24.f, 1.f/120.f, 1.f/720.f};

    constexpr int ncomb = NCOMB(p);
    // __shared__ float Dn[32][ncomb];
    float Dn[NCOMB(p)];

    for (int iint = 0; iint < interactions_per_block; iint++) {
        int int_id = imin + iint * gridDim.x + blockIdx.x ; //blockIdx.x * interactions_per_block + iint;
        if (int_id >= imax) {
            return;
        }

        int2 interaction = interactions[int_id];
        int iNode = interaction.x, iLeaf = interaction.y;
        int iPartStart = isplit[iLeaf], iPartEnd = isplit[iLeaf + 1];

        int ipart = threadIdx.x + iPartStart;
        bool valid = (ipart < iPartEnd);
        ipart = valid ? ipart : iPartEnd-1; // if not valid, keep in bounds, we discard the result later

        float4 xmpart = xm[ipart];
        float3 xnode = xnodes[iNode];

        float3 dx = {xmpart.x - xnode.x, xmpart.y - xnode.y, xmpart.z - xnode.z};
        setupDnG<p>(dx, epsilon2, Dn);

        /* Calculating the derivative tensor is clearly the bottleneck here, have to optimize it later*/
        // setupDnG<p>(dx, epsilon2, &Dn[threadIdx.x][0]);


        __syncthreads();
        
        if(valid) {
            for(int kflat=0; kflat<ncomb; kflat++) {
                int3 k = flat_to_multi_index[kflat];
                int ksum = k.x + k.y + k.z;
                float sign = ksum % 2 == 0 ? -1.f : 1.f;
                float Lnew = Dn[kflat] * sign * invfact[k.x] * invfact[k.y] * invfact[k.z];

                atomicAdd(&Lout[iNode*ncomb + kflat],  Lnew); //Dn[0][k]
            }
        }
    }
}



void launch_IlistLeaf2NodeM2LKernel(int p, size_t grid_size, size_t block_size, cudaStream_t stream, const float3 *xnodes, const float4 *xm, int32_t *isplit, int2 *interactions, int *iminmax, float *Lout, size_t interactions_per_block, float epsilon) {
    // This launch mechanic is needed so that p can be treated as a compile time constant
    // I wish there was a simpler way...
    switch(p) {
        case 1: IlistLeaf2NodeM2LKernel<1><<<grid_size, block_size, 0, stream>>>(xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 2: IlistLeaf2NodeM2LKernel<2><<<grid_size, block_size, 0, stream>>>(xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 3: IlistLeaf2NodeM2LKernel<3><<<grid_size, block_size, 0, stream>>>(xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 4: IlistLeaf2NodeM2LKernel<4><<<grid_size, block_size, 0, stream>>>(xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 5: IlistLeaf2NodeM2LKernel<5><<<grid_size, block_size, 0, stream>>>(xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;
        case 6: IlistLeaf2NodeM2LKernel<6><<<grid_size, block_size, 0, stream>>>(xnodes, xm, isplit, interactions, iminmax, Lout, interactions_per_block, epsilon); break;

        default: throw std::runtime_error("Unsupported p value for IlistM2LKernel"); break;
    }
}

ffi::Error IlistLeaf2NodeM2LHost(cudaStream_t stream, ffi::Buffer<ffi::F32> xnodes, ffi::Buffer<ffi::F32> xm, ffi::Buffer<ffi::S32> isplit, ffi::Buffer<ffi::S32> interactions, ffi::Buffer<ffi::S32> iminmax, ffi::ResultBuffer<ffi::F32> loc, int p, size_t interactions_per_block, float epsilon) {
    size_t ninteractions = interactions.element_count() / 2;
    size_t block_size = 32; // In this case a constant blocksize makes the kernel simpler

    size_t grid_size = (ninteractions + interactions_per_block - 1) / interactions_per_block;

    auto* xnodes_float3 = reinterpret_cast<const float3*>(xnodes.typed_data());
    auto* xm_float4 = reinterpret_cast<const float4*>(xm.typed_data());
    auto* interactions_i2 = reinterpret_cast<int2*>(interactions.typed_data());
    
    cudaMemsetAsync(loc->typed_data(), 0, loc->element_count() * sizeof(float), stream);

    setup_index_tables();

    launch_IlistLeaf2NodeM2LKernel(p, grid_size, block_size, stream, xnodes_float3, xm_float4, isplit.typed_data(), interactions_i2, iminmax.typed_data(), loc->typed_data(), interactions_per_block, epsilon);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IlistLeaf2NodeM2L, IlistLeaf2NodeM2LHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()     // xnodes
        .Arg<ffi::Buffer<ffi::F32>>()     // xm
        .Arg<ffi::Buffer<ffi::S32>>()     // isplit
        .Arg<ffi::Buffer<ffi::S32>>()     // interactions
        .Arg<ffi::Buffer<ffi::S32>>()     // iminmax
        .Ret<ffi::Buffer<ffi::F32>>()     // Lk output
        .Attr<int>("p")
        .Attr<size_t>("interactions_per_block")
        .Attr<float>("epsilon"),
    {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(nb_multipoles, m) {
    m.def("ilist_m2l", []() { return EncapsulateFfiCall(IlistM2L); });
    m.def("ilist_leaf2node_m2l", []() { return EncapsulateFfiCall(IlistLeaf2NodeM2L); });
}