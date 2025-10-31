#ifndef COMMON_MATH_H
#define COMMON_MATH_H

/* ---------------------------------------------------------------------------------------------- */
/*                                           Vector Math                                          */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ float3 float3sum(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 float3diff(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float norm2(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Integer Math                                          */
/* ---------------------------------------------------------------------------------------------- */

// Calculates integer division in round-up mode
inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ __forceinline__
float powi_upto6(float x, int n) {
    switch (n) {
        case 0: return 1.0f;
        case 1: return x;
        case 2: return x * x;
        case 3: { float x2 = x * x; return x2 * x; }
        case 4: { float x2 = x * x; return x2 * x2; }
        case 5: { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
        case 6: { float x2 = x * x; float x3 = x2 * x; return x3 * x3; }
        default: // fallback if someone passes >6
            return powf(x, (float)n);
    }
}

__device__ __forceinline__ float fact_upto6f(unsigned k) {
    float f = 1.f;
    f *= (k >= 2) ? 2.f : 1.f;
    f *= (k >= 3) ? 3.f : 1.f;
    f *= (k >= 4) ? 4.f : 1.f;
    f *= (k >= 5) ? 5.f : 1.f;
    f *= (k >= 6) ? 6.f : 1.f;
    return f;
}

__device__ __forceinline__ float binomial(unsigned n, unsigned k) {
    if (k > n) return 0.f;
    float res = 1.f;
    for (unsigned i = 1; i <= k; i++) {
        res *= float(n - (k - i)) / float(i);
    }
    return res;
}

__device__ __forceinline__ float fact3f(unsigned kx, unsigned ky, unsigned kz) {
    return fact_upto6f(kx) * fact_upto6f(ky) * fact_upto6f(kz);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Multipole Indexing                                       */
/* ---------------------------------------------------------------------------------------------- */

#define NCOMB(p) (((p) + 1) * ((p) + 2) * ((p) + 3) / 6)

__device__ __forceinline__ constexpr  int multi_to_flat(const int kx, const int ky, const int kz) {
    int p = kx + ky + kz;
    int npoff = ((p+2)*(p+1)*p) / 6; // offset of the p-th symmeric tensor
    int off = npoff + (kz*(2*p + 3 - kz))/2 + ky;

    return off > 0 ? off : 0; // Ensure we don't return negative indices
}

template<int pmax>
__device__ __forceinline__ constexpr  int3 flat_to_multi(const int kflat) {
    int i = 0, ksum, kz, ky;
    #pragma unroll
    for(ksum=0; ksum <= pmax; ksum++) {
        int nadd = ((ksum+2)*(ksum+1)) >> 1;
        if (i + nadd > kflat)
            break;
        i += nadd;
    }
    #pragma unroll
    for(kz=0; kz <= ksum; kz++) {
        int nadd = (ksum-kz+1);
        if (i + nadd > kflat)
            break;
        i += nadd;
    }
    ky = kflat - i;

    return int3{ksum-ky-kz, ky, kz};
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            Node Math                                           */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ float3 LvlToExt(int level) {
    // Converts a node's or leaf's binary level to its extend per dimension

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 3000) / 3 - 1000;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2);
    int lz = olvl + (omod >= 1);
    
    return make_float3(ldexpf(1.0f, lx), ldexpf(1.0f, ly), ldexpf(1.0f, lz));
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Bit operations                                         */
/* ---------------------------------------------------------------------------------------------- */

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

// Whether pos1 should appear before pos2 in a z-curve ordering
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

/* ---------------------------------------------------------------------------------------------- */
/*                                     Warp and Group helpers                                     */
/* ---------------------------------------------------------------------------------------------- */

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(__activemask(), v, offset);
    return v;
}

template<int N>
__device__ void add_warp_reduced(const float (&Dn)[N], float* __restrict__ Dsum, bool valid) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        float v = valid ? Dn[k] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(&Dsum[k], v);
    }
}

#endif // COMMON_MATH_H