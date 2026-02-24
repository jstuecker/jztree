#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#include "data.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                           Vector Math                                          */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float norm2(float3 a) {
    return dot(a, a);
}

__host__ __device__ __forceinline__
float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ __forceinline__
float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__
float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__
float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ __forceinline__
float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ __forceinline__
float4 operator*(float a, float4 b) {
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__device__ __forceinline__ float3 sumf3(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float dotf3(const float3 &a, const float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Kahan Summation                                        */
/* ---------------------------------------------------------------------------------------------- */

__forceinline__ __device__ void kahan_add(float &sum, float add, float &c) {
    // Cancels summation error with an extra variable c, that needs to start at 0
    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    float y = add - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

__forceinline__ __device__ void kahan_add_f4(float4 &sum, float4 add, float4 &c) {
    kahan_add(sum.x, add.x, c.x);
    kahan_add(sum.y, add.y, c.y);
    kahan_add(sum.z, add.z, c.z);
    kahan_add(sum.w, add.w, c.w);
}

template<int num>
__forceinline__ __device__ void kahan_add_array(float *sum, float *add, float *c) {
    #pragma unroll
    for (int i = 0; i < num; i++) {
        kahan_add(sum[i], add[i], c[i]);
    }
}

template <bool kahan>
__forceinline__ __device__ void add_f4(float4 &sum, float4 add, float4 &c) {
    if(kahan)
        kahan_add_f4(sum, add, c);
    else
        sum = sum + add;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Integer Math                                          */
/* ---------------------------------------------------------------------------------------------- */

// Calculates integer division in round-up mode
__host__ __device__ __forceinline__ int div_ceil(int a, int b) {
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
/*                                         Bit operations                                         */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ int32_t msb_xor_float(float a, float b) {
    // Finds the most significant bit that differs between x and y
    // For floating point numbers we need to treat the exponent and the mantissa differently:
    // If the exponent differs, then the (power of two) of the difference is given by the larger
    // exponent.
    // If the exponent is the same, then we need to compare the mantissas. The (power of two) of the
    // difference is then given by the differing bit in the mantissa, offset by the exponent

    if (isnan(a) || isnan(b) || (signbit(a) != signbit(b))) {
        return 128;  // One higher than highest possible float diff (2**127)
    }
    uint32_t a_bits = (uint32_t)__float_as_int(fabsf(a));
    uint32_t b_bits = (uint32_t)__float_as_int(fabsf(b));

    int32_t a_exp = (int32_t)(a_bits >> 23) - 127;
    int32_t b_exp = (int32_t)(b_bits >> 23) - 127;

    if (a_exp == b_exp) { // If both floats have the same exponent, we need to compare mantissas
        // clz counts bit-zeros from the left. There will be always 8 leading zeros due to the
        // exponent
        if(a_bits == b_bits)
            return -151; // one smaller than the lowest possible float diff (-127 - 23)
        else
            return a_exp + (8 - __clz(a_bits ^ b_bits)); 
    }
    else { // If exponents differ, return the larger exponent
        return max(a_exp, b_exp);
    }
}

__device__ __forceinline__ int32_t msb_xor_double(double a, double b) {
    if (signbit(a) != signbit(b)) {
        return 1024; // double has exponent with 1+10 bits
    }
    uint64_t a_bits = (uint64_t)__double_as_longlong(fabs(a));
    uint64_t b_bits = (uint64_t)__double_as_longlong(fabs(b));

    int32_t a_exp = (int32_t)(a_bits >> 52) - 1023;
    int32_t b_exp = (int32_t)(b_bits >> 52) - 1023;

    if (a_exp == b_exp) {
        if(a_bits == b_bits)
            return -1076;
        else
            return a_exp + (11 - __clzll(a_bits ^ b_bits));
    }
    else {
        return max(a_exp, b_exp);
    }
}

template<typename tpos>
__device__ __forceinline__ int32_t msb_xor(tpos a, tpos b) {
    if constexpr (std::is_same_v<tpos, float>) 
        return msb_xor_float(a, b);
    else if constexpr (std::is_same_v<tpos, double>) {
        return msb_xor_double(a, b);
    }
    else if constexpr (std::is_same_v<tpos, int64_t>) {
        uint64_t x = (uint64_t)a ^ (uint64_t)b;
        return 63 - __clzll(x);
    }
    else if constexpr (std::is_same_v<tpos, int32_t>) {
        uint32_t x = (uint32_t)a ^ (uint32_t)b;
        return 31 - __clz(x);
    }
    else {
        // undefined
        return 0;
    }
}

template<int dim, typename tpos>
__device__ __forceinline__ int32_t msb_diff_level(
    const Pos<dim,tpos> &pos1, const Pos<dim,tpos> &pos2
) {
    int msb_dim = 0;
    int msb = msb_xor<tpos>(pos1[0], pos2[0]);
    #pragma unroll
    for(int i=1; i<dim; i++) {
        int new_msb = msb_xor<tpos>(pos1[i], pos2[i]);
        msb_dim = new_msb > msb ? i : msb_dim;
        msb = new_msb > msb ? new_msb : msb;
    }

    return msb*(dim+1) - msb_dim;
}

__device__ __forceinline__ int32_t msb_diff_level_old(const float3 &p1, const float3 &p2) {
    int msb_x = msb_xor_float(p1.x, p2.x);
    int msb_y = msb_xor_float(p1.y, p2.y);
    int msb_z = msb_xor_float(p1.z, p2.z);

    // The level is given by the most significant differing bit
    // but offset according to the dimension
    return max(3*msb_x+3, max(3*msb_y+2, 3*msb_z+1));
}

__device__ __forceinline__ float2 float_common_ext(float a, float b) {
    // Warning: This function cannot be used dimension wise in z-order curves
    //          to get node extends!
    // Finds the center and the extend of the domain where floating point numbers
    // have the same most significant bit as both a and b

    if (signbit(a) != signbit(b)) {
        return {0.f, INFINITY};  // The sign is the highest significant bit
    }
    int32_t a_bits = __float_as_int(fabsf(a));
    int32_t b_bits = __float_as_int(fabsf(b));

    int32_t a_exp = (a_bits >> 23) - 127;
    int32_t b_exp = (b_bits >> 23) - 127;

    float cent, ext;

    if (a_exp == b_exp) {

        int32_t msb = __clz(a_bits ^ b_bits); // leading different bits
        int32_t common = (0xFFFFFFFFu << (32 - msb)) & a_bits; // common bits of a and b
        
        // set the next bit to one to get the center of the common range
        common |= (1u << (32 - msb - 1));

        cent = (signbit(a) ? -__int_as_float(common) : __int_as_float(common));
        ext = ldexpf(1.0f, a_exp + (8 - msb));
    }
    else { // If exponents differ, the larger exponent gives the difference level
        int lvl = max(a_exp, b_exp) + 1;
        cent = ldexpf(signbit(a) ? -0.5f : 0.5f, lvl);
        ext = ldexpf(1.0f, lvl);
    }
    return {cent, ext};
}

// Whether pos1 should appear before pos2 in a z-curve ordering
__device__ __forceinline__ bool z_pos_less3(float3 pos1, float3 pos2)
{
    const bool nan1 = isnan(pos1.x) || isnan(pos1.y) || isnan(pos1.z);
    const bool nan2 = isnan(pos2.x) || isnan(pos2.y) || isnan(pos2.z);
    if (nan1) return false;
    if (nan2) return true;

    int msb_x = msb_xor_float(pos1.x, pos2.x);
    int msb_y = msb_xor_float(pos1.y, pos2.y);
    int msb_z = msb_xor_float(pos1.z, pos2.z);

    int ms_dim = (msb_x >= msb_y && msb_x >= msb_z) ? 0 : ((msb_y >= msb_z) ? 1 : 2);

    if (ms_dim == 0) return pos1.x < pos2.x;
    if (ms_dim == 1) return pos1.y < pos2.y;
    return pos1.z < pos2.z;
}

template <int dim, typename tpos> __device__  __forceinline__ bool has_nan(
    Pos<dim,tpos> pos
) {
    if constexpr (!std::is_floating_point_v<tpos>) {
        return false;
    } else {
        bool any_nan = false;
        #pragma unroll
        for(int i=0; i<dim; i++) {
            any_nan = any_nan | isnan(pos[i]);
        }
        return any_nan;
    }
}

// Whether pos1 should appear before pos2 in a z-order
template <int dim, typename tpos>
__device__ __forceinline__ bool z_pos_less(Pos<dim,tpos> pos1, Pos<dim,tpos> pos2)
{
    if(has_nan<dim,tpos>(pos1)) return false;
    if(has_nan<dim,tpos>(pos2)) return true;

    int msb_dim = 0;
    int msb = msb_xor<tpos>(pos1[0], pos2[0]);
    #pragma unroll
    for(int i=1; i<dim; i++) {
        int new_msb = msb_xor<tpos>(pos1[i], pos2[i]);
        msb_dim = new_msb > msb ? i : msb_dim;
        msb = new_msb > msb ? new_msb : msb;
    }

    #pragma unroll
    for(int i=0; i<dim-1; i++) {
        if(msb_dim == i) return pos1[i] < pos2[i];
    }

    return pos1[dim-1] < pos2[dim-1];
}

__device__ __forceinline__ float round_float_pow2_cent(float x, int level)
{
    // rounds to the next center at resolution 2**level
    int32_t x_bits = __float_as_int(fabsf(x));
    int32_t x_exp = (x_bits >> 23) - 127;

    int32_t new_bits = x_bits;

    if(level >= 128) {
        return 0.f; // level represents sign bit difference -> center = 0
    }
    if (level > x_exp) { // exponent larger -- our bits don't matter
        return ldexpf(signbit(x) ? -0.5f : 0.5f, level);
    }
    else {
        // the rounding is done by zeroing out mantissa bits
        int32_t keep_bits = x_exp - level; // how many mantissa bits to keep
        int32_t mask = 0xFFFFFFFFu << max(23 - keep_bits, 0);
        new_bits = x_bits & mask;
        new_bits = new_bits | (1u << max(22 - keep_bits, 0)); // set next bit to one for center
        return signbit(x) ? -__int_as_float(new_bits) : __int_as_float(new_bits);
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            Node Math                                           */
/* ---------------------------------------------------------------------------------------------- */

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

__device__ __forceinline__ int3 lvl_xyz(const int level) {
    // Converts a node's or leaf's binary level to its level per dimension

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 6000) / 3 - 2000;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2);
    int lz = olvl + (omod >= 1);
    
    return int3{lx, ly, lz};
}

__device__ __forceinline__ float3 LvlToExt(const int level) {
    // Converts a node's or leaf's binary level to its extend per dimension
    int3 l = lvl_xyz(level);
    
    return make_float3(ldexpf(1.0f, l.x), ldexpf(1.0f, l.y), ldexpf(1.0f, l.z));
}

__device__ __forceinline__ float3 LvlToHalfExt(int level) {
    // Same as above, but with half-extends
    int3 l = lvl_xyz(level);
    
    return make_float3(ldexpf(1.0f, l.x-1), ldexpf(1.0f, l.y-1), ldexpf(1.0f, l.z-1));
}

__device__ __forceinline__ NodeWithExt NodeLvlToHalfExt(Node node) {
    NodeWithExt node_ext;
    node_ext.center = node.center;
    node_ext.extent = LvlToHalfExt(node.level);
    return node_ext;
}

__device__ __forceinline__ float3 LvlToCenter(const float3 pos, const int level) {
    // Converts a node's or leaf's binary level to its extend per dimension
    int3 l = lvl_xyz(level);
    
    return make_float3(
        round_float_pow2_cent(pos.x, l.x),
        round_float_pow2_cent(pos.y, l.y),
        round_float_pow2_cent(pos.z, l.z)
    );
}

__device__ __forceinline__ NodeWithExt get_common_node(const float3 p1, const float3 p2) {
    int lvl = msb_diff_level_old(p1, p2);

    NodeWithExt node;
    node.center = LvlToCenter(p1, lvl);
    node.extent = LvlToExt(lvl);
    return node;
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