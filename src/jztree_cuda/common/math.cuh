#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#include "data.cuh"

#include <cmath>
using std::abs; // be sure we have a floating point and int compatible abs function

/* ---------------------------------------------------------------------------------------------- */
/*                                 Data type specific definitions                                 */
/* ---------------------------------------------------------------------------------------------- */

template <class> inline constexpr bool dependent_false_v = false;

template<typename tvec>
__device__ __forceinline__ tvec invalid_val() {
    if constexpr (std::is_same_v<tvec, float>) {
        return __uint_as_float(0x7fc00000u); // quite NaN
    }
    else if constexpr (std::is_same_v<tvec, double>) {
        return __longlong_as_double(0x7ff8000000000000ULL); // quite NaN
    }
    else if constexpr (std::is_same_v<tvec, int32_t>) {
        return static_cast<int32_t>(0x7fffffff); // max int32
    }
    else if constexpr (std::is_same_v<tvec, int64_t>) {
        return static_cast<int64_t>(0x7fffffffffffffffLL); // max int64
    } 
    else {
        return tvec{};
    }
}

template <typename tvec>
__device__ __forceinline__ int min_node_lvl() {
    if constexpr (std::is_same_v<tvec, float>) {
        return -450;
    } 
    else if constexpr (std::is_same_v<tvec, double>) {
        return -3225;
    } 
    else if constexpr (std::is_same_v<tvec, int32_t>) {
        return 0;
    } 
    else if constexpr (std::is_same_v<tvec, int64_t>) {
        return 0;
    } 
    else {
        static_assert(std::is_same_v<tvec, void>, 
            "min_node_lvl<T>: unsupported type. Add a branch for this T."
        );
    }
}

template <typename tvec>
__host__ __device__ __forceinline__ int max_node_lvl() {
    if constexpr (std::is_same_v<tvec, float>) {
        return 388; // 128*3 + 3 + 1
    } 
    else if constexpr (std::is_same_v<tvec, double>) {
        return 3075; // unsure about this
    } 
    else if constexpr (std::is_same_v<tvec, int32_t>) {
        return 32;
    } 
    else if constexpr (std::is_same_v<tvec, int64_t>) {
        return 64;
    } 
    else {
        static_assert(dependent_false_v<tvec>, "max_node_lvl<T>: unsupported type");
    }
}

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

template<typename tvec>
__forceinline__ __device__ void kahan_add(
    tvec &sum, tvec add, tvec &c
) {
    if constexpr (std::is_floating_point_v<tvec>) {
        tvec y = add - c;
        tvec t = sum + y;
        c = (t - sum) - y;
        sum = t;
    } else {
        // for integer types we simply do a normal add, since they are anyways exact
        sum += add;
    }
}

template<int dim, typename tvec>
__forceinline__ __device__ void kahan_add_vec(
    Vec<dim,tvec> &sum, Vec<dim,tvec> add, Vec<dim,tvec> &c
) {
    #pragma unroll
    for(int i=0; i<dim; i++) {
        kahan_add<tvec>(sum[i], add[i], c[i]);
    }
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
        return 128;  // One higher than highest finite float diff (2**127)
    }
    uint32_t a_bits = (uint32_t)__float_as_int(fabsf(a));
    uint32_t b_bits = (uint32_t)__float_as_int(fabsf(b));

    int32_t a_exp = (int32_t)(a_bits >> 23) - 127;
    int32_t b_exp = (int32_t)(b_bits >> 23) - 127;

    if (a_exp == b_exp) { // If both floats have the same exponent, we need to compare mantissas
        // clz counts bit-zeros from the left. There will be always 8 leading zeros due to the
        // exponent
        // if(a_bits == b_bits)
        //     return -150; // one smaller than the smallest non-zero float 2**-149
        a_exp = max(a_exp, -126); // because of special sub-precision number behaviour
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
    uint64_t a_bits = (uint64_t)__double_as_longlong(abs(a));
    uint64_t b_bits = (uint64_t)__double_as_longlong(abs(b));

    int32_t a_exp = (int32_t)(a_bits >> 52) - 1023;
    int32_t b_exp = (int32_t)(b_bits >> 52) - 1023;

    if (a_exp == b_exp) {
        // if(a_bits == b_bits)
        //     return -1075;
        a_exp = max(a_exp, -1022);
        return a_exp + (11 - __clzll(a_bits ^ b_bits));
    }
    else {
        return max(a_exp, b_exp);
    }
}

template<typename tvec>
__device__ __forceinline__ int32_t msb_xor(tvec a, tvec b) {
    if constexpr (std::is_same_v<tvec, float>) 
        return msb_xor_float(a, b);
    else if constexpr (std::is_same_v<tvec, double>) {
        return msb_xor_double(a, b);
    }
    else if constexpr (std::is_same_v<tvec, int64_t>) {
        uint64_t x = (uint64_t)a ^ (uint64_t)b;
        return 63 - __clzll(x);
    }
    else if constexpr (std::is_same_v<tvec, int32_t>) {
        uint32_t x = (uint32_t)a ^ (uint32_t)b;
        return 31 - __clz(x);
    }
    else {
        // undefined
        return 0;
    }
}

template<int dim, typename tvec>
__device__ __forceinline__ int32_t msb_diff_level(
    const Vec<dim,tvec> &pos1, const Vec<dim,tvec> &pos2
) {
    int msb_dim = 0;
    int msb = msb_xor<tvec>(pos1[0], pos2[0]);
    #pragma unroll
    for(int i=1; i<dim; i++) {
        int new_msb = msb_xor<tvec>(pos1[i], pos2[i]);
        msb_dim = new_msb > msb ? i : msb_dim;
        msb = new_msb > msb ? new_msb : msb;
    }

    return (msb+1)*dim - msb_dim;
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

template <int dim, typename tvec> __device__  __forceinline__ bool has_nan(
    Vec<dim,tvec> pos
) {
    if constexpr (!std::is_floating_point_v<tvec>) {
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
template <int dim, typename tvec>
__device__ __forceinline__ bool z_pos_less(Vec<dim,tvec> pos1, Vec<dim,tvec> pos2)
{
    if(has_nan<dim,tvec>(pos1)) return false;
    if(has_nan<dim,tvec>(pos2)) return true;

    int msb_dim = 0;
    int msb = msb_xor<tvec>(pos1[0], pos2[0]);
    #pragma unroll
    for(int i=1; i<dim; i++) {
        int new_msb = msb_xor<tvec>(pos1[i], pos2[i]);
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
    if(level > x_exp) { // exponent larger -- our bits don't matter
        return ldexpf(signbit(x) ? -0.5f : 0.5f, level);
    }
    // the rounding is done by zeroing out mantissa bits
    int32_t keep_bits = max(x_exp, -126) - level; // how many mantissa bits to keep
    if(keep_bits >= 23) return x; // are exact center already
    int32_t mask = 0xFFFFFFFFu << 23 - keep_bits;
    new_bits = x_bits & mask;
    new_bits = new_bits | (1u << 22 - keep_bits); // set next bit to one for center
    return signbit(x) ? -__int_as_float(new_bits) : __int_as_float(new_bits);
}

__device__ __forceinline__ double round_double_pow2_cent(double x, int level)
{
    int64_t x_bits = __double_as_longlong(abs(x));
    int32_t x_exp = (int32_t)(x_bits >> 52) - 1023;

    int64_t new_bits = x_bits;

    if(level >= 1024) {
        return 0.0;
    }
    if (level > x_exp) {
        return ldexp(signbit(x) ? -0.5 : 0.5, level);
    }

    int32_t keep_bits = max(x_exp,-1022) - level;
    if(keep_bits >= 52) return x; // are exact center already
    int64_t mask = 0xFFFFFFFFFFFFFFFFULL << 52 - keep_bits;
    new_bits = x_bits & mask;
    new_bits = new_bits | (1ULL << 51 - keep_bits);
    return signbit(x) ? -__longlong_as_double(new_bits) : __longlong_as_double(new_bits);
}

template <typename tvec>
__device__ __forceinline__ tvec round_pow2_cent(tvec x, int level) {
    if constexpr (std::is_same_v<tvec, float>) {
        return round_float_pow2_cent(x, level);
    } 
    else if constexpr (std::is_same_v<tvec, double>) {
        return round_double_pow2_cent(x, level);
    } 
    else if constexpr (std::is_same_v<tvec, int32_t> || std::is_same_v<tvec, int64_t>) {
        if (level <= 0) return x;
        
        tvec mask = (static_cast<tvec>(-1) << level);
        tvec center_offset = (static_cast<tvec>(1) << (level - 1));
        
        return (x & mask) | center_offset;
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            Node Math                                           */
/* ---------------------------------------------------------------------------------------------- */

template <typename tvec>
__device__ __forceinline__ tvec wrap_dx(tvec dx, const tvec boxsize) {
    // wraps a coordinate difference into the interval [-boxsize/2, boxsize/2)
    // Note: in principle this code would be slightly more optimal if we decided at compile time
    // whether we are periodic. However, my tests suggest that this would only be 3% or so.
    if(boxsize > static_cast<tvec>(0)) {
        if constexpr (std::is_floating_point_v<tvec>) {
            float bh = static_cast<tvec>(0.5) * boxsize;
            dx = dx < -bh ? dx + boxsize : dx;
            dx = dx >= bh ? dx - boxsize : dx;
        }
        else {
            static_assert(dependent_false_v<tvec>, "wrapping not implemented for integers.");
        }
    }

    return dx;
}

__device__ __forceinline__ float wrap_old(float dx, const float boxsize) {
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

__device__ __forceinline__ float distance_squared_old(const float3 &a, const float3 &b, const float boxsize) {
    float dx = wrap_old(a.x - b.x, boxsize);
    float dy = wrap_old(a.y - b.y, boxsize);
    float dz = wrap_old(a.z - b.z, boxsize);

    return dx * dx + dy * dy + dz * dz;
}

template<int dim, typename tvec>
__device__ __forceinline__ float distance_squared(
    const Vec<dim,tvec> &a, const Vec<dim,tvec> &b, const tvec boxsize
) {
    tvec res = 0.;
    #pragma unroll
    for(int i=0; i<dim; i++) {
        tvec dx = wrap_dx<tvec>(a[i] - b[i], boxsize);
        res += dx*dx;
    }
    return res;
}

template<int dim>
__device__ __forceinline__ Vec<dim,int32_t> lvl_vec(const int level) {
    // Converts a node's or leaf's binary level to its level per dimension

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 2000*dim) / dim - 2000;
    int omod = level - olvl * dim;

    Vec<dim,int32_t> lvec;
    #pragma unroll
    for(int i=0; i<dim; i++) {
        lvec[i] = olvl + (omod >= (dim-i));
    }
    
    return lvec;
}

__device__ __forceinline__ int3 lvl_xyz_old(const int level) {
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

template <typename tvec>
__device__ __forceinline__ tvec pow2(tvec val, int pow) {
    if constexpr (std::is_same_v<tvec, float>)
        return ldexpf(val, pow);
    else if constexpr (std::is_same_v<tvec, double>)
        return ldexp(val, pow);
    else if constexpr (std::is_same_v<tvec, int32_t>)
        return pow >= 0 ? val << pow : val >> -pow;
    else if constexpr (std::is_same_v<tvec, int64_t>)
        return pow >= 0 ? val << pow : val >> -pow;
}

template <int dim, typename tvec>
__device__ __forceinline__ Vec<dim, tvec> LvlToExt(const int level) {
    // Converts a node's or leaf's binary level to its extend per dimension
    Vec<dim,int32_t> l = lvl_vec<dim>(level);
    Vec<dim,tvec> ext;
    
    #pragma unroll
    for(int i=0; i<dim; i++)
        ext[i] = pow2(static_cast<tvec>(1.0), l[i]);
        
    return ext;
}

template <int dim, typename tvec>
__device__ __forceinline__ Vec<dim, tvec> LvlToHalfExt(const int level) {
    // Converts a node's or leaf's binary level to its extend per dimension
    Vec<dim,int32_t> l = lvl_vec<dim>(level);
    Vec<dim,tvec> ext;
    
    #pragma unroll
    for(int i=0; i<dim; i++)
        ext[i] = pow2(static_cast<tvec>(1.0), l[i]-1);
        
    return ext;
}

template <int dim, typename tvec>
__device__ __forceinline__ NodeWithExt<dim,tvec> NodeLvlToHalfExt(Node<dim,tvec> node) {
    return NodeWithExt<dim,tvec>{
        node.center,
        LvlToHalfExt<dim,tvec>(node.level)
    };
}

template<int dim, typename tvec>
__device__ __forceinline__ Vec<dim,tvec> LvlToCenter(const Vec<dim,tvec> pos, const int level) {
    Vec<dim,int32_t> l = lvl_vec<dim>(level);
    
    Vec<dim,tvec> res;
    for(int i=0; i<dim; i++) {
        res[i] = round_pow2_cent<tvec>(pos[i], l[i]);
    }
    return res;
}

template<int dim, typename tvec>
__device__ __forceinline__ NodeWithExt<dim,tvec> get_common_node(
    const Vec<dim,tvec> p1, const Vec<dim,tvec> p2
) {
    int lvl = msb_diff_level<dim,tvec>(p1, p2);

    NodeWithExt<dim,tvec> node;
    node.center = LvlToCenter<dim,tvec>(p1, lvl);
    node.extent = LvlToExt<dim,tvec>(lvl);
    return node;
}


template<int dim, typename tvec>
__device__ __forceinline__ tvec mindist2(
    Vec<dim,tvec> x1,
    Vec<dim,tvec> x2,
    Vec<dim,tvec> width_half,
    tvec boxsize=static_cast<tvec>(0)
) {
    tvec r2 = static_cast<tvec>(0);
    #pragma unroll
    for(int i=0; i<dim; i++) {
        tvec dcent = wrap_dx<tvec>(x1[i]-x2[i], boxsize);
        tvec dx = max(abs(dcent) - width_half[i], static_cast<tvec>(0));
        r2 += dx*dx;
    }
    
    return r2;
}

template<int dim, typename tvec>
__device__ __forceinline__ tvec maxdist2(
    Vec<dim,tvec> x1,
    Vec<dim,tvec> x2,
    Vec<dim,tvec> width_half,
    tvec boxsize=static_cast<tvec>(0)
) {
    tvec r2 = static_cast<tvec>(0);
    #pragma unroll
    for(int i=0; i<dim; i++) {
        tvec dcent = wrap_dx<tvec>(x1[i]-x2[i], boxsize);
        tvec dx = max(abs(dcent) + width_half[i], static_cast<tvec>(0));
        r2 += dx*dx;
    }
    return r2;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Warp and Group helpers                                     */
/* ---------------------------------------------------------------------------------------------- */

template<typename tvec>
__device__ __forceinline__ tvec warp_reduce_sum(tvec v) {
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