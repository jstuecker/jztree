#ifndef COMMON_MATH_H
#define COMMON_MATH_H

/* ---------------------------------------------------------------------------------------------- */
/*                                          Integer Math                                          */
/* ---------------------------------------------------------------------------------------------- */

// Calculates integer division in round-up mode
inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

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

#endif // COMMON_MATH_H