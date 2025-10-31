#ifndef COMMON_MATH_H
#define COMMON_MATH_H

// A function calculating the ceil of integer division on CPU
inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float3 float3sum(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 float3diff(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float norm2(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

#endif // COMMON_MATH_H