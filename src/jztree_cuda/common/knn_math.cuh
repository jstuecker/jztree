#ifndef COMMON_KNN_MATH_CUH
#define COMMON_KNN_MATH_CUH

#include "data.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                            Node math                                           */
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

__device__ __forceinline__ float3 LvlToHalfExt(int level) {
    // Same as above, but with half-extends
    int olvl = (level + 3000) / 3 - 1000;
    int omod = level - olvl * 3;
    int lx = olvl;
    int ly = olvl + (omod >= 2);
    int lz = olvl + (omod >= 1);
    
    return make_float3(ldexpf(1.0f, lx-1), ldexpf(1.0f, ly-1), ldexpf(1.0f, lz-1));
}

__device__ __forceinline__ NodeWithExt NodeLvlToHalfExt(Node node) {
    NodeWithExt node_ext;
    node_ext.center = node.center;
    node_ext.extent = LvlToHalfExt(node.level);
    return node_ext;
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



#endif