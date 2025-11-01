#ifndef COMMON_DATA_H
#define COMMON_DATA_H

struct __align__(16) Node {
    float3 center;
    int level;
};

struct NodeWithExt {
    float3 center;
    float3 extent;
};

struct __align__(16) PMass {
    float3 pos;
    float mass;
};

// redundant... get rid of it later!
struct __align__(16) PosMass {
    float x, y, z, mass;
};

struct __align__(16) PosId {
    float3 pos;
    int32_t id;
};

struct __align__(16) ForcePot {
    float3 force;
    float  pot;
};

#endif // COMMON_DATA_H