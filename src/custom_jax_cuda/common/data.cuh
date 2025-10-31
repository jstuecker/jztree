#ifndef COMMON_DATA_H
#define COMMON_DATA_H

struct __align__(16) NodeInfo {
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

struct __align__(16) ForceAndPot {
    float3 force;
    float  pot;
};

#endif // COMMON_DATA_H