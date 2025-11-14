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

struct __align__(16) PosMass {
    union {
        struct {
            union {
                struct {
                    float x, y, z;
                };
                float3 pos;
            };
            float mass;
        };
        float4 f4;
    };
};

struct __align__(16) PosId {
    float3 pos;
    int32_t id;
};

struct __align__(16) ForcePot {
    union {
        struct {
            float3 force;
            float  pot;
        };

        float4 f4;
    };
};

#endif // COMMON_DATA_H