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

struct __align__(16) PosIdOld {
    float3 pos;
    int32_t id;
};

// template<int dim> struct PosId {
//     float pos[dim];
//     int32_t id;
// };

template<int dim>
struct Pos {
  float v[dim];

  __host__ __device__ __forceinline__
  float& operator[](int i) { return v[i]; }

  __host__ __device__ __forceinline__
  const float& operator[](int i) const { return v[i]; }

  __host__ __device__ __forceinline__
  float* data() { return v; }

  __host__ __device__ __forceinline__
  const float* data() const { return v; }
};

template <int dim=3>
struct PosId {
    Pos<dim> pos;
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