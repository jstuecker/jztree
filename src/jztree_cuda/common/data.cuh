#ifndef COMMON_DATA_H
#define COMMON_DATA_H

struct __align__(16) Node {
    float3 center;
    int level;
};

struct NodeWithExtOld {
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

template<int dim, typename tvec>
struct Vec {
  tvec v[dim];

  __host__ __device__ __forceinline__
  static Vec constant(tvec val) {
    Vec p;
    #pragma unroll
    for (int i = 0; i < dim; ++i) p.v[i] = val;
    return p;
  }

  __host__ __device__ __forceinline__
  tvec& operator[](int i) { return v[i]; }

  __host__ __device__ __forceinline__
  const tvec& operator[](int i) const { return v[i]; }

  __host__ __device__ __forceinline__
  tvec* data() { return v; }

  __host__ __device__ __forceinline__
  const tvec* data() const { return v; }
};

template <int dim, typename tvec>
struct PosId {
    Vec<dim, tvec> pos;
    int32_t id;
};

template <int dim, typename tvec>
struct NodeWithExt {
    Vec<dim,tvec> center;
    Vec<dim,tvec> extent;
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