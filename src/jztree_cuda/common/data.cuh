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

template<int dim, typename tpos>
struct Pos {
  tpos v[dim];

  __host__ __device__ __forceinline__
  static Pos constant(tpos val) {
    Pos p;
    #pragma unroll
    for (int i = 0; i < dim; ++i) p.v[i] = val;
    return p;
  }

  __host__ __device__ __forceinline__
  tpos& operator[](int i) { return v[i]; }

  __host__ __device__ __forceinline__
  const tpos& operator[](int i) const { return v[i]; }

  __host__ __device__ __forceinline__
  tpos* data() { return v; }

  __host__ __device__ __forceinline__
  const tpos* data() const { return v; }
};

template <int dim, typename tpos>
struct PosId {
    Pos<dim, tpos> pos;
    int32_t id;
};

template <int dim, typename tpos>
struct NodeWithExt {
    Pos<dim,tpos> center;
    Pos<dim,tpos> extent;
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