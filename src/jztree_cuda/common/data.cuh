#ifndef COMMON_DATA_H
#define COMMON_DATA_H

/* ---------------------------------------------------------------------------------------------- */
/*                                   Vector class and operators                                   */
/* ---------------------------------------------------------------------------------------------- */

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

    __host__ __device__ __forceinline__ tvec& operator[](int i) { return v[i]; }
    __host__ __device__ __forceinline__ const tvec& operator[](int i) const { return v[i]; }

    __host__ __device__ __forceinline__ tvec* data() { return v; }
    __host__ __device__ __forceinline__ const tvec* data() const { return v; }

    // ---- compound ops (often useful and enables +,-,* to be implemented via these) ----
    __host__ __device__ __forceinline__
    Vec& operator+=(const Vec& rhs) {
        #pragma unroll
        for (int i = 0; i < dim; ++i) v[i] += rhs.v[i];
        return *this;
    }

    __host__ __device__ __forceinline__
    Vec& operator-=(const Vec& rhs) {
        #pragma unroll
        for (int i = 0; i < dim; ++i) v[i] -= rhs.v[i];
        return *this;
    }

    __host__ __device__ __forceinline__
    Vec& operator*=(tvec s) {
        #pragma unroll
        for (int i = 0; i < dim; ++i) v[i] *= s;
        return *this;
    }

    // ---- dot product as a member function ----
    __host__ __device__ __forceinline__
    tvec dot(const Vec& rhs) const {
        tvec sum = tvec(0);
        #pragma unroll
        for (int i = 0; i < dim; ++i) sum += v[i] * rhs.v[i];
        return sum;
    }
};

// ---- binary + and - ----
template<int dim, typename tvec>
__host__ __device__ __forceinline__
Vec<dim, tvec> operator+(Vec<dim, tvec> a, const Vec<dim, tvec>& b) {
    a += b;
    return a;
}

template<int dim, typename tvec>
__host__ __device__ __forceinline__
Vec<dim, tvec> operator-(Vec<dim, tvec> a, const Vec<dim, tvec>& b) {
    a -= b;
    return a;
}

// ---- scalar multiply (both orders) ----
template<int dim, typename tvec>
__host__ __device__ __forceinline__
Vec<dim, tvec> operator*(Vec<dim, tvec> a, tvec s) {
    a *= s;
    return a;
}

template<int dim, typename tvec>
__host__ __device__ __forceinline__
Vec<dim, tvec> operator*(tvec s, Vec<dim, tvec> a) {
    a *= s;
    return a;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Derived classes                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
struct Node {
    Vec<dim,tvec> center;
    int level;
};

struct __align__(16) NodeOld {
    float3 center;
    int level;
};

template<int dim, typename tvec>
struct NodeWithExt {
    Vec<dim,tvec> center;
    Vec<dim,tvec> extent;
};

struct NodeWithExtOld {
    float3 center;
    float3 extent;
};

template<int dim, typename tvec>
struct PosMass {
    Vec<dim,tvec> pos;
    tvec mass;
};

struct __align__(16) PosIdOld {
    float3 pos;
    int32_t id;
};

// template<int dim> struct PosId {
//     float pos[dim];
//     int32_t id;
// };

template <int dim, typename tvec>
struct PosId {
    Vec<dim, tvec> pos;
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