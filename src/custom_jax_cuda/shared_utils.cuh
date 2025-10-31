#define INFTY  INFINITY //__int_as_float(0x7f800000)

// Utility: next power-of-two >= n (cap at MAX_THREADS)
// Smallest power of two >= x (32-bit)
__device__ __forceinline__ unsigned int next_pow2_u32(unsigned int x) {
    if (x == 0) return 1u;                 // define as 1 for x=0
    if (x > 0x80000000u) return 0u;        // overflow (no 33rd bit in u32)
    return 1u << (32 - __clz(x - 1));      // __clz: count leading zeros
}

struct KV {
    float k;   // key (float32)
    int32_t v; // value (int32)
};

// Compare-and-swap: ascending on key; swap value with it
__device__ __forceinline__
void cas(float &ak, int32_t &av, float &bk, int32_t &bv, bool dir /*true=>ascending*/) {
    // Treat NaN as +inf so they go to the end in ascending
    float ak_eff = isnan(ak) ? INFTY : ak;
    float bk_eff = isnan(bk) ? INFTY : bk;

    bool greater = (ak_eff > bk_eff);
    if (greater == dir) {
        float tk = ak; ak = bk; bk = tk;
        int32_t tv = av; av = bv; bv = tv;
    }
}

template <typename T>
void comparator(std::vector<T> &a, std::size_t i, std::size_t j) {
  if (i < j && j < a.size() && a[j] < a[i])
    std::swap(a[i], a[j]);
}

template <typename T> void impBitonicSort(std::vector<T> &a) {
  // Iterate k as if the array size were rounded up to the nearest power of two.
  for (std::size_t k = 2; (k >> 1) < a.size(); k <<= 1) {
    for (std::size_t i = 0; i < a.size(); i++)
      comparator(a, i, i ^ (k - 1));
    for (std::size_t j = k >> 1; 0 < j; j >>= 1)
      for (std::size_t i = 0; i < a.size(); i++)
        comparator(a, i, i ^ j);
  }
}

int main() {
  for (int n = 2; n <= 8; n++) {
    std::vector<int> unsorted(n);
    std::iota(unsorted.begin(), unsorted.end(), 0);
    do {
      auto sorted = unsorted;
      impBitonicSort(sorted);
      if (!std::is_sorted(sorted.begin(), sorted.end())) {
        for (int i : unsorted)
          std::printf(" %d", i);
        std::printf("\n");
        return 1;
      }
    } while (std::next_permutation(unsorted.begin(), unsorted.end()));
  }
}


__device__ __forceinline__ 
void compare_and_swap(float *val, int32_t *idx, int i, int j, int len) {
    if(i < j && j < len && val[i] > val[j]) {
        float t = val[i];
        val[i] = val[j];
        val[j] = t;

        int32_t ti = idx[i];
        idx[i] = idx[j];
        idx[j] = ti;
    }
}

// Bitonic sort network -- works in-place and can use non-power-of-two lengths
__device__ void bitonic_sort(float* skeys, int32_t* svals, int len) {
    // see https://stackoverflow.com/questions/73147204
    int nPow2 = next_pow2_u32(len);

    for (int k = 2; k <= nPow2; k <<= 1) {
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            compare_and_swap(skeys, svals, i, i ^ (k - 1), len);
        }
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < len; i += blockDim.x) {
                compare_and_swap(skeys, svals, i, i ^ j, len);
            }
        }
    }
    __syncthreads();
}

// -------- Kernel: segmented sort (keys + values) --------
__global__ void segmented_bitonic_sort_kv(
    float* __restrict__ keys,
    int32_t* __restrict__ values,
    const int32_t* __restrict__ offsets,
    int32_t num_segments,
    int32_t smem_size // the maximum segment length where we do the sort in shared memory
)
{
    int seg = blockIdx.x;
    if (seg >= num_segments) return;

    int32_t start = offsets[seg];
    int32_t end   = offsets[seg + 1];
    int32_t len   = max(end - start, 0);
    if (len <= 1) return;

    if(len <= smem_size) { // We can do the sort in shared memory
        extern __shared__ unsigned char smem[];
        float*   skeys = reinterpret_cast<float*>(smem);
        int32_t* svals = reinterpret_cast<int32_t*>(skeys + smem_size);

        // Load to shared memory
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            skeys[i] = keys[start + i];
            svals[i] = values[start + i];
        }
        __syncthreads();

        // Sort in shared memory
        bitonic_sort(skeys, svals, len);

        // Store back
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            keys[start + i]   = skeys[i];
            values[start + i] = svals[i];
        }
    }
    else { // Fall back to global memory sort (inefficient, but rare)
        bitonic_sort(&keys[start], &values[start], len);
    }
}
