#ifndef CUSTOM_JAX_INTERFACE_TESTS_H
#define CUSTOM_JAX_INTERFACE_TESTS_H

template <int p>
__global__ void SimpleArange(
    const int* add,
    int* output,
    int size
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if( i < size )
            output[i] = i + add[i] * p;
    }
}

#endif // CUSTOM_JAX_INTERFACE_TESTS_H