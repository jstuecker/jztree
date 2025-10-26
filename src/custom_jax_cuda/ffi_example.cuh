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

template <int p>
__global__ void SetToConstant(
    int* output,
    int value,
    int size
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if( i < size )
            output[i] = value * p;
    }
}

template <int tpar>
void SetToConstantCall(
    cudaStream_t stream,
    int* output,
    int value,
    int size,
    size_t block_size
) {
    dim3 gridDim(div_ceil(size, block_size));
    dim3 blockDim(block_size);

    SetToConstant<tpar><<<gridDim, blockDim, 0, stream>>>(
        output,
        value,
        size
    );
}


#endif // CUSTOM_JAX_INTERFACE_TESTS_H