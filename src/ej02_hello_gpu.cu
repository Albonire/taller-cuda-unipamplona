#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

__global__ void hello_from_gpu(void) {
    printf("Hola desde GPU -> bloque %d, hilo %d\n", blockIdx.x, threadIdx.x);
}

int main(void) {
    printf("Hola desde CPU.\n");

    hello_from_gpu<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
