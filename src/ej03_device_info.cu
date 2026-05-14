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

int main(void) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    printf("Cantidad de dispositivos CUDA: %d\n", count);

    for (int d = 0; d < count; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));

        printf("\nDispositivo %d: %s\n", d, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memoria global: %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocesadores SM: %d\n", prop.multiProcessorCount);
        printf("  Max hilos por bloque: %d\n", prop.maxThreadsPerBlock);
        printf("  Max dimensiones bloque: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max dimensiones grid: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Memoria compartida por bloque: %zu bytes\n",
               prop.sharedMemPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
    }

    return EXIT_SUCCESS;
}
