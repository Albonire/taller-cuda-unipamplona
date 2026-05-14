#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 20)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main(void) {
    size_t size = N * sizeof(float);

    float *x = NULL;
    float *y = NULL;

    CUDA_CHECK(cudaMallocManaged((void **) &x, size));
    CUDA_CHECK(cudaMallocManaged((void **) &y, size));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    saxpy<<<blocks, threads>>>(N, 2.0f, x, y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int errors = 0;

    for (int i = 0; i < N; i++) {
        if (fabsf(y[i] - 4.0f) > 1e-5f) {
            errors++;
        }
    }

    printf("Memoria unificada CUDA\n");
    printf("Operacion: y = 2*x + y\n");
    printf("Resultado esperado por elemento: 4.0\n");
    printf("Primeros valores: %.1f %.1f %.1f %.1f %.1f\n",
           y[0], y[1], y[2], y[3], y[4]);
    printf("Errores: %d\n", errors);

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
