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

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    size_t size = N * sizeof(float);

    float *h_a = (float *) malloc(size);
    float *h_b = (float *) malloc(size);
    float *h_c = (float *) malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Error reservando memoria en Host.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;

    CUDA_CHECK(cudaMalloc((void **) &d_a, size));
    CUDA_CHECK(cudaMalloc((void **) &d_b, size));
    CUDA_CHECK(cudaMalloc((void **) &d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c[i] - 3.0f) > 1e-5f) {
            errors++;
            if (errors < 5) {
                printf("Error en i=%d: %f\n", i, h_c[i]);
            }
        }
    }

    printf("Vector add CUDA\n");
    printf("N = %d\n", N);
    printf("Bloques = %d, hilos por bloque = %d\n", blocks, threads);
    printf("Primeros valores: %.1f %.1f %.1f %.1f %.1f\n",
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);
    printf("Errores: %d\n", errors);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
