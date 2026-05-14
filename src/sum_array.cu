#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

__global__ void SumVector(int *a, int *b, int *c, int *r, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        r[i] = a[i] + b[i] + c[i];
    }
}

int main(void) {
    size_t size = N * sizeof(int);

    int *h_a = (int *) malloc(size);
    int *h_b = (int *) malloc(size);
    int *h_c = (int *) malloc(size);
    int *h_r = (int *) malloc(size);

    if (!h_a || !h_b || !h_c || !h_r) {
        fprintf(stderr, "Error reservando memoria en Host.\n");
        free(h_a);
        free(h_b);
        free(h_c);
        free(h_r);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
        h_c[i] = i * 3;
    }

    int *d_a = NULL;
    int *d_b = NULL;
    int *d_c = NULL;
    int *d_r = NULL;

    CUDA_CHECK(cudaMalloc((void **) &d_a, size));
    CUDA_CHECK(cudaMalloc((void **) &d_b, size));
    CUDA_CHECK(cudaMalloc((void **) &d_c, size));
    CUDA_CHECK(cudaMalloc((void **) &d_r, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice));

    /*
        Según el apunte del taller:
        SumVector<<<1, 1000>>>

        N = 1000 hilos en un solo bloque.
        Esto es válido porque muchas GPU permiten hasta 1024 hilos por bloque.
    */
    SumVector<<<1, N>>>(d_a, d_b, d_c, d_r, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost));

    long long total = 0;
    for (int i = 0; i < N; i++) {
        total += h_r[i];
    }

    printf("Version CUDA GPU\n");
    printf("Primeros 5 valores de R[i] = A[i]+B[i]+C[i]:\n");
    for (int i = 0; i < 5; i++) {
        printf("  R[%d] = %d\n", i, h_r[i]);
    }
    printf("Reduccion suma(R): %lld\n", total);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_r));

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_r);

    return EXIT_SUCCESS;
}
