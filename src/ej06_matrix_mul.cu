#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 16

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

__global__ void matrix_mul(const float *A, const float *B, float *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float acc = 0.0f;

        for (int k = 0; k < n; k++) {
            acc += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = acc;
    }
}

int main(void) {
    int total = N * N;
    size_t size = total * sizeof(float);

    float *h_A = (float *) malloc(size);
    float *h_B = (float *) malloc(size);
    float *h_C = (float *) malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error reservando memoria en Host.\n");
        return EXIT_FAILURE;
    }

    /*
        A contiene valores simples.
        B es la matriz identidad.
        Entonces C = A * I = A.
    */
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            h_A[r * N + c] = (float)(r + c);
            h_B[r * N + c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **) &d_A, size));
    CUDA_CHECK(cudaMalloc((void **) &d_B, size));
    CUDA_CHECK(cudaMalloc((void **) &d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    matrix_mul<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < total; i++) {
        if (fabsf(h_C[i] - h_A[i]) > 1e-5f) {
            errors++;
        }
    }

    printf("Multiplicacion de matrices CUDA %dx%d\n", N, N);
    printf("Como B es identidad, C debe ser igual a A.\n");
    printf("Errores: %d\n", errors);

    printf("Submatriz 4x4 de C:\n");
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            printf("%6.1f", h_C[r * N + c]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
