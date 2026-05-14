#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ROWS 8
#define COLS 8

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

__global__ void matrix_add(const int *A, const int *B, int *C, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int n = ROWS * COLS;
    size_t size = n * sizeof(int);

    int *h_A = (int *) malloc(size);
    int *h_B = (int *) malloc(size);
    int *h_C = (int *) malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error reservando memoria en Host.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = 100 + i;
    }

    int *d_A = NULL;
    int *d_B = NULL;
    int *d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **) &d_A, size));
    CUDA_CHECK(cudaMalloc((void **) &d_B, size));
    CUDA_CHECK(cudaMalloc((void **) &d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((COLS + block.x - 1) / block.x,
              (ROWS + block.y - 1) / block.y);

    matrix_add<<<grid, block>>>(d_A, d_B, d_C, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    printf("Suma de matrices CUDA %dx%d\n", ROWS, COLS);
    printf("Submatriz 4x4 del resultado:\n");

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            printf("%5d", h_C[r * COLS + c]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
