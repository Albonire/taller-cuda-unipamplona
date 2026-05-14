#include <stdio.h>
#include <stdlib.h>
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

__global__ void reduce_sum_shared(const float *input, float *partial, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;

    if (i < n) {
        sum = input[i];
    }

    if (i + blockDim.x < n) {
        sum += input[i + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

int main(void) {
    size_t size = N * sizeof(float);

    float *h_data = (float *) malloc(size);
    if (!h_data) {
        fprintf(stderr, "Error reservando memoria Host.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void **) &d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);

    float *d_partial = NULL;
    float *h_partial = (float *) malloc(blocks * sizeof(float));

    if (!h_partial) {
        fprintf(stderr, "Error reservando h_partial.\n");
        CUDA_CHECK(cudaFree(d_data));
        free(h_data);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMalloc((void **) &d_partial, blocks * sizeof(float)));

    reduce_sum_shared<<<blocks, threads, threads * sizeof(float)>>>(d_data, d_partial, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    /*
        Segunda etapa en CPU para mantener el ejemplo simple:
        la primera reducción paralela usa memoria compartida en GPU.
    */
    double total = 0.0;
    for (int i = 0; i < blocks; i++) {
        total += h_partial[i];
    }

    printf("Reduccion con memoria compartida\n");
    printf("N = %d\n", N);
    printf("Bloques parciales = %d\n", blocks);
    printf("Resultado esperado = %d\n", N);
    printf("Resultado obtenido = %.0f\n", total);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_partial));

    free(h_data);
    free(h_partial);

    return EXIT_SUCCESS;
}
