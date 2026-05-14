#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

__global__ void vector_add_large(const float *x, const float *y, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main(void) {
    size_t size = (size_t) N * sizeof(float);

    float *h_x = (float *) malloc(size);
    float *h_y = (float *) malloc(size);
    float *h_z_cpu = (float *) malloc(size);
    float *h_z_gpu = (float *) malloc(size);

    if (!h_x || !h_y || !h_z_cpu || !h_z_gpu) {
        fprintf(stderr, "Error reservando memoria Host.\n");
        free(h_x);
        free(h_y);
        free(h_z_cpu);
        free(h_z_gpu);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.5f;
        h_y[i] = 2.5f;
    }

    double cpu_start = now_ms();

    for (int i = 0; i < N; i++) {
        h_z_cpu[i] = h_x[i] + h_y[i];
    }

    double cpu_end = now_ms();
    double cpu_ms = cpu_end - cpu_start;

    float *d_x = NULL;
    float *d_y = NULL;
    float *d_z = NULL;

    CUDA_CHECK(cudaMalloc((void **) &d_x, size));
    CUDA_CHECK(cudaMalloc((void **) &d_y, size));
    CUDA_CHECK(cudaMalloc((void **) &d_z, size));

    cudaEvent_t start, after_h2d, after_kernel, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_h2d));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(after_h2d));

    vector_add_large<<<blocks, threads>>>(d_x, d_y, d_z, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(after_kernel));

    CUDA_CHECK(cudaMemcpy(h_z_gpu, d_z, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_gpu_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, after_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, after_h2d, after_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, after_kernel, stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));

    int errors = 0;
    float max_error = 0.0f;

    for (int i = 0; i < N; i++) {
        float err = fabsf(h_z_cpu[i] - h_z_gpu[i]);

        if (err > max_error) {
            max_error = err;
        }

        if (err > 1e-5f) {
            errors++;
        }
    }

    printf("Comparacion CPU vs GPU con vector grande\n");
    printf("N = %d floats\n", N);
    printf("CPU tiempo suma: %.3f ms\n", cpu_ms);
    printf("GPU H2D: %.3f ms\n", h2d_ms);
    printf("GPU kernel: %.3f ms\n", kernel_ms);
    printf("GPU D2H: %.3f ms\n", d2h_ms);
    printf("GPU total con transferencias: %.3f ms\n", total_gpu_ms);
    printf("Speedup aproximado CPU/kernel: %.2fx\n", kernel_ms > 0.0f ? cpu_ms / kernel_ms : 0.0f);
    printf("Errores: %d, error maximo: %g\n", errors, max_error);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_h2d));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    free(h_x);
    free(h_y);
    free(h_z_cpu);
    free(h_z_gpu);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
