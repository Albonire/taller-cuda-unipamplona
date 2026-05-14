#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

int main(void) {
    size_t size = N * sizeof(int);

    int *a = (int *) malloc(size);
    int *b = (int *) malloc(size);
    int *c = (int *) malloc(size);
    int *r = (int *) malloc(size);

    if (!a || !b || !c || !r) {
        fprintf(stderr, "Error reservando memoria en CPU.\n");
        free(a);
        free(b);
        free(c);
        free(r);
        return EXIT_FAILURE;
    }

    double t0 = now_ms();

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
        c[i] = i * 3;
    }

    for (int i = 0; i < N; i++) {
        r[i] = a[i] + b[i] + c[i];
    }

    long long total = 0;
    for (int i = 0; i < N; i++) {
        total += r[i];
    }

    double t1 = now_ms();

    printf("Version CPU C\n");
    printf("Primeros 5 valores de R[i] = A[i]+B[i]+C[i]:\n");
    for (int i = 0; i < 5; i++) {
        printf("  R[%d] = %d\n", i, r[i]);
    }
    printf("Reduccion suma(R): %lld\n", total);
    printf("Tiempo CPU aproximado: %.6f ms\n", t1 - t0);

    free(a);
    free(b);
    free(c);
    free(r);

    return EXIT_SUCCESS;
}
