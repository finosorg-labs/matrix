/**
 * @file bench_gemv.c
 * @brief Gemv benchmarks
 */

#include "bench_framework.h"
#include <gemv.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


typedef struct {
    int m, n;
    double *A, *x, *y;
} gemv_data_t;

/* Helper: fill array with pseudo-random values in [0, 1) */
static void fill_random(double* arr, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (double)(seed & 0x7fffffff) / 2147483648.0;
    }
}

/* GEMV benchmark function */
static void bench_gemv_fn(void* user_data) {
    gemv_data_t* d = (gemv_data_t*)user_data;
    fc_mat_gemv_f64(d->m, d->n, 1.0,
                    d->A, d->n, d->x, 0.0, d->y);
}

/* Run GEMV benchmarks */
static void run_gemv_benchmarks(void) {
    printf("\n");
    printf("GEMV Benchmarks (y = A*x, double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        gemv_data_t data;
        data.m = n;
        data.n = n;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.x = (double*)malloc(n * sizeof(double));
        data.y = (double*)malloc(n * sizeof(double));

        fill_random(data.A, n * n);
        fill_random(data.x, n);
        memset(data.y, 0, n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "GEMV %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = (n * n + 2 * n) * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_gemv_fn, &data, &result);

        free(data.A);
        free(data.x);
        free(data.y);
    }
}

void bench_gemv_run(void) {
    run_gemv_benchmarks();
}
