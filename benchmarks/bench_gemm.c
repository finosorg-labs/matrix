/**
 * @file bench_gemm.c
 * @brief Gemm benchmarks
 */

#include "bench_framework.h"
#include <gemm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


typedef struct {
    int m, n, k;
    double *A, *B, *C;
} gemm_data_t;

/* Helper: fill array with pseudo-random values in [0, 1) */
static void fill_random(double* arr, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (double)(seed & 0x7fffffff) / 2147483648.0;
    }
}

/* GEMM benchmark function */
static void bench_gemm_fn(void* user_data) {
    gemm_data_t* d = (gemm_data_t*)user_data;
    fc_mat_gemm_f64(d->m, d->n, d->k, 1.0,
                    d->A, d->k, d->B, d->n, 0.0, d->C, d->n);
}

/* Run GEMM benchmarks at various sizes */
static void run_gemm_benchmarks(void) {
    printf("\n");
    printf("GEMM Benchmarks (C = A*B, double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        gemm_data_t data;
        data.m = n;
        data.n = n;
        data.k = n;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.B = (double*)malloc(n * n * sizeof(double));
        data.C = (double*)malloc(n * n * sizeof(double));

        fill_random(data.A, n * n);
        fill_random(data.B, n * n);
        memset(data.C, 0, n * n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "GEMM %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        /* GEMM: 2*m*n*k FLOPs per iteration */
        config.flop_count = 2.0 * n * n * n;
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_gemm_fn, &data, &result);

        free(data.A);
        free(data.B);
        free(data.C);
    }
}

void bench_gemm_run(void) {
    run_gemm_benchmarks();
}
