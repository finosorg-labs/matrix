/**
 * @file bench_solve.c
 * @brief Solve benchmarks
 */

#include "bench_framework.h"
#include <solve.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


typedef struct {
    int n, nrhs;
    double *A, *A_orig, *b, *b_orig;
    int64_t *ipiv;
} solve_data_t;

typedef struct {
    int n;
    double *A, *A_orig;
} inverse_data_t;

/* Helper: fill array with pseudo-random values in [0, 1) */
static void fill_random(double* arr, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (double)(seed & 0x7fffffff) / 2147483648.0;
    }
}

/* bench_solve_fn benchmark function */
static void bench_solve_fn(void* user_data) {
    solve_data_t* d = (solve_data_t*)user_data;
    /* Restore input data (in-place operation) */
    memcpy(d->A, d->A_orig, d->n * d->n * sizeof(double));
    memcpy(d->b, d->b_orig, d->n * sizeof(double));
    fc_mat_solve_linear_f64(d->n, 1, d->A, d->n, d->b, 1);
}

/* bench_inverse_fn benchmark function */
static void bench_inverse_fn(void* user_data) {
    inverse_data_t* d = (inverse_data_t*)user_data;
    /* Restore input data (in-place operation) */
    memcpy(d->A, d->A_orig, d->n * d->n * sizeof(double));
    fc_mat_inverse_f64(d->n, d->A, d->n);
}

/* Run linear solve benchmarks */
static void run_solve_benchmarks(void) {
    printf("\n");
    printf("Linear Solve Benchmarks (Ax=b, double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        solve_data_t data;
        data.n = n;
        data.nrhs = 1;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.A_orig = (double*)malloc(n * n * sizeof(double));
        data.b = (double*)malloc(n * sizeof(double));
        data.b_orig = (double*)malloc(n * sizeof(double));
        data.ipiv = (int64_t*)malloc(n * sizeof(int64_t));

        fill_random(data.A_orig, n * n);
        fill_random(data.b_orig, n);
        memcpy(data.A, data.A_orig, n * n * sizeof(double));
        memcpy(data.b, data.b_orig, n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "Solve %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = (n * n + n) * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_solve_fn, &data, &result);

        free(data.A);
        free(data.A_orig);
        free(data.b);
        free(data.b_orig);
        free(data.ipiv);
    }
}

/* Run matrix inverse benchmarks */
static void run_inverse_benchmarks(void) {
    printf("\n");
    printf("Matrix Inverse Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        inverse_data_t data;
        data.n = n;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.A_orig = (double*)malloc(n * n * sizeof(double));

        fill_random(data.A_orig, n * n);
        memcpy(data.A, data.A_orig, n * n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "Inverse %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = n * n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_inverse_fn, &data, &result);

        free(data.A);
        free(data.A_orig);
    }
}

void bench_solve_run(void) {
    run_solve_benchmarks();
    run_inverse_benchmarks();
}
