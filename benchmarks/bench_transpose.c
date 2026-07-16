/**
 * @file bench_transpose.c
 * @brief Transpose benchmarks
 */

#include "bench_framework.h"
#include <transpose.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


typedef struct {
    int rows, cols;
    double *src, *dst;
} transpose_data_t;

/* Helper: fill array with pseudo-random values in [0, 1) */
static void fill_random(double* arr, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (double)(seed & 0x7fffffff) / 2147483648.0;
    }
}

/* Transpose benchmark function */
static void bench_transpose_fn(void* user_data) {
    transpose_data_t* d = (transpose_data_t*)user_data;
    fc_mat_transpose_f64(d->rows, d->cols,
                         d->src, d->cols, d->dst, d->rows);
}

/* Run Transpose benchmarks */
static void run_transpose_benchmarks(void) {
    printf("\n");
    printf("Transpose Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        transpose_data_t data;
        data.rows = n;
        data.cols = n;
        data.src = (double*)malloc(n * n * sizeof(double));
        data.dst = (double*)malloc(n * n * sizeof(double));

        fill_random(data.src, n * n);

        char name[64];
        snprintf(name, sizeof(name), "Transpose %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = 2 * n * n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_transpose_fn, &data, &result);

        free(data.src);
        free(data.dst);
    }
}

void bench_transpose_run(void) {
    run_transpose_benchmarks();
}
