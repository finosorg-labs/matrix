/**
 * @file bench_vector_ops.c
 * @brief Vector Ops benchmarks
 */

#include "bench_framework.h"
#include <vector_ops.h>
#include <stdlib.h>
#include <stdio.h>


typedef struct {
    int n;
    double *x, *y;
} vec_data_t;

/* Helper: fill array with pseudo-random values in [0, 1) */
static void fill_random(double* arr, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (double)(seed & 0x7fffffff) / 2147483648.0;
    }
}

/* VecDot benchmark function */
static void bench_vec_dot_fn(void* user_data) {
    vec_data_t* d = (vec_data_t*)user_data;
    double result;
    fc_vec_dot_f64(d->x, d->y, d->n, &result);
}

/* VecNormL2 benchmark function */
static void bench_vec_norm_l2_fn(void* user_data) {
    vec_data_t* d = (vec_data_t*)user_data;
    double result;
    fc_vec_norm_l2_f64(d->x, d->n, &result);
}

/* Run vector operation benchmarks */
static void run_vector_benchmarks(void) {
    printf("\n");
    printf("Vector Operation Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {256, 1024, 4096, 16384, 65536};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        vec_data_t data;
        data.n = n;
        data.x = (double*)malloc(n * sizeof(double));
        data.y = (double*)malloc(n * sizeof(double));

        fill_random(data.x, n);
        fill_random(data.y, n);

        /* VecDot */
        {
            char name[64];
            snprintf(name, sizeof(name), "VecDot n=%d", n);

            fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
            config.name = name;
            config.data_size = 2 * n * sizeof(double);
            config.min_time_ms = 200.0;
            config.warmup_ms = 50.0;

            fc_bench_result_t result;
            fc_bench_run(&config, bench_vec_dot_fn, &data, &result);
        }

        /* VecNormL2 */
        {
            char name[64];
            snprintf(name, sizeof(name), "VecNormL2 n=%d", n);

            fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
            config.name = name;
            config.data_size = n * sizeof(double);
            config.min_time_ms = 200.0;
            config.warmup_ms = 50.0;

            fc_bench_result_t result;
            fc_bench_run(&config, bench_vec_norm_l2_fn, &data, &result);
        }

        free(data.x);
        free(data.y);
    }
}

void bench_vector_ops_run(void) {
    run_vector_benchmarks();
}
