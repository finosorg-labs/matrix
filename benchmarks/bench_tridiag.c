/**
 * @file bench_tridiag.c
 * @brief Performance benchmarks for tridiagonal matrix solver
 *
 * Measures Thomas algorithm performance for single and multiple RHS.
 */

#include "bench_framework.h"
#include <tridiag.h>

#include <stdlib.h>
#include <stdio.h>

/* Benchmark data structures */

typedef struct {
    int64_t n;
    double *a, *b, *c, *d, *x;
} tridiag_data_t;

typedef struct {
    int64_t n, nrhs;
    double *a, *b, *c, *D, *X;
} tridiag_multi_data_t;

/* Tridiagonal solve benchmark function */
static void bench_tridiag_solve_fn(void* user_data) {
    tridiag_data_t* d = (tridiag_data_t*)user_data;
    fc_mat_tridiag_solve_f64(d->n, d->a, d->b, d->c, d->d, d->x);
}

/* Tridiagonal solve multi benchmark function */
static void bench_tridiag_solve_multi_fn(void* user_data) {
    tridiag_multi_data_t* d = (tridiag_multi_data_t*)user_data;
    fc_mat_tridiag_solve_multi_f64(d->n, d->nrhs, d->a, d->b, d->c, d->D, d->nrhs, d->X, d->nrhs);
}

/* Entry point for tridiagonal benchmarks */
void bench_tridiag_run(void) {
    printf("\n");
    printf("============================================================\n");
    printf("  Tridiagonal Solver Benchmarks (Thomas algorithm)\n");
    printf("============================================================\n");

    /* Single RHS benchmarks */
    printf("\n");
    printf("Single RHS Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {100, 500, 1000, 5000, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int64_t n = sizes[s];
        tridiag_data_t data;
        data.n = n;
        data.a = (double*)malloc(n * sizeof(double));
        data.b = (double*)malloc(n * sizeof(double));
        data.c = (double*)malloc(n * sizeof(double));
        data.d = (double*)malloc(n * sizeof(double));
        data.x = (double*)malloc(n * sizeof(double));

        /* Initialize tridiagonal matrix (diagonally dominant) */
        for (int64_t i = 0; i < n; i++) {
            data.a[i] = 1.0;
            data.b[i] = 4.0;
            data.c[i] = 1.0;
            data.d[i] = 6.0;
        }

        char name[64];
        snprintf(name, sizeof(name), "Tridiag Solve n=%lld", (long long)n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = 5 * n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_tridiag_solve_fn, &data, &result);

        free(data.a);
        free(data.b);
        free(data.c);
        free(data.d);
        free(data.x);
    }

    /* Multi-RHS benchmarks */
    printf("\n");
    printf("Multiple RHS Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    for (int s = 0; s < 3; s++) {
        int64_t n = (s == 0) ? 1000 : (s == 1) ? 5000 : 10000;
        int64_t nrhs = 10;

        tridiag_multi_data_t data;
        data.n = n;
        data.nrhs = nrhs;
        data.a = (double*)malloc(n * sizeof(double));
        data.b = (double*)malloc(n * sizeof(double));
        data.c = (double*)malloc(n * sizeof(double));
        data.D = (double*)malloc(n * nrhs * sizeof(double));
        data.X = (double*)malloc(n * nrhs * sizeof(double));

        for (int64_t i = 0; i < n; i++) {
            data.a[i] = 1.0;
            data.b[i] = 4.0;
            data.c[i] = 1.0;
            for (int64_t j = 0; j < nrhs; j++) {
                data.D[i * nrhs + j] = 6.0;
            }
        }

        char name[64];
        snprintf(name, sizeof(name), "Tridiag Solve Multi n=%lld nrhs=%lld", (long long)n, (long long)nrhs);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = (3 * n + 2 * n * nrhs) * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_tridiag_solve_multi_fn, &data, &result);

        free(data.a);
        free(data.b);
        free(data.c);
        free(data.D);
        free(data.X);
    }

    printf("\n");
    printf("============================================================\n");
    printf("  Tridiagonal benchmarks complete\n");
    printf("============================================================\n");
}
