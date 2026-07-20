/**
 * @file bench_decompose.c
 * @brief Decompose benchmarks
 */

#include "bench_framework.h"
#include <decompose.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


typedef struct {
    int n;
    double *A, *A_orig;
    int64_t *ipiv;
} lu_data_t;

typedef struct {
    int m, n;
    double *A, *A_orig, *tau;
} qr_data_t;

typedef struct {
    int n;
    double *A, *A_orig;
} cholesky_data_t;

/* Helper: fill array with pseudo-random values in [0, 1) */
static void fill_random(double* arr, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (double)(seed & 0x7fffffff) / 2147483648.0;
    }
}

/* Helper: create symmetric positive definite matrix */
static void make_spd_matrix(double* A, int n) {
    /* A = B^T * B + n*I ensures SPD */
    double* B = (double*)malloc(n * n * sizeof(double));
    fill_random(B, n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += B[k * n + i] * B[k * n + j];
            }
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;  /* Add diagonal dominance */
    }

    free(B);
}

/* bench_lu_fn benchmark function */
static void bench_lu_fn(void* user_data) {
    lu_data_t* d = (lu_data_t*)user_data;
    /* Restore input data (in-place operation) */
    memcpy(d->A, d->A_orig, d->n * d->n * sizeof(double));
    fc_mat_lu_decompose_f64(d->n, d->A, d->n, d->ipiv);
}

/* bench_qr_fn benchmark function */
static void bench_qr_fn(void* user_data) {
    qr_data_t* d = (qr_data_t*)user_data;
    /* Restore input data (in-place operation) */
    memcpy(d->A, d->A_orig, d->m * d->n * sizeof(double));
    fc_mat_qr_decompose_f64(d->m, d->n, d->A, d->n, d->tau);
}

/* bench_cholesky_fn benchmark function */
static void bench_cholesky_fn(void* user_data) {
    cholesky_data_t* d = (cholesky_data_t*)user_data;
    /* Restore input data (in-place operation) */
    memcpy(d->A, d->A_orig, d->n * d->n * sizeof(double));
    fc_mat_cholesky_decompose_f64(d->n, d->A, d->n);
}

/* Run LU decomposition benchmarks */
static void run_lu_benchmarks(void) {
    printf("\n");
    printf("LU Decomposition Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        lu_data_t data;
        data.n = n;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.A_orig = (double*)malloc(n * n * sizeof(double));
        data.ipiv = (int64_t*)malloc(n * sizeof(int64_t));

        fill_random(data.A_orig, n * n);
        memcpy(data.A, data.A_orig, n * n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "LU %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = n * n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_lu_fn, &data, &result);

        free(data.A);
        free(data.A_orig);
        free(data.ipiv);
    }
}

/* Run QR decomposition benchmarks */
static void run_qr_benchmarks(void) {
    printf("\n");
    printf("QR Decomposition Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        qr_data_t data;
        data.m = n;
        data.n = n;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.A_orig = (double*)malloc(n * n * sizeof(double));
        data.tau = (double*)malloc(n * sizeof(double));

        fill_random(data.A_orig, n * n);
        memcpy(data.A, data.A_orig, n * n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "QR %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = n * n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_qr_fn, &data, &result);

        free(data.A);
        free(data.A_orig);
        free(data.tau);
    }
}

/* Run Cholesky decomposition benchmarks */
static void run_cholesky_benchmarks(void) {
    printf("\n");
    printf("Cholesky Decomposition Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        cholesky_data_t data;
        data.n = n;
        data.A = (double*)malloc(n * n * sizeof(double));
        data.A_orig = (double*)malloc(n * n * sizeof(double));

        make_spd_matrix(data.A_orig, n);
        memcpy(data.A, data.A_orig, n * n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "Cholesky %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = n * n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_cholesky_fn, &data, &result);

        free(data.A);
        free(data.A_orig);
    }
}

/* Benchmark data structures for new functions */
typedef struct {
    int m, n;
    double *A, *tau, *b, *b_orig;
} apply_qt_data_t;

typedef struct {
    int n;
    double *R, *b, *b_orig;
} solve_triangular_data_t;

/* bench_apply_qt_fn benchmark function */
static void bench_apply_qt_fn(void* user_data) {
    apply_qt_data_t* d = (apply_qt_data_t*)user_data;
    /* Restore vector (in-place operation) */
    memcpy(d->b, d->b_orig, d->m * sizeof(double));
    fc_mat_apply_qt_vector_f64(d->m, d->n, d->A, d->m, d->tau, d->b);  /* lda = m */
}

/* bench_solve_triangular_fn benchmark function */
static void bench_solve_triangular_fn(void* user_data) {
    solve_triangular_data_t* d = (solve_triangular_data_t*)user_data;
    /* Restore vector (in-place operation) */
    memcpy(d->b, d->b_orig, d->n * sizeof(double));
    fc_mat_solve_triangular_upper_f64(d->n, d->R, d->n, d->b);
}

/* Run apply Q^T benchmarks */
static void run_apply_qt_benchmarks(void) {
    printf("\n");
    printf("Apply Q^T to Vector Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int m = sizes[s];
        int n = (m >= 64) ? (m >> 1) : m;  /* n = m/2 for large matrices */

        apply_qt_data_t data;
        data.m = m;
        data.n = n;
        data.A = (double*)malloc(m * n * sizeof(double));
        data.tau = (double*)malloc(n * sizeof(double));
        data.b = (double*)malloc(m * sizeof(double));
        data.b_orig = (double*)malloc(m * sizeof(double));

        /* Create QR decomposition first */
        fill_random(data.A, m * n);
        fc_mat_qr_decompose_f64(m, n, data.A, m, data.tau);  /* lda = m (column-major) */

        fill_random(data.b_orig, m);
        memcpy(data.b, data.b_orig, m * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "Apply Q^T %dx%d", m, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        /* Data size includes matrix A (m×n), vector b (m), and tau (n) */
        config.data_size = (m * n + m + n) * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_apply_qt_fn, &data, &result);

        free(data.A);
        free(data.tau);
        free(data.b);
        free(data.b_orig);
    }
}

/* Run triangular solve benchmarks */
static void run_solve_triangular_benchmarks(void) {
    printf("\n");
    printf("Solve Upper Triangular System Benchmarks (double precision)\n");
    printf("------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];

        solve_triangular_data_t data;
        data.n = n;
        data.R = (double*)malloc(n * n * sizeof(double));
        data.b = (double*)malloc(n * sizeof(double));
        data.b_orig = (double*)malloc(n * sizeof(double));

        /* Create upper triangular matrix with non-zero diagonal */
        fill_random(data.R, n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                data.R[i * n + j] = 0.0;  /* Zero lower triangle */
            }
            data.R[i * n + i] = 1.0 + data.R[i * n + i];  /* Ensure non-zero diagonal */
        }

        fill_random(data.b_orig, n);
        memcpy(data.b, data.b_orig, n * sizeof(double));

        char name[64];
        snprintf(name, sizeof(name), "Solve Triangular %dx%d", n, n);

        fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
        config.name = name;
        config.data_size = n * sizeof(double);
        config.min_time_ms = 200.0;
        config.warmup_ms = 50.0;

        fc_bench_result_t result;
        fc_bench_run(&config, bench_solve_triangular_fn, &data, &result);

        free(data.R);
        free(data.b);
        free(data.b_orig);
    }
}

void bench_decompose_run(void) {
    run_lu_benchmarks();
    run_qr_benchmarks();
    run_cholesky_benchmarks();
    run_apply_qt_benchmarks();
    run_solve_triangular_benchmarks();
}
