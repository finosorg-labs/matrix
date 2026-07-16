/**
 * @file test_precision.c
 * @brief Numerical precision validation for matrix operations
 *
 * Validates that SIMD-optimized implementations produce results within
 * acceptable error bounds compared to a high-precision reference.
 * Uses Kahan summation as the reference for dot products and reductions.
 */

#include "test_framework.h"
#include <matrix.h>
#include <simd_detect.h>

#include <math.h>
#include <stdlib.h>
#include <float.h>

/* Precision thresholds */
#define PREC_EPS_TIGHT  1e-14   /* Near machine epsilon for double */
#define PREC_EPS_NORMAL 1e-12   /* Acceptable for accumulated operations */
#define PREC_EPS_GEMM   1e-9    /* GEMM with large matrices */

/* Kahan summation for high-precision reference */
static double kahan_dot(const double* x, const double* y, int n) {
    double sum = 0.0;
    double c = 0.0;
    for (int i = 0; i < n; i++) {
        double t = x[i] * y[i] - c;
        double s = sum + t;
        c = (s - sum) - t;
        sum = s;
    }
    return sum;
}

/* Reference GEMM using Kahan summation per element */
static void reference_gemm(int m, int n, int k,
                           double alpha, const double* A, int lda,
                           const double* B, int ldb,
                           double beta, double* C, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            double c = 0.0;
            for (int p = 0; p < k; p++) {
                double t = A[i * lda + p] * B[p * ldb + j] - c;
                double s = sum + t;
                c = (s - sum) - t;
                sum = s;
            }
            if (beta == 0.0) {
                C[i * ldc + j] = alpha * sum;
            } else {
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            }
        }
    }
}

/* Reference GEMV using Kahan summation */
static void reference_gemv(int m, int n,
                           double alpha, const double* A, int lda,
                           const double* x,
                           double beta, double* y) {
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        double c = 0.0;
        for (int j = 0; j < n; j++) {
            double t = A[i * lda + j] * x[j] - c;
            double s = sum + t;
            c = (s - sum) - t;
            sum = s;
        }
        if (beta == 0.0) {
            y[i] = alpha * sum;
        } else {
            y[i] = alpha * sum + beta * y[i];
        }
    }
}

/* Fill with values that stress floating-point precision */
static void fill_precision_stress(double* arr, int n, unsigned int seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        /* Mix large and small values to stress precision */
        double base = (double)((int)(seed >> 16) % 2001 - 1000) / 100.0;
        seed = seed * 1103515245 + 12345;
        double scale = pow(10.0, (double)((int)(seed >> 16) % 7 - 3));
        arr[i] = base * scale;
    }
}

/* Compute relative error */
static double relative_error(double computed, double reference) {
    if (reference == 0.0) {
        return fabs(computed);
    }
    return fabs(computed - reference) / fabs(reference);
}

/* Compute max relative error across arrays */
static double max_relative_error(const double* computed, const double* reference, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = relative_error(computed[i], reference[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* VecDot precision test */
TEST(test_precision_vec_dot) {
    int sizes[] = {7, 16, 63, 128, 255, 1024, 4096, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* x = (double*)malloc(n * sizeof(double));
        double* y = (double*)malloc(n * sizeof(double));

        fill_precision_stress(x, n, 42 + s);
        fill_precision_stress(y, n, 137 + s);

        double ref = kahan_dot(x, y, n);
        double result;
        int status = fc_vec_dot_f64(x, y, n, &result);
        ASSERT_EQ(status, 0);

        double err = relative_error(result, ref);
        FC_TEST_ASSERT_MSG(err < PREC_EPS_NORMAL,
                    "VecDot n=%d: relative error %.2e exceeds threshold %.2e",
                    n, err, PREC_EPS_NORMAL);

        free(x);
        free(y);
    }
}

/* VecNormL2 precision test */
TEST(test_precision_vec_norm_l2) {
    int sizes[] = {7, 64, 255, 1024, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* x = (double*)malloc(n * sizeof(double));

        fill_precision_stress(x, n, 99 + s);

        /* Reference: Kahan dot of x with itself, then sqrt */
        double ref_sq = kahan_dot(x, x, n);
        double ref = sqrt(ref_sq);

        double result;
        int status = fc_vec_norm_l2_f64(x, n, &result);
        ASSERT_EQ(status, 0);

        double err = relative_error(result, ref);
        FC_TEST_ASSERT_MSG(err < PREC_EPS_NORMAL,
                    "VecNormL2 n=%d: relative error %.2e exceeds threshold %.2e",
                    n, err, PREC_EPS_NORMAL);

        free(x);
    }
}

/* GEMM precision test */
TEST(test_precision_gemm_small) {
    int sizes[] = {4, 8, 16, 32};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* A = (double*)malloc(n * n * sizeof(double));
        double* B = (double*)malloc(n * n * sizeof(double));
        double* C = (double*)malloc(n * n * sizeof(double));
        double* C_ref = (double*)malloc(n * n * sizeof(double));

        fill_precision_stress(A, n * n, 11 + s);
        fill_precision_stress(B, n * n, 23 + s);
        memset(C, 0, n * n * sizeof(double));
        memset(C_ref, 0, n * n * sizeof(double));

        reference_gemm(n, n, n, 1.0, A, n, B, n, 0.0, C_ref, n);
        int status = fc_mat_gemm_f64(n, n, n, 1.0, A, n, B, n, 0.0, C, n);
        ASSERT_EQ(status, 0);

        double err = max_relative_error(C, C_ref, n * n);
        FC_TEST_ASSERT_MSG(err < PREC_EPS_NORMAL,
                    "GEMM %dx%d: max relative error %.2e exceeds threshold %.2e",
                    n, n, err, PREC_EPS_NORMAL);

        free(A);
        free(B);
        free(C);
        free(C_ref);
    }
}

/* GEMM precision test with larger matrices */
TEST(test_precision_gemm_large) {
    int sizes[] = {64, 128, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* A = (double*)malloc(n * n * sizeof(double));
        double* B = (double*)malloc(n * n * sizeof(double));
        double* C = (double*)malloc(n * n * sizeof(double));
        double* C_ref = (double*)malloc(n * n * sizeof(double));

        fill_precision_stress(A, n * n, 31 + s);
        fill_precision_stress(B, n * n, 47 + s);
        memset(C, 0, n * n * sizeof(double));
        memset(C_ref, 0, n * n * sizeof(double));

        reference_gemm(n, n, n, 1.0, A, n, B, n, 0.0, C_ref, n);
        int status = fc_mat_gemm_f64(n, n, n, 1.0, A, n, B, n, 0.0, C, n);
        ASSERT_EQ(status, 0);

        double err = max_relative_error(C, C_ref, n * n);
        FC_TEST_ASSERT_MSG(err < PREC_EPS_GEMM,
                    "GEMM %dx%d: max relative error %.2e exceeds threshold %.2e",
                    n, n, err, PREC_EPS_GEMM);

        free(A);
        free(B);
        free(C);
        free(C_ref);
    }
}

/* GEMM precision with alpha/beta */
TEST(test_precision_gemm_alpha_beta) {
    int n = 64;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    double* C_ref = (double*)malloc(n * n * sizeof(double));

    fill_precision_stress(A, n * n, 53);
    fill_precision_stress(B, n * n, 67);

    /* Initialize C with non-zero values */
    fill_precision_stress(C, n * n, 79);
    memcpy(C_ref, C, n * n * sizeof(double));

    double alpha = 2.5;
    double beta = 0.3;

    reference_gemm(n, n, n, alpha, A, n, B, n, beta, C_ref, n);
    int status = fc_mat_gemm_f64(n, n, n, alpha, A, n, B, n, beta, C, n);
    ASSERT_EQ(status, 0);

    double err = max_relative_error(C, C_ref, n * n);
    FC_TEST_ASSERT_MSG(err < PREC_EPS_GEMM,
                "GEMM alpha/beta: max relative error %.2e exceeds threshold %.2e",
                err, PREC_EPS_GEMM);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

/* GEMV precision test */
TEST(test_precision_gemv) {
    int sizes[] = {7, 16, 64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* A = (double*)malloc(n * n * sizeof(double));
        double* x = (double*)malloc(n * sizeof(double));
        double* y = (double*)malloc(n * sizeof(double));
        double* y_ref = (double*)malloc(n * sizeof(double));

        fill_precision_stress(A, n * n, 41 + s);
        fill_precision_stress(x, n, 59 + s);
        memset(y, 0, n * sizeof(double));
        memset(y_ref, 0, n * sizeof(double));

        reference_gemv(n, n, 1.0, A, n, x, 0.0, y_ref);
        int status = fc_mat_gemv_f64(n, n, 1.0, A, n, x, 0.0, y);
        ASSERT_EQ(status, 0);

        double err = max_relative_error(y, y_ref, n);
        FC_TEST_ASSERT_MSG(err < PREC_EPS_NORMAL,
                    "GEMV %dx%d: max relative error %.2e exceeds threshold %.2e",
                    n, n, err, PREC_EPS_NORMAL);

        free(A);
        free(x);
        free(y);
        free(y_ref);
    }
}

/* GEMV precision with alpha/beta */
TEST(test_precision_gemv_alpha_beta) {
    int n = 128;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* y_ref = (double*)malloc(n * sizeof(double));

    fill_precision_stress(A, n * n, 71);
    fill_precision_stress(x, n, 83);
    fill_precision_stress(y, n, 97);
    memcpy(y_ref, y, n * sizeof(double));

    double alpha = 1.5;
    double beta = -0.7;

    reference_gemv(n, n, alpha, A, n, x, beta, y_ref);
    int status = fc_mat_gemv_f64(n, n, alpha, A, n, x, beta, y);
    ASSERT_EQ(status, 0);

    double err = max_relative_error(y, y_ref, n);
    FC_TEST_ASSERT_MSG(err < PREC_EPS_NORMAL,
                "GEMV alpha/beta: max relative error %.2e exceeds threshold %.2e",
                err, PREC_EPS_NORMAL);

    free(A);
    free(x);
    free(y);
    free(y_ref);
}

/* Transpose precision (should be exact) */
TEST(test_precision_transpose) {
    int sizes[] = {7, 16, 63, 128, 255};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* src = (double*)malloc(n * n * sizeof(double));
        double* dst = (double*)malloc(n * n * sizeof(double));

        fill_precision_stress(src, n * n, 101 + s);

        int status = fc_mat_transpose_f64(n, n, src, n, dst, n);
        ASSERT_EQ(status, 0);

        /* Transpose should be bit-exact */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                FC_TEST_ASSERT_DOUBLE_EQ_EXACT(dst[j * n + i], src[i * n + j]);
            }
        }

        free(src);
        free(dst);
    }
}

/* VecDistanceL2 precision */
TEST(test_precision_vec_distance_l2) {
    int sizes[] = {7, 64, 1024, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double* x = (double*)malloc(n * sizeof(double));
        double* y = (double*)malloc(n * sizeof(double));
        double* diff = (double*)malloc(n * sizeof(double));

        fill_precision_stress(x, n, 111 + s);
        fill_precision_stress(y, n, 127 + s);

        for (int i = 0; i < n; i++) {
            diff[i] = x[i] - y[i];
        }
        double ref = sqrt(kahan_dot(diff, diff, n));

        double result;
        int status = fc_vec_distance_l2_f64(x, y, n, &result);
        ASSERT_EQ(status, 0);

        double err = relative_error(result, ref);
        FC_TEST_ASSERT_MSG(err < PREC_EPS_NORMAL,
                    "VecDistanceL2 n=%d: relative error %.2e exceeds threshold %.2e",
                    n, err, PREC_EPS_NORMAL);

        free(x);
        free(y);
        free(diff);
    }
}

void register_precision_tests(void) {
    RUN_TEST(test_precision_vec_dot);
    RUN_TEST(test_precision_vec_norm_l2);
    RUN_TEST(test_precision_gemm_small);
    RUN_TEST(test_precision_gemm_large);
    RUN_TEST(test_precision_gemm_alpha_beta);
    RUN_TEST(test_precision_gemv);
    RUN_TEST(test_precision_gemv_alpha_beta);
    RUN_TEST(test_precision_transpose);
    RUN_TEST(test_precision_vec_distance_l2);
}
