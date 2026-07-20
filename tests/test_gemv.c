/**
 * @file test_matrix.c
 * @brief Unit tests for matrix operations
 *
 * Tests vector operations and GEMM
 */

#ifndef FC_ENABLE_INTERNAL_TESTS
#define FC_ENABLE_INTERNAL_TESTS
#endif

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "test_framework.h"
#include <matrix.h>
#include <error.h>
#include <simd_detect.h>
#include <matrix_internal.h>

/* Test tolerance for floating-point comparisons */
#define TEST_EPSILON 1e-12
#define TEST_EPSILON_RELAXED 1e-10

/*
 * Helper functions
*/

static int double_equals(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

/* Generate random double in range [min, max] */
static double rand_double(double min, double max) {
    double scale = rand() / (double)RAND_MAX;
    return min + scale * (max - min);
}

/* Fill matrix with random values */
static void fill_random_matrix(double* mat, int rows, int cols, int ld,
                                double min, double max) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = rand_double(min, max);
        }
    }
}

/* Fill vector with random values */
static void fill_random_vector(double* vec, int n, double min, double max) {
    for (int i = 0; i < n; i++) {
        vec[i] = rand_double(min, max);
    }
}

/*
 * Vector operations tests
*/


TEST(test_gemv_alpha_beta) {
    double A[] = {
        1.0, 2.0,
        3.0, 4.0
    };
    double x[] = {1.0, 2.0};
    double y[] = {1.0, 1.0};

    int status = fc_mat_gemv_f64(2, 2, 2.0, A, 2, x, 3.0, y);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(y[0], 2.0 * 5.0 + 3.0 * 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 2.0 * 11.0 + 3.0 * 1.0, TEST_EPSILON));
}

TEST(test_gemv_alpha_zero) {
    /* Test alpha = 0 path */
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double x[] = {1.0, 1.0};
    double y[] = {5.0, 6.0};

    int status = fc_mat_gemv_f64(2, 2, 0.0, A, 2, x, 2.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 10.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 12.0, TEST_EPSILON));
}

TEST(test_gemv_avx2_direct) {
    /* Direct test of AVX2 GEMV implementation */
    const int m = 3, n = 8;
    double A[24];
    double x[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double y[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < 24; i++) A[i] = 1.0;

    fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);

    ASSERT_TRUE(double_equals(y[0], 8.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 8.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[2], 8.0, TEST_EPSILON));
}

TEST(test_gemv_basic) {
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {0.0, 0.0};

    int status = fc_mat_gemv_f64(2, 3, 1.0, A, 3, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(y[0], 14.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 32.0, TEST_EPSILON));
}

TEST(test_gemv_beta_nonzero) {
    /* Test beta != 0 path in SSE42 */
    double A[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };
    double x[] = {1.0, 1.0, 1.0, 1.0};
    double y[] = {10.0, 20.0};

    int status = fc_mat_gemv_f64(2, 4, 1.0, A, 4, x, 0.5, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 10.0 + 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 26.0 + 10.0, TEST_EPSILON));
}

TEST(test_gemv_beta_zero) {
    /* Test beta = 0 path with non-zero y */
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double x[] = {1.0, 1.0};
    double y[] = {999.0, 999.0};

    int status = fc_mat_gemv_f64(2, 2, 1.0, A, 2, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 7.0, TEST_EPSILON));
}

TEST(test_gemv_invalid_args) {
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double x[] = {1.0, 2.0};
    double y[] = {0.0, 0.0};

    ASSERT_EQ(fc_mat_gemv_f64(0, 2, 1.0, A, 2, x, 0.0, y), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemv_f64(2, 0, 1.0, A, 2, x, 0.0, y), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemv_f64(2, 2, 1.0, NULL, 2, x, 0.0, y), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemv_f64(2, 2, 1.0, A, 2, NULL, 0.0, y), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemv_f64(2, 2, 1.0, A, 2, x, 0.0, NULL), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemv_f64(2, 2, 1.0, A, 1, x, 0.0, y), FC_ERR_INVALID_ARG);
}

TEST(test_gemv_large_matrix) {
    /* Test larger matrix to exercise SIMD paths */
    const int m = 16;
    const int n = 16;
    double* A = (double*)malloc(m * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(m * sizeof(double));

    for (int i = 0; i < m * n; i++) {
        A[i] = 1.0;
    }
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
    }
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
    }

    int status = fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    /* Each row sums to n */
    for (int i = 0; i < m; i++) {
        ASSERT_TRUE(double_equals(y[i], (double)n, TEST_EPSILON));
    }

    free(A);
    free(x);
    free(y);
}

TEST(test_gemv_n_2) {
    /* Test n=2 to exercise SSE path with exact fit */
    const int m = 3;
    const int n = 2;
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[2] = {1.0, 1.0};
    double y[3] = {0.0, 0.0, 0.0};

    int status = fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 7.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[2], 11.0, TEST_EPSILON));
}

TEST(test_gemv_n_3) {
    /* Test n=3 to exercise remainder in SSE path */
    const int m = 2;
    const int n = 3;
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[3] = {1.0, 1.0, 1.0};
    double y[2] = {0.0, 0.0};

    int status = fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 6.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 15.0, TEST_EPSILON));
}

TEST(test_gemv_n_5) {
    /* Test n=5 to exercise SSE42 with remainder */
    double A[] = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0
    };
    double x[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double y[] = {0.0, 0.0};

    int status = fc_mat_gemv_f64(2, 5, 1.0, A, 5, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 15.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 40.0, TEST_EPSILON));
}

TEST(test_gemv_n_6) {
    /* Test n=6 (multiple of 2 for SSE42) */
    double A[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0
    };
    double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double y[] = {0.0, 0.0};

    int status = fc_mat_gemv_f64(2, 6, 1.0, A, 6, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 21.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 57.0, TEST_EPSILON));
}

TEST(test_gemv_n_not_multiple_of_4) {
    /* Test n=5 to exercise remainder path in AVX2 */
    const int m = 3;
    const int n = 5;
    double A[15];
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < m * n; i++) {
        A[i] = 1.0;
    }

    int status = fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    /* Each row: 1+2+3+4+5 = 15 */
    for (int i = 0; i < m; i++) {
        ASSERT_TRUE(double_equals(y[i], 15.0, TEST_EPSILON));
    }
}

TEST(test_gemv_non_multiple_of_4) {
    /* Test n not multiple of 4 to exercise remainder path */
    const int m = 5;
    const int n = 7;
    double A[35];
    double x[7];
    double y[5];

    for (int i = 0; i < m * n; i++) {
        A[i] = 1.0;
    }
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
    }
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
    }

    int status = fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < m; i++) {
        ASSERT_TRUE(double_equals(y[i], 14.0, TEST_EPSILON));
    }
}

TEST(test_gemv_odd_n) {
    /* Test n=3 to exercise SSE42 remainder path (n not multiple of 2) */
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double x[] = {1.0, 1.0, 1.0};
    double y[] = {0.0, 0.0, 0.0};

    int status = fc_mat_gemv_f64(3, 3, 1.0, A, 3, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 6.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 15.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[2], 24.0, TEST_EPSILON));
}

TEST(test_gemv_random) {
    srand(456);

    const int m = 20, n = 16;
    double A[320], x[16], y[20], y_ref[20];

    fill_random_matrix(A, m, n, n, -5.0, 5.0);
    fill_random_vector(x, n, -2.0, 2.0);
    fill_random_vector(y, m, -1.0, 1.0);
    memcpy(y_ref, y, sizeof(y));

    double alpha = 1.0, beta = 0.5;

    int status = fc_mat_gemv_f64(m, n, alpha, A, n, x, beta, y);
    ASSERT_EQ(status, FC_OK);

    /* Reference computation */
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y_ref[i] = alpha * sum + beta * y_ref[i];
    }

    /* Compare */
    for (int i = 0; i < m; i++) {
        ASSERT_TRUE(double_equals(y[i], y_ref[i], TEST_EPSILON_RELAXED));
    }
}

TEST(test_gemv_scalar_direct) {
    /* Direct test of scalar GEMV implementation */
    const int m = 3, n = 4;
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double x[4] = {1.0, 1.0, 1.0, 1.0};
    double y[3] = {0.0, 0.0, 0.0};

    fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);

    ASSERT_TRUE(double_equals(y[0], 10.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 26.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[2], 42.0, TEST_EPSILON));
}

TEST(test_gemv_scalar_with_beta) {
    /* Test scalar GEMV with beta != 0 */
    const int m = 2, n = 3;
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[3] = {1.0, 1.0, 1.0};
    double y[2] = {10.0, 20.0};

    fc_mat_gemv_f64(m, n, 2.0, A, n, x, 3.0, y);

    /* y = 2.0 * (6.0) + 3.0 * 10.0 = 42.0 */
    ASSERT_TRUE(double_equals(y[0], 42.0, TEST_EPSILON));
    /* y = 2.0 * (15.0) + 3.0 * 20.0 = 90.0 */
    ASSERT_TRUE(double_equals(y[1], 90.0, TEST_EPSILON));
}

TEST(test_gemv_single_col) {
    /* Test single column (n=1) */
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double x[] = {2.0};
    double y[] = {0.0, 0.0, 0.0, 0.0};

    int status = fc_mat_gemv_f64(4, 1, 1.0, A, 1, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 2.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 4.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[2], 6.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[3], 8.0, TEST_EPSILON));
}

TEST(test_gemv_single_row) {
    /* Test single row (m=1) */
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {0.0};

    int status = fc_mat_gemv_f64(1, 4, 1.0, A, 4, x, 0.0, y);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(y[0], 30.0, TEST_EPSILON));
}

TEST(test_gemv_sse42_direct) {
    /* Direct test of SSE4.2 GEMV implementation */
    const int m = 3, n = 4;
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double x[4] = {1.0, 1.0, 1.0, 1.0};
    double y[3] = {0.0, 0.0, 0.0};

    fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);

    ASSERT_TRUE(double_equals(y[0], 10.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 26.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[2], 42.0, TEST_EPSILON));
}

TEST(test_gemv_sse42_odd_n) {
    /* Test SSE4.2 GEMV with odd n to exercise remainder path */
    const int m = 2, n = 3;
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[3] = {1.0, 1.0, 1.0};
    double y[2] = {0.0, 0.0};

    fc_mat_gemv_f64(m, n, 1.0, A, n, x, 0.0, y);

    ASSERT_TRUE(double_equals(y[0], 6.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 15.0, TEST_EPSILON));
}

TEST(test_gemv_sse42_with_beta) {
    /* Test SSE4.2 GEMV with beta != 0 */
    const int m = 2, n = 4;
    double A[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double x[4] = {1.0, 1.0, 1.0, 1.0};
    double y[2] = {5.0, 10.0};

    fc_mat_gemv_f64(m, n, 1.0, A, n, x, 2.0, y);

    /* y = 1.0 * (10.0) + 2.0 * 5.0 = 20.0 */
    ASSERT_TRUE(double_equals(y[0], 20.0, TEST_EPSILON));
    /* y = 1.0 * (26.0) + 2.0 * 10.0 = 46.0 */
    ASSERT_TRUE(double_equals(y[1], 46.0, TEST_EPSILON));
}

TEST(test_gemv_with_beta_nonzero) {
    /* Test GEMV with beta != 0 to cover that branch */
    const int m = 4;
    const int n = 8;
    double A[32];
    double x[8];
    double y[4] = {1.0, 2.0, 3.0, 4.0};

    for (int i = 0; i < m * n; i++) {
        A[i] = 1.0;
    }
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
    }

    int status = fc_mat_gemv_f64(m, n, 2.0, A, n, x, 3.0, y);
    ASSERT_EQ(status, FC_OK);

    /* y = 2.0 * (8.0) + 3.0 * y_old */
    ASSERT_TRUE(double_equals(y[0], 2.0 * 8.0 + 3.0 * 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(y[1], 2.0 * 8.0 + 3.0 * 2.0, TEST_EPSILON));
}

void test_gemv_register(void) {
    RUN_TEST(test_gemv_alpha_beta);
    RUN_TEST(test_gemv_alpha_zero);
    RUN_TEST(test_gemv_avx2_direct);
    RUN_TEST(test_gemv_basic);
    RUN_TEST(test_gemv_beta_nonzero);
    RUN_TEST(test_gemv_beta_zero);
    RUN_TEST(test_gemv_invalid_args);
    RUN_TEST(test_gemv_large_matrix);
    RUN_TEST(test_gemv_n_2);
    RUN_TEST(test_gemv_n_3);
    RUN_TEST(test_gemv_n_5);
    RUN_TEST(test_gemv_n_6);
    RUN_TEST(test_gemv_n_not_multiple_of_4);
    RUN_TEST(test_gemv_non_multiple_of_4);
    RUN_TEST(test_gemv_odd_n);
    RUN_TEST(test_gemv_random);
    RUN_TEST(test_gemv_scalar_direct);
    RUN_TEST(test_gemv_scalar_with_beta);
    RUN_TEST(test_gemv_single_col);
    RUN_TEST(test_gemv_single_row);
    RUN_TEST(test_gemv_sse42_direct);
    RUN_TEST(test_gemv_sse42_odd_n);
    RUN_TEST(test_gemv_sse42_with_beta);
    RUN_TEST(test_gemv_with_beta_nonzero);
}
