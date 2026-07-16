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

/* Naive GEMM for reference (C = alpha*A*B + beta*C) */
static void gemm_reference(int m, int n, int k,
                           double alpha,
                           const double* A, int lda,
                           const double* B, int ldb,
                           double beta,
                           double* C, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

/*
 * Vector operations tests
*/


TEST(test_inverse_2x2) {
    /* Simple 2x2 matrix */
    double A[] = {
        4.0, 7.0,
        2.0, 6.0
    };

    int status = fc_mat_inverse_f64(2, A, 2);
    ASSERT_EQ(status, FC_OK);

    /* Expected inverse: [0.6, -0.7; -0.2, 0.4] */
    ASSERT_TRUE(double_equals(A[0], 0.6, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(A[1], -0.7, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(A[2], -0.2, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(A[3], 0.4, TEST_EPSILON_RELAXED));
}

TEST(test_inverse_basic) {
    /* Invert a 3x3 matrix */
    double A[] = {
        2.0, 1.0, 1.0,
        4.0, 3.0, 3.0,
        8.0, 7.0, 9.0
    };
    double A_orig[9];
    memcpy(A_orig, A, sizeof(A));

    int status = fc_mat_inverse_f64(3, A, 3);
    ASSERT_EQ(status, FC_OK);

    /* Verify A * A_inv = I */
    double result[9] = {0};
    gemm_reference(3, 3, 3, 1.0, A_orig, 3, A, 3, 0.0, result, 3);

    /* Check diagonal elements are 1 */
    ASSERT_TRUE(double_equals(result[0], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(result[4], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(result[8], 1.0, TEST_EPSILON_RELAXED));

    /* Check off-diagonal elements are 0 */
    ASSERT_TRUE(fabs(result[1]) < TEST_EPSILON_RELAXED);
    ASSERT_TRUE(fabs(result[2]) < TEST_EPSILON_RELAXED);
    ASSERT_TRUE(fabs(result[3]) < TEST_EPSILON_RELAXED);
}

TEST(test_inverse_identity) {
    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    int status = fc_mat_inverse_f64(3, A, 3);
    ASSERT_EQ(status, FC_OK);

    /* Identity inverse is identity */
    ASSERT_TRUE(double_equals(A[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(A[4], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(A[8], 1.0, TEST_EPSILON));
}

TEST(test_inverse_invalid_args) {
    double A[9];

    ASSERT_EQ(fc_mat_inverse_f64(3, NULL, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_inverse_f64(0, A, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_inverse_f64(-1, A, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_inverse_f64(3, A, 2), FC_ERR_INVALID_ARG);
}

TEST(test_solve_linear_basic) {
    /* Solve A*x = b where A is 3x3 */
    double A[] = {
        2.0, 1.0, 1.0,
        4.0, 3.0, 3.0,
        8.0, 7.0, 9.0
    };
    double b[] = {4.0, 10.0, 24.0};

    int status = fc_mat_solve_linear_f64(3, 1, A, 3, b, 1);
    ASSERT_EQ(status, FC_OK);

    /* Solution should be [1, 1, 1] */
    ASSERT_TRUE(double_equals(b[0], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(b[1], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(b[2], 1.0, TEST_EPSILON_RELAXED));
}

TEST(test_solve_linear_identity) {
    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    double b[] = {3.0, 5.0, 7.0};

    int status = fc_mat_solve_linear_f64(3, 1, A, 3, b, 1);
    ASSERT_EQ(status, FC_OK);

    /* Solution should be b itself */
    ASSERT_TRUE(double_equals(b[0], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[1], 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[2], 7.0, TEST_EPSILON));
}

TEST(test_solve_linear_invalid_args) {
    double A[4], b[2];

    ASSERT_EQ(fc_mat_solve_linear_f64(2, 1, NULL, 2, b, 1), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_linear_f64(2, 1, A, 2, NULL, 1), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_linear_f64(0, 1, A, 2, b, 1), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_linear_f64(2, 0, A, 2, b, 1), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_linear_f64(2, 1, A, 1, b, 1), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_linear_f64(2, 1, A, 2, b, 0), FC_ERR_INVALID_ARG);
}

TEST(test_solve_linear_multiple_rhs) {
    /* Solve A*X = B where B has 2 columns */
    double A[] = {
        2.0, 1.0,
        1.0, 3.0
    };
    /* B is stored in row-major order with ldb=2 (2 columns) */
    /* For solution X = [[1, 2], [2, 3]], B = A*X = [[4, 7], [7, 11]] */
    double B[] = {
        4.0, 7.0,   /* First row */
        7.0, 11.0   /* Second row */
    };

    int status = fc_mat_solve_linear_f64(2, 2, A, 2, B, 2);
    ASSERT_EQ(status, FC_OK);

    /* First solution should be [1, 2] (column 0) */
    ASSERT_TRUE(double_equals(B[0], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(B[2], 2.0, TEST_EPSILON_RELAXED));

    /* Second solution should be [2, 3] (column 1) */
    ASSERT_TRUE(double_equals(B[1], 2.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(B[3], 3.0, TEST_EPSILON_RELAXED));
}

void test_solve_register(void) {
    RUN_TEST(test_inverse_2x2);
    RUN_TEST(test_inverse_basic);
    RUN_TEST(test_inverse_identity);
    RUN_TEST(test_inverse_invalid_args);
    RUN_TEST(test_solve_linear_basic);
    RUN_TEST(test_solve_linear_identity);
    RUN_TEST(test_solve_linear_invalid_args);
    RUN_TEST(test_solve_linear_multiple_rhs);
}
