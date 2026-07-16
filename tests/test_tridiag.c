/**
 * @file test_tridiag.c
 * @brief Unit tests for tridiagonal matrix solver
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_framework.h"
#include <tridiag.h>
#include <error.h>

/* Test tolerance */
#define TEST_EPSILON 1e-10

/* Test basic tridiagonal solve */
TEST(test_tridiag_solve_basic) {
    /* Solve a simple 4x4 tridiagonal system:
     * [2 1 0 0] [x0]   [1]
     * [1 2 1 0] [x1] = [2]
     * [0 1 2 1] [x2]   [3]
     * [0 0 1 2] [x3]   [4]
     */
    int64_t n = 4;
    double a[] = {0.0, 1.0, 1.0, 1.0};  /* Lower diagonal (a[0] unused) */
    double b[] = {2.0, 2.0, 2.0, 2.0};  /* Main diagonal */
    double c[] = {1.0, 1.0, 1.0, 0.0};  /* Upper diagonal (c[n-1] unused) */
    double d[] = {1.0, 2.0, 3.0, 4.0};  /* RHS */
    double x[4];

    int status = fc_mat_tridiag_solve_f64(n, a, b, c, d, x);
    ASSERT_EQ(status, FC_OK);

    /* Verify solution by computing residual */
    double residual[4];
    residual[0] = b[0] * x[0] + c[0] * x[1] - d[0];
    for (int64_t i = 1; i < n - 1; i++) {
        residual[i] = a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] - d[i];
    }
    residual[n-1] = a[n-1] * x[n-2] + b[n-1] * x[n-1] - d[n-1];

    for (int64_t i = 0; i < n; i++) {
        ASSERT_TRUE(fabs(residual[i]) < TEST_EPSILON);
    }
}

/* Test tridiagonal solve with identity matrix */
TEST(test_tridiag_solve_identity) {
    int64_t n = 5;
    double a[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double b[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double c[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double d[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[5];

    int status = fc_mat_tridiag_solve_f64(n, a, b, c, d, x);
    ASSERT_EQ(status, FC_OK);

    /* Solution should equal RHS */
    for (int64_t i = 0; i < n; i++) {
        ASSERT_TRUE(fabs(x[i] - d[i]) < TEST_EPSILON);
    }
}

/* Test tridiagonal solve with multiple RHS */
TEST(test_tridiag_solve_multi) {
    int64_t n = 3;
    int64_t nrhs = 2;
    double a[] = {0.0, 1.0, 1.0};
    double b[] = {2.0, 2.0, 2.0};
    double c[] = {1.0, 1.0, 0.0};
    double D[] = {1.0, 2.0,   /* First RHS */
                  2.0, 3.0,   /* Second RHS */
                  3.0, 4.0};  /* Third row */
    double X[6];

    int status = fc_mat_tridiag_solve_multi_f64(n, nrhs, a, b, c, D, nrhs, X, nrhs);
    ASSERT_EQ(status, FC_OK);

    /* Verify each solution */
    for (int64_t j = 0; j < nrhs; j++) {
        double residual[3];
        residual[0] = b[0] * X[0 * nrhs + j] + c[0] * X[1 * nrhs + j] - D[0 * nrhs + j];
        residual[1] = a[1] * X[0 * nrhs + j] + b[1] * X[1 * nrhs + j] + c[1] * X[2 * nrhs + j] - D[1 * nrhs + j];
        residual[2] = a[2] * X[1 * nrhs + j] + b[2] * X[2 * nrhs + j] - D[2 * nrhs + j];

        for (int64_t i = 0; i < n; i++) {
            ASSERT_TRUE(fabs(residual[i]) < TEST_EPSILON);
        }
    }
}

/* Test tridiagonal solve with invalid arguments */
TEST(test_tridiag_solve_invalid_args) {
    int64_t n = 3;
    double a[3] = {0}, b[3] = {0}, c[3] = {0}, d[3] = {0}, x[3] = {0};

    /* NULL pointers */
    ASSERT_EQ(fc_mat_tridiag_solve_f64(n, NULL, b, c, d, x), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_tridiag_solve_f64(n, a, NULL, c, d, x), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_tridiag_solve_f64(n, a, b, NULL, d, x), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_tridiag_solve_f64(n, a, b, c, NULL, x), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_tridiag_solve_f64(n, a, b, c, d, NULL), FC_ERR_INVALID_ARG);

    /* Invalid dimensions */
    ASSERT_EQ(fc_mat_tridiag_solve_f64(0, a, b, c, d, x), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_tridiag_solve_f64(-1, a, b, c, d, x), FC_ERR_INVALID_ARG);
}

/* Register all tridiagonal tests */
void register_tridiag_tests(void) {
    RUN_TEST(test_tridiag_solve_basic);
    RUN_TEST(test_tridiag_solve_identity);
    RUN_TEST(test_tridiag_solve_multi);
    RUN_TEST(test_tridiag_solve_invalid_args);
}
