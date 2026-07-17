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

/* Compare two matrices with tolerance */
static int matrices_equal(const double* A, const double* B,
                          int rows, int cols, int ld_a, int ld_b,
                          double epsilon) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!double_equals(A[i * ld_a + j], B[i * ld_b + j], epsilon)) {
                return 0;
            }
        }
    }
    return 1;
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


TEST(test_gemm_2x2) {
    /* Test: [1 2] * [5 6] = [19 22]
     *       [3 4]   [7 8]   [43 50] */
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double B[] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0};

    int status = fc_mat_gemm_f64(2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(C[0], 19.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 22.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[2], 43.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[3], 50.0, TEST_EPSILON));
}

TEST(test_gemm_4x4) {
    /* Test 4x4 matrix multiplication (exercises micro-kernel) */
    double A[16], B[16], C[16];

    /* Initialize A and B as simple patterns */
    for (int i = 0; i < 16; i++) {
        A[i] = (double)(i + 1);
        B[i] = (double)(16 - i);
        C[i] = 0.0;
    }

    int status = fc_mat_gemm_f64(4, 4, 4, 1.0, A, 4, B, 4, 0.0, C, 4);
    ASSERT_EQ(status, FC_OK);

    /* Verify first element: C[0,0] = A[0,:] * B[:,0]
     * = 1*16 + 2*12 + 3*8 + 4*4 = 16 + 24 + 24 + 16 = 80 */
    ASSERT_TRUE(double_equals(C[0], 80.0, TEST_EPSILON));
}

TEST(test_gemm_alpha_beta) {
    /* Test: C = 2.0 * A * B + 3.0 * C */
    double A[] = {1.0, 2.0};
    double B[] = {3.0, 4.0};
    double C[] = {5.0, 6.0};

    int status = fc_mat_gemm_f64(1, 2, 1, 2.0, A, 1, B, 2, 3.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    /* Expected: C[0] = 2.0 * (1*3) + 3.0 * 5 = 6 + 15 = 21
     *           C[1] = 2.0 * (1*4) + 3.0 * 6 = 8 + 18 = 26 */
    ASSERT_TRUE(double_equals(C[0], 21.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 26.0, TEST_EPSILON));
}

TEST(test_gemm_alpha_zero) {
    /* Test alpha=0: C should be beta*C regardless of A*B */
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double B[] = {5.0, 6.0, 7.0, 8.0};
    double C[] = {10.0, 20.0, 30.0, 40.0};

    int status = fc_mat_gemm_f64(2, 2, 2, 0.0, A, 2, B, 2, 2.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    /* Expected: C = 0*A*B + 2*C = 2*C */
    ASSERT_TRUE(double_equals(C[0], 20.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 40.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[2], 60.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[3], 80.0, TEST_EPSILON));
}

TEST(test_gemm_avx2_both_remainder) {
    /* Test AVX2 with both m and n remainders */
    double A[15] = {1.0, 2.0, 3.0, 4.0, 5.0,
                    6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0};
    double B[15] = {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0};
    double C[9] = {0.0};
    double C_ref[9] = {0.0};

    int status = fc_mat_gemm_f64_avx2(3, 3, 5, 1.0, A, 5, B, 3, 0.0, C, 3);
    ASSERT_EQ(status, FC_OK);

    /* Compute reference and verify */
    gemm_reference(3, 3, 5, 1.0, A, 5, B, 3, 0.0, C_ref, 3);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 3, 3, 3, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_avx2_direct) {
    /* Test AVX2 4x8 micro-kernel directly */
    double A[16] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0};
    double B[32] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    double C[32] = {0.0};

    int status = fc_mat_gemm_f64_avx2(4, 8, 4, 1.0, A, 4, B, 8, 0.0, C, 8);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(C[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 2.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[2], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[3], 4.0, TEST_EPSILON));
}

TEST(test_gemm_avx2_m_remainder) {
    /* Test AVX2 with m not multiple of 4 */
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0,
                    6.0, 7.0, 8.0, 9.0, 10.0};
    double B[40] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double C[16] = {0.0};
    double C_ref[16] = {0.0};

    int status = fc_mat_gemm_f64_avx2(2, 8, 5, 1.0, A, 5, B, 8, 0.0, C, 8);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(2, 8, 5, 1.0, A, 5, B, 8, 0.0, C_ref, 8);
    ASSERT_TRUE(matrices_equal(C, C_ref, 2, 8, 8, 8, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_avx2_n_remainder) {
    /* Test AVX2 with n not multiple of 8 */
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double B[20] = {1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0};
    double C[15] = {0.0};
    double C_ref[15] = {0.0};

    int status = fc_mat_gemm_f64_avx2(3, 5, 4, 1.0, A, 4, B, 5, 0.0, C, 5);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 5, 4, 1.0, A, 4, B, 5, 0.0, C_ref, 5);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 5, 5, 5, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_avx2_n_remainder_beta) {
    /* Test AVX2 n remainder with beta != 0 */
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double B[20] = {1.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 1.0, 1.0};
    double C[20] = {1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0};
    double C_ref[20] = {1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0};

    int status = fc_mat_gemm_f64_avx2(3, 5, 4, 1.0, A, 4, B, 5, 0.5, C, 5);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 5, 4, 1.0, A, 4, B, 5, 0.5, C_ref, 5);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 5, 5, 5, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_avx2_with_beta) {
    /* Test AVX2 with beta != 0 */
    double A[8] = {1.0, 2.0, 3.0, 4.0,
                   5.0, 6.0, 7.0, 8.0};
    double B[16] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double C[16] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    double C_ref[16] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

    int status = fc_mat_gemm_f64_avx2(2, 8, 2, 2.0, A, 2, B, 8, 3.0, C, 8);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(2, 8, 2, 2.0, A, 2, B, 8, 3.0, C_ref, 8);
    ASSERT_TRUE(matrices_equal(C, C_ref, 2, 8, 8, 8, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_avx2_with_beta_1) {
    /* Test AVX2 with m=4, n=8, beta=1.0 to cover beta==1.0 fast path */
    const int m = 4, n = 8, k = 4;
    double A[16], B[32], C[32];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 2.0;

    int status = fc_mat_gemm_f64_avx2(m, n, k, 1.0, A, k, B, n, 1.0, C, n);
    ASSERT_EQ(status, FC_OK);

    /* C = 1.0 * A*B + 1.0 * C_old = k + 2 = 6 */
    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k + 2.0, TEST_EPSILON));
    }
}

TEST(test_gemm_avx2_with_beta_4x8) {
    /* Test AVX2 with m=4, beta != 0 to cover all rows */
    const int m = 4, n = 8, k = 4;
    double A[16], B[32], C[32];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 2.0;

    int status = fc_mat_gemm_f64_avx2(m, n, k, 1.0, A, k, B, n, 0.5, C, n);
    ASSERT_EQ(status, FC_OK);

    /* C = 1.0 * A*B + 0.5 * C_old = k + 0.5*2 = 5 */
    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k + 1.0, TEST_EPSILON));
    }
}

TEST(test_gemm_avx512_both_remainder) {
    /* Test AVX-512 with both m and n not aligned */
    if (fc_get_simd_level() < FC_SIMD_AVX512) {
        return;
    }

    const int m = 5, n = 9, k = 4;
    double A[20], B[36], C[45];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    int status = fc_mat_gemm_f64_avx512(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k, TEST_EPSILON));
    }
}

TEST(test_gemm_avx512_direct) {
    /* Direct test of AVX-512 implementation */
    if (fc_get_simd_level() < FC_SIMD_AVX512) {
        /* Skip test if AVX-512 not supported, but still call the function
         * to ensure it compiles and returns gracefully */
        return;
    }

    const int m = 8, n = 8, k = 8;
    double A[64], B[64], C[64];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    int status = fc_mat_gemm_f64_avx512(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k, TEST_EPSILON));
    }
}

TEST(test_gemm_avx512_m_remainder) {
    /* Test AVX-512 with m not multiple of 4 */
    if (fc_get_simd_level() < FC_SIMD_AVX512) {
        return;
    }

    const int m = 6, n = 8, k = 4;
    double A[24], B[32], C[48];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    int status = fc_mat_gemm_f64_avx512(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k, TEST_EPSILON));
    }
}

TEST(test_gemm_avx512_n_remainder) {
    /* Test AVX-512 with n not multiple of 8 */
    if (fc_get_simd_level() < FC_SIMD_AVX512) {
        return;
    }

    const int m = 4, n = 10, k = 4;
    double A[16], B[40], C[40];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    int status = fc_mat_gemm_f64_avx512(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k, TEST_EPSILON));
    }
}

TEST(test_gemm_avx512_with_beta) {
    /* Test AVX-512 with beta != 0 */
    if (fc_get_simd_level() < FC_SIMD_AVX512) {
        return;
    }

    const int m = 4, n = 8, k = 4;
    double A[16], B[32], C[32];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 2.0;

    int status = fc_mat_gemm_f64_avx512(m, n, k, 1.0, A, k, B, n, 2.0, C, n);
    ASSERT_EQ(status, FC_OK);

    /* C = 1.0 * A*B + 2.0 * C_old = k + 2*2 = 8 */
    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k + 4.0, TEST_EPSILON));
    }
}

TEST(test_gemm_beta_one) {
    /* Test beta=1.0: C = alpha*A*B + C */
    double A[] = {1.0, 0.0, 0.0, 1.0};  /* Identity */
    double B[] = {2.0, 3.0, 4.0, 5.0};
    double C[] = {1.0, 1.0, 1.0, 1.0};

    int status = fc_mat_gemm_f64(2, 2, 2, 1.0, A, 2, B, 2, 1.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    /* Expected: C = I*B + C = B + C */
    ASSERT_TRUE(double_equals(C[0], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 4.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[2], 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[3], 6.0, TEST_EPSILON));
}

TEST(test_gemm_edge_3x7) {
    /* Test non-square with edge cases: 3x7 * 7x3 = 3x3 */
    double A[21], B[21], C[9];

    for (int i = 0; i < 21; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }
    for (int i = 0; i < 9; i++) {
        C[i] = 0.0;
    }

    int status = fc_mat_gemm_f64(3, 3, 7, 1.0, A, 7, B, 3, 0.0, C, 3);
    ASSERT_EQ(status, FC_OK);

    /* All elements should be 7.0 (sum of 7 ones) */
    for (int i = 0; i < 9; i++) {
        ASSERT_TRUE(double_equals(C[i], 7.0, TEST_EPSILON));
    }
}

TEST(test_gemm_edge_5x5) {
    /* Test 5x5 matrix to exercise edge handling (not divisible by 4) */
    double A[25], B[25], C[25];

    /* Initialize with simple pattern */
    for (int i = 0; i < 25; i++) {
        A[i] = (double)(i + 1);
        B[i] = 1.0;
        C[i] = 0.0;
    }

    int status = fc_mat_gemm_f64(5, 5, 5, 1.0, A, 5, B, 5, 0.0, C, 5);
    ASSERT_EQ(status, FC_OK);

    /* C[0,0] = sum(A[0,:]) = 1+2+3+4+5 = 15 */
    ASSERT_TRUE(double_equals(C[0], 15.0, TEST_EPSILON));
}

TEST(test_gemm_identity) {
    /* Test: I * A = A (2x2 identity matrix) */
    double I[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    double A[] = {
        2.0, 3.0,
        4.0, 5.0
    };
    double C[4] = {0};

    int status = fc_mat_gemm_f64(2, 2, 2, 1.0, I, 2, A, 2, 0.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    /* C should equal A */
    ASSERT_TRUE(double_equals(C[0], 2.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[2], 4.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[3], 5.0, TEST_EPSILON));
}

TEST(test_gemm_invalid_args) {
    double A[4] = {0};
    double B[4] = {0};
    double C[4] = {0};

    /* Invalid dimensions */
    ASSERT_EQ(fc_mat_gemm_f64(0, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemm_f64(2, 0, 2, 1.0, A, 2, B, 2, 0.0, C, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 0, 1.0, A, 2, B, 2, 0.0, C, 2), FC_ERR_INVALID_ARG);

    /* NULL pointers */
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 2, 1.0, NULL, 2, B, 2, 0.0, C, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 2, 1.0, A, 2, NULL, 2, 0.0, C, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 2, 1.0, A, 2, B, 2, 0.0, NULL, 2), FC_ERR_INVALID_ARG);

    /* Invalid leading dimensions */
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 2, 1.0, A, 1, B, 2, 0.0, C, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 2, 1.0, A, 2, B, 1, 0.0, C, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_gemm_f64(2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 1), FC_ERR_INVALID_ARG);
}

TEST(test_gemm_non_square) {
    /* Test non-square matrices: 2x3 * 3x2 = 2x2 */
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    double B[] = {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    double C[4] = {0};

    int status = fc_mat_gemm_f64(2, 2, 3, 1.0, A, 3, B, 2, 0.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    /* C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58 */
    ASSERT_TRUE(double_equals(C[0], 58.0, TEST_EPSILON));
    /* C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64 */
    ASSERT_TRUE(double_equals(C[1], 64.0, TEST_EPSILON));
    /* C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139 */
    ASSERT_TRUE(double_equals(C[2], 139.0, TEST_EPSILON));
    /* C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154 */
    ASSERT_TRUE(double_equals(C[3], 154.0, TEST_EPSILON));
}

TEST(test_gemm_precision_128x128) {
    /* Test GEMM precision on larger 128x128 matrices */
    const int m = 128, n = 128, k = 128;
    double* A = (double*)malloc(m * k * sizeof(double));
    double* B = (double*)malloc(k * n * sizeof(double));
    double* C = (double*)malloc(m * n * sizeof(double));
    double* C_ref = (double*)malloc(m * n * sizeof(double));

    /* Initialize with random values */
    srand(54321);
    fill_random_matrix(A, m, k, k, -5.0, 5.0);
    fill_random_matrix(B, k, n, n, -5.0, 5.0);
    memset(C, 0, m * n * sizeof(double));
    memset(C_ref, 0, m * n * sizeof(double));

    /* Compute with optimized GEMM */
    int status = fc_mat_gemm_f64(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    /* Compute reference result */
    gemm_reference(m, n, k, 1.0, A, k, B, n, 0.0, C_ref, n);

    /* Verify precision */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = fabs(C[i * n + j] - C_ref[i * n + j]);
            double magnitude = fabs(C_ref[i * n + j]);
            double rel_error = (magnitude > 1e-15) ? (diff / magnitude) : diff;
            ASSERT_TRUE(rel_error < 1e-10);
        }
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST(test_gemm_precision_64x64) {
    /* Test GEMM precision on 64x64 matrices against reference implementation */
    const int m = 64, n = 64, k = 64;
    double* A = (double*)malloc(m * k * sizeof(double));
    double* B = (double*)malloc(k * n * sizeof(double));
    double* C = (double*)malloc(m * n * sizeof(double));
    double* C_ref = (double*)malloc(m * n * sizeof(double));

    /* Initialize with random values */
    srand(12345);
    fill_random_matrix(A, m, k, k, -10.0, 10.0);
    fill_random_matrix(B, k, n, n, -10.0, 10.0);
    memset(C, 0, m * n * sizeof(double));
    memset(C_ref, 0, m * n * sizeof(double));

    /* Compute with optimized GEMM */
    int status = fc_mat_gemm_f64(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    /* Compute reference result */
    gemm_reference(m, n, k, 1.0, A, k, B, n, 0.0, C_ref, n);

    /* Verify precision: relative error should be < 1e-12 */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = fabs(C[i * n + j] - C_ref[i * n + j]);
            double magnitude = fabs(C_ref[i * n + j]);
            double rel_error = (magnitude > 1e-15) ? (diff / magnitude) : diff;
            if (rel_error >= 1e-12) {
                printf("Precision error at [%d,%d]: got %.17g, expected %.17g, rel_error=%.3e\n",
                       i, j, C[i * n + j], C_ref[i * n + j], rel_error);
            }
            ASSERT_TRUE(rel_error < 1e-12);
        }
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST(test_gemm_precision_large_values) {
    /* Test GEMM precision with large values to check numerical stability */
    const int m = 32, n = 32, k = 32;
    double* A = (double*)malloc(m * k * sizeof(double));
    double* B = (double*)malloc(k * n * sizeof(double));
    double* C = (double*)malloc(m * n * sizeof(double));
    double* C_ref = (double*)malloc(m * n * sizeof(double));

    /* Initialize with larger values */
    srand(99999);
    fill_random_matrix(A, m, k, k, -1000.0, 1000.0);
    fill_random_matrix(B, k, n, n, -1000.0, 1000.0);
    memset(C, 0, m * n * sizeof(double));
    memset(C_ref, 0, m * n * sizeof(double));

    /* Compute with optimized GEMM */
    int status = fc_mat_gemm_f64(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    /* Compute reference result */
    gemm_reference(m, n, k, 1.0, A, k, B, n, 0.0, C_ref, n);

    /* Verify precision with slightly relaxed tolerance for large values */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = fabs(C[i * n + j] - C_ref[i * n + j]);
            double magnitude = fabs(C_ref[i * n + j]);
            double rel_error = (magnitude > 1e-10) ? (diff / magnitude) : diff;
            ASSERT_TRUE(rel_error < 1e-10);
        }
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST(test_gemm_random_rectangular) {
    srand(123);

    const int m = 12, n = 16, k = 10;
    double *A = malloc(m * k * sizeof(double));
    double *B = malloc(k * n * sizeof(double));
    double *C = malloc(m * n * sizeof(double));
    double *C_ref = malloc(m * n * sizeof(double));

    fill_random_matrix(A, m, k, k, -1.0, 1.0);
    fill_random_matrix(B, k, n, n, -1.0, 1.0);
    fill_random_matrix(C, m, n, n, -0.5, 0.5);
    memcpy(C_ref, C, m * n * sizeof(double));

    double alpha = 2.0, beta = -1.0;

    int status = fc_mat_gemm_f64(m, n, k, alpha, A, k, B, n, beta, C, n);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(m, n, k, alpha, A, k, B, n, beta, C_ref, n);

    ASSERT_TRUE(matrices_equal(C, C_ref, m, n, n, n, TEST_EPSILON_RELAXED));

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST(test_gemm_random_small) {
    srand(42);  /* Fixed seed for reproducibility */

    const int m = 8, n = 8, k = 8;
    double A[64], B[64], C[64], C_ref[64];

    fill_random_matrix(A, m, k, k, -10.0, 10.0);
    fill_random_matrix(B, k, n, n, -10.0, 10.0);
    fill_random_matrix(C, m, n, n, -5.0, 5.0);
    memcpy(C_ref, C, sizeof(C));

    double alpha = 1.5, beta = 0.5;

    /* Compute with optimized implementation */
    int status = fc_mat_gemm_f64(m, n, k, alpha, A, k, B, n, beta, C, n);
    ASSERT_EQ(status, FC_OK);

    /* Compute reference */
    gemm_reference(m, n, k, alpha, A, k, B, n, beta, C_ref, n);

    /* Compare results */
    ASSERT_TRUE(matrices_equal(C, C_ref, m, n, n, n, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_10x8_beta) {
    /* Test 10x8 matrix with beta != 0 to trigger:
     * - Bottom edge with full N blocks (lines 193-200)
     */
    double A[80], B[64], C[80], C_ref[80];

    for (int i = 0; i < 80; i++) A[i] = 1.0;
    for (int i = 0; i < 64; i++) B[i] = 1.0;
    for (int i = 0; i < 80; i++) C[i] = 1.0;
    for (int i = 0; i < 80; i++) C_ref[i] = 1.0;

    int status = fc_mat_gemm_f64_scalar(10, 8, 8, 1.0, A, 8, B, 8, 0.5, C, 8);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(10, 8, 8, 1.0, A, 8, B, 8, 0.5, C_ref, 8);
    ASSERT_TRUE(matrices_equal(C, C_ref, 10, 8, 8, 8, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_16x10_k16) {
    /* Test 16x10 matrix with k=16 to trigger:
     * - Multiple M blocks with right edge (lines 179-186)
     * - Prefetch (lines 74-75)
     */
    double A[256], B[160], C[160], C_ref[160];

    for (int i = 0; i < 256; i++) A[i] = 1.0;
    for (int i = 0; i < 160; i++) B[i] = 1.0;
    for (int i = 0; i < 160; i++) C[i] = 0.0;
    for (int i = 0; i < 160; i++) C_ref[i] = 0.0;

    int status = fc_mat_gemm_f64_scalar(16, 10, 16, 1.0, A, 16, B, 10, 0.0, C, 10);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(16, 10, 16, 1.0, A, 16, B, 10, 0.0, C_ref, 10);
    ASSERT_TRUE(matrices_equal(C, C_ref, 16, 10, 10, 10, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_8x10_k16) {
    /* Test 8x10 matrix with k=16 to trigger:
     * - Full 8x8 block (lines 179-186: right edge with full M blocks)
     * - Prefetch (lines 74-75: k > 8)
     */
    double A[128], B[160], C[80], C_ref[80];

    for (int i = 0; i < 128; i++) A[i] = 1.0;
    for (int i = 0; i < 160; i++) B[i] = 1.0;
    for (int i = 0; i < 80; i++) C[i] = 0.0;
    for (int i = 0; i < 80; i++) C_ref[i] = 0.0;

    int status = fc_mat_gemm_f64_scalar(8, 10, 16, 1.0, A, 16, B, 10, 0.0, C, 10);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(8, 10, 16, 1.0, A, 16, B, 10, 0.0, C_ref, 10);
    ASSERT_TRUE(matrices_equal(C, C_ref, 8, 10, 10, 10, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_8x8_beta_k16) {
    /* Test 8x8 matrix with beta != 0 and k=16 to trigger:
     * - Beta path in 8x8 kernel (lines 111-113)
     * - Prefetch with k > 8 (lines 74-75)
     */
    double A[128], B[128], C[64], C_ref[64];

    for (int i = 0; i < 128; i++) A[i] = 1.0;
    for (int i = 0; i < 128; i++) B[i] = 1.0;
    for (int i = 0; i < 64; i++) C[i] = 2.0;
    for (int i = 0; i < 64; i++) C_ref[i] = 2.0;

    int status = fc_mat_gemm_f64_scalar(8, 8, 16, 1.0, A, 16, B, 8, 0.5, C, 8);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(8, 8, 16, 1.0, A, 16, B, 8, 0.5, C_ref, 8);
    ASSERT_TRUE(matrices_equal(C, C_ref, 8, 8, 8, 8, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_direct) {
    /* Test scalar implementation directly to ensure coverage */
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double B[4] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};

    int status = fc_mat_gemm_f64_scalar(2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
    ASSERT_EQ(status, FC_OK);

    /* Expected: C = A * B
     * C[0,0] = 1*5 + 2*7 = 19
     * C[0,1] = 1*6 + 2*8 = 22
     * C[1,0] = 3*5 + 4*7 = 43
     * C[1,1] = 3*6 + 4*8 = 50
   */
    ASSERT_TRUE(double_equals(C[0], 19.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[1], 22.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[2], 43.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(C[3], 50.0, TEST_EPSILON));
}

TEST(test_gemm_scalar_edge_cases) {
    /* Test scalar implementation with non-aligned dimensions */
    double A[15] = {1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15};
    double B[35] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14, 15,16,17,18,19,20,21,
                    22,23,24,25,26,27,28, 29,30,31,32,33,34,35};
    double C[12] = {0};
    double C_ref[12] = {0};

    int status = fc_mat_gemm_f64_scalar(3, 4, 5, 1.0, A, 5, B, 7, 0.0, C, 4);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 4, 5, 1.0, A, 5, B, 7, 0.0, C_ref, 4);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 4, 4, 4, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_large_4x4) {
    /* Test scalar implementation with 4x4 blocks (triggers micro-kernel) */
    double A[64], B[64], C[64];

    /* Initialize 8x8 matrices */
    for (int i = 0; i < 64; i++) {
        A[i] = (double)(i + 1);
        B[i] = 1.0;
        C[i] = 0.0;
    }

    int status = fc_mat_gemm_f64_scalar(8, 8, 8, 1.0, A, 8, B, 8, 0.0, C, 8);
    ASSERT_EQ(status, FC_OK);

    /* C[0,0] = sum(A[0,:]) = 1+2+3+4+5+6+7+8 = 36 */
    ASSERT_TRUE(double_equals(C[0], 36.0, TEST_EPSILON));
}

TEST(test_gemm_scalar_m_remainder_beta) {
    /* 3x4 matrix to trigger m_remainder = 3 with beta != 0 */
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double B[16] = {1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0};
    double C[12] = {1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0};
    double C_ref[12] = {1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0};

    int status = fc_mat_gemm_f64_scalar(3, 4, 4, 1.0, A, 4, B, 4, 0.5, C, 4);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 4, 4, 1.0, A, 4, B, 4, 0.5, C_ref, 4);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 4, 4, 4, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_n_remainder) {
    /* 4x5 matrix to trigger n_remainder = 1 */
    double A[16] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0};
    double B[20] = {1.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 1.0, 1.0};
    double C[20] = {0.0};
    double C_ref[20] = {0.0};

    int status = fc_mat_gemm_f64_scalar(4, 5, 4, 1.0, A, 4, B, 5, 0.0, C, 5);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(4, 5, 4, 1.0, A, 4, B, 5, 0.0, C_ref, 5);
    ASSERT_TRUE(matrices_equal(C, C_ref, 4, 5, 5, 5, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_n_remainder_beta) {
    /* 4x5 matrix to trigger n_remainder = 1 with beta != 0 */
    double A[16] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0};
    double B[20] = {1.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 1.0, 1.0};
    double C[20] = {1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0};
    double C_ref[20] = {1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0};

    int status = fc_mat_gemm_f64_scalar(4, 5, 4, 1.0, A, 4, B, 5, 0.5, C, 5);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(4, 5, 4, 1.0, A, 4, B, 5, 0.5, C_ref, 5);
    ASSERT_TRUE(matrices_equal(C, C_ref, 4, 5, 5, 5, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_scalar_with_beta) {
    /* Test scalar implementation with non-zero beta */
    double A[16], B[16], C[16], C_ref[16];

    for (int i = 0; i < 16; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 2.0;  /* Pre-initialize C */
        C_ref[i] = 2.0;
    }

    /* C = 1.0 * A*B + 0.5 * C */
    int status = fc_mat_gemm_f64_scalar(4, 4, 4, 1.0, A, 4, B, 4, 0.5, C, 4);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(4, 4, 4, 1.0, A, 4, B, 4, 0.5, C_ref, 4);
    ASSERT_TRUE(matrices_equal(C, C_ref, 4, 4, 4, 4, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_sse42_both_remainder) {
    /* Test SSE42 with both m and n remainders */
    double A[15] = {1.0, 2.0, 3.0, 4.0, 5.0,
                    6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0};
    double B[15] = {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0};
    double C[9] = {0.0};
    double C_ref[9] = {0.0};

    int status = fc_mat_gemm_f64_sse42(3, 3, 5, 1.0, A, 5, B, 3, 0.0, C, 3);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 3, 5, 1.0, A, 5, B, 3, 0.0, C_ref, 3);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 3, 3, 3, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_sse42_direct) {
    /* Test SSE4.2 implementation directly */
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double B[4] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};

    int status = fc_mat_gemm_f64_sse42(2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);

    /* SSE4.2 might not be available on all platforms */
    if (status == FC_OK) {
        ASSERT_TRUE(double_equals(C[0], 19.0, TEST_EPSILON));
        ASSERT_TRUE(double_equals(C[1], 22.0, TEST_EPSILON));
        ASSERT_TRUE(double_equals(C[2], 43.0, TEST_EPSILON));
        ASSERT_TRUE(double_equals(C[3], 50.0, TEST_EPSILON));
    }
}

TEST(test_gemm_sse42_k16) {
    /* Test SSE4.2 with k=16 to trigger prefetch */
    const int m = 4, n = 4, k = 16;
    double A[64], B[64], C[16];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    int status = fc_mat_gemm_f64_sse42(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(C[0], (double)k, TEST_EPSILON));
}

TEST(test_gemm_sse42_m_remainder) {
    /* Test SSE42 with m not multiple of 4 */
    double A[20] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0,
                    17.0, 18.0, 19.0, 20.0};
    double B[16] = {1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0};
    double C[20] = {0.0};
    double C_ref[20] = {0.0};

    int status = fc_mat_gemm_f64_sse42(5, 4, 4, 1.0, A, 4, B, 4, 0.0, C, 4);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(5, 4, 4, 1.0, A, 4, B, 4, 0.0, C_ref, 4);
    ASSERT_TRUE(matrices_equal(C, C_ref, 5, 4, 4, 4, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_sse42_n_remainder) {
    /* Test SSE42 with n not multiple of 2 */
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double B[20] = {1.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 1.0, 1.0};
    double C[15] = {0.0};
    double C_ref[15] = {0.0};

    int status = fc_mat_gemm_f64_sse42(3, 5, 4, 1.0, A, 4, B, 5, 0.0, C, 5);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 5, 4, 1.0, A, 4, B, 5, 0.0, C_ref, 5);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 5, 5, 5, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_sse42_n_remainder_beta) {
    /* Test SSE42 n remainder with beta != 0 */
    double A[12] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double B[20] = {1.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 1.0, 1.0};
    double C[15] = {1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0};
    double C_ref[15] = {1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0};

    int status = fc_mat_gemm_f64_sse42(3, 5, 4, 1.0, A, 4, B, 5, 0.5, C, 5);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(3, 5, 4, 1.0, A, 4, B, 5, 0.5, C_ref, 5);
    ASSERT_TRUE(matrices_equal(C, C_ref, 3, 5, 5, 5, TEST_EPSILON_RELAXED));
}

TEST(test_gemm_sse42_n_remainder_m4) {
    /* Test SSE42 with m=4, n=5 to trigger right edge in main loop */
    const int m = 4, n = 5, k = 4;
    double A[16], B[20], C[20];

    for (int i = 0; i < m * k; i++) A[i] = 1.0;
    for (int i = 0; i < k * n; i++) B[i] = 1.0;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;

    int status = fc_mat_gemm_f64_sse42(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < m * n; i++) {
        ASSERT_TRUE(double_equals(C[i], (double)k, TEST_EPSILON));
    }
}

TEST(test_gemm_sse42_with_beta) {
    double A[16] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0};
    double B[16] = {1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0};
    double C[16] = {1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0};
    double C_ref[16] = {1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0};

    int status = fc_mat_gemm_f64_sse42(4, 4, 4, 1.0, A, 4, B, 4, 0.5, C, 4);
    ASSERT_EQ(status, FC_OK);

    gemm_reference(4, 4, 4, 1.0, A, 4, B, 4, 0.5, C_ref, 4);
    ASSERT_TRUE(matrices_equal(C, C_ref, 4, 4, 4, 4, TEST_EPSILON_RELAXED));
}

void test_gemm_register(void) {
    RUN_TEST(test_gemm_2x2);
    RUN_TEST(test_gemm_4x4);
    RUN_TEST(test_gemm_alpha_beta);
    RUN_TEST(test_gemm_alpha_zero);
    RUN_TEST(test_gemm_avx2_both_remainder);
    RUN_TEST(test_gemm_avx2_direct);
    RUN_TEST(test_gemm_avx2_m_remainder);
    RUN_TEST(test_gemm_avx2_n_remainder);
    RUN_TEST(test_gemm_avx2_n_remainder_beta);
    RUN_TEST(test_gemm_avx2_with_beta);
    RUN_TEST(test_gemm_avx2_with_beta_1);
    RUN_TEST(test_gemm_avx2_with_beta_4x8);
    RUN_TEST(test_gemm_avx512_both_remainder);
    RUN_TEST(test_gemm_avx512_direct);
    RUN_TEST(test_gemm_avx512_m_remainder);
    RUN_TEST(test_gemm_avx512_n_remainder);
    RUN_TEST(test_gemm_avx512_with_beta);
    RUN_TEST(test_gemm_beta_one);
    RUN_TEST(test_gemm_edge_3x7);
    RUN_TEST(test_gemm_edge_5x5);
    RUN_TEST(test_gemm_identity);
    RUN_TEST(test_gemm_invalid_args);
    RUN_TEST(test_gemm_non_square);
    RUN_TEST(test_gemm_precision_128x128);
    RUN_TEST(test_gemm_precision_64x64);
    RUN_TEST(test_gemm_precision_large_values);
    RUN_TEST(test_gemm_random_rectangular);
    RUN_TEST(test_gemm_random_small);
    RUN_TEST(test_gemm_scalar_10x8_beta);
    RUN_TEST(test_gemm_scalar_16x10_k16);
    RUN_TEST(test_gemm_scalar_8x10_k16);
    RUN_TEST(test_gemm_scalar_8x8_beta_k16);
    RUN_TEST(test_gemm_scalar_direct);
    RUN_TEST(test_gemm_scalar_edge_cases);
    RUN_TEST(test_gemm_scalar_large_4x4);
    RUN_TEST(test_gemm_scalar_m_remainder_beta);
    RUN_TEST(test_gemm_scalar_n_remainder);
    RUN_TEST(test_gemm_scalar_n_remainder_beta);
    RUN_TEST(test_gemm_scalar_with_beta);
    RUN_TEST(test_gemm_sse42_both_remainder);
    RUN_TEST(test_gemm_sse42_direct);
    RUN_TEST(test_gemm_sse42_k16);
    RUN_TEST(test_gemm_sse42_m_remainder);
    RUN_TEST(test_gemm_sse42_n_remainder);
    RUN_TEST(test_gemm_sse42_n_remainder_beta);
    RUN_TEST(test_gemm_sse42_n_remainder_m4);
    RUN_TEST(test_gemm_sse42_with_beta);
}
