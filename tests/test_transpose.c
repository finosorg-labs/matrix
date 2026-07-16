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

/*
 * Vector operations tests
*/


TEST(test_transpose_8x8) {
    /* Test 8x8 matrix */
    double src[64];
    double dst[64];

    for (int i = 0; i < 64; i++) {
        src[i] = (double)i;
    }

    int status = fc_mat_transpose_f64(8, 8, src, 8, dst, 8);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            ASSERT_TRUE(double_equals(dst[j * 8 + i], src[i * 8 + j], TEST_EPSILON));
        }
    }
}

TEST(test_transpose_basic) {
    double src[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    double dst[6];

    int status = fc_mat_transpose_f64(2, 3, src, 3, dst, 2);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(dst[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], 4.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[2], 2.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[3], 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[4], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[5], 6.0, TEST_EPSILON));
}

TEST(test_transpose_blocked_direct) {
    /* Direct test of blocked transpose implementation */
    const int rows = 64, cols = 64;
    double* src = (double*)malloc(rows * cols * sizeof(double));
    double* dst = (double*)malloc(rows * cols * sizeof(double));

    for (int i = 0; i < rows * cols; i++) {
        src[i] = (double)i;
    }

    transpose_blocked(rows, cols, src, cols, dst, rows);

    /* Verify transpose */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ASSERT_TRUE(double_equals(dst[j * rows + i], src[i * cols + j], TEST_EPSILON));
        }
    }

    free(src);
    free(dst);
}

TEST(test_transpose_blocked_non_multiple) {
    /* Test blocked transpose with dimensions not multiple of block size */
    const int rows = 50, cols = 70;
    double* src = (double*)malloc(rows * cols * sizeof(double));
    double* dst = (double*)malloc(rows * cols * sizeof(double));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            src[i * cols + j] = (double)(i * cols + j);
        }
    }

    transpose_blocked(rows, cols, src, cols, dst, rows);

    /* Verify transpose */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ASSERT_TRUE(double_equals(dst[j * rows + i], src[i * cols + j], TEST_EPSILON));
        }
    }

    free(src);
    free(dst);
}

TEST(test_transpose_invalid_args) {
    double src[] = {1.0, 2.0, 3.0, 4.0};
    double dst[4];

    ASSERT_EQ(fc_mat_transpose_f64(0, 2, src, 2, dst, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_transpose_f64(2, 0, src, 2, dst, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_transpose_f64(2, 2, NULL, 2, dst, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_transpose_f64(2, 2, src, 2, NULL, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_transpose_f64(2, 2, src, 1, dst, 2), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_transpose_f64(2, 2, src, 2, dst, 1), FC_ERR_INVALID_ARG);
}

TEST(test_transpose_large) {
    /* Test large matrix to trigger blocked transpose */
    const int rows = 128, cols = 128;
    double src[16384], dst[16384];

    for (int i = 0; i < rows * cols; i++) src[i] = (double)i;

    int status = fc_mat_transpose_f64(rows, cols, src, cols, dst, rows);
    ASSERT_EQ(status, FC_OK);

    /* Verify a few elements */
    ASSERT_TRUE(double_equals(dst[0], 0.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], (double)cols, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[rows], 1.0, TEST_EPSILON));
}

TEST(test_transpose_large_blocked) {
    /* Test large matrix to trigger blocking path (>= 32x32) */
    const int rows = 64;
    const int cols = 64;
    double* src = (double*)malloc(rows * cols * sizeof(double));
    double* dst = (double*)malloc(rows * cols * sizeof(double));

    for (int i = 0; i < rows * cols; i++) {
        src[i] = (double)i;
    }

    int status = fc_mat_transpose_f64(rows, cols, src, cols, dst, rows);
    ASSERT_EQ(status, FC_OK);

    /* Verify a few elements */
    ASSERT_TRUE(double_equals(dst[0], src[0], TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], src[cols], TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[rows], src[1], TEST_EPSILON));

    free(src);
    free(dst);
}

TEST(test_transpose_large_for_blocked) {
    /* Test large matrix (>= 32) to trigger blocked path */
    const int rows = 64;
    const int cols = 64;
    double* src = (double*)malloc(rows * cols * sizeof(double));
    double* dst = (double*)malloc(rows * cols * sizeof(double));

    for (int i = 0; i < rows * cols; i++) {
        src[i] = (double)i;
    }

    int status = fc_mat_transpose_f64(rows, cols, src, cols, dst, rows);
    ASSERT_EQ(status, FC_OK);

    /* Verify a few elements */
    ASSERT_TRUE(double_equals(dst[0], src[0], TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[rows], src[1], TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], src[cols], TEST_EPSILON));

    free(src);
    free(dst);
}

TEST(test_transpose_medium_blocked) {
    /* Test medium matrix to trigger blocking without AVX2 4x4 optimization */
    const int rows = 40;
    const int cols = 40;
    double* src = (double*)malloc(rows * cols * sizeof(double));
    double* dst = (double*)malloc(rows * cols * sizeof(double));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            src[i * cols + j] = (double)(i * cols + j);
        }
    }

    int status = fc_mat_transpose_f64(rows, cols, src, cols, dst, rows);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ASSERT_TRUE(double_equals(dst[j * rows + i], src[i * cols + j], TEST_EPSILON));
        }
    }

    free(src);
    free(dst);
}

TEST(test_transpose_non_multiple_of_4) {
    /* Test dimensions not multiple of 4 to exercise edge handling */
    const int rows = 7;
    const int cols = 9;
    double src[63];
    double dst[63];

    for (int i = 0; i < rows * cols; i++) {
        src[i] = (double)i;
    }

    int status = fc_mat_transpose_f64(rows, cols, src, cols, dst, rows);
    ASSERT_EQ(status, FC_OK);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ASSERT_TRUE(double_equals(dst[j * rows + i], src[i * cols + j], TEST_EPSILON));
        }
    }
}

TEST(test_transpose_non_square) {
    double src[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    double dst[6];

    int status = fc_mat_transpose_f64(3, 2, src, 2, dst, 3);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(dst[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], 3.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[2], 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[3], 2.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[4], 4.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[5], 6.0, TEST_EPSILON));
}

TEST(test_transpose_random) {
    srand(789);

    const int rows = 15, cols = 12;
    double src[180], dst[180], dst_ref[180];

    fill_random_matrix(src, rows, cols, cols, -100.0, 100.0);

    int status = fc_mat_transpose_f64(rows, cols, src, cols, dst, rows);
    ASSERT_EQ(status, FC_OK);

    /* Reference transpose */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst_ref[j * rows + i] = src[i * cols + j];
        }
    }

    /* Compare */
    ASSERT_TRUE(matrices_equal(dst, dst_ref, cols, rows, rows, rows, TEST_EPSILON));
}

TEST(test_transpose_small_3x3) {
    /* Test small matrix that uses scalar path */
    double src[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double dst[9];

    int status = fc_mat_transpose_f64(3, 3, src, 3, dst, 3);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(dst[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], 4.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[2], 7.0, TEST_EPSILON));
}

TEST(test_transpose_square_4x4) {
    /* Test 4x4 to potentially trigger AVX2 4x4 optimization */
    double src[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    double dst[16];

    int status = fc_mat_transpose_f64(4, 4, src, 4, dst, 4);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(dst[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[1], 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[2], 9.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(dst[3], 13.0, TEST_EPSILON));
}

void test_transpose_register(void) {
    RUN_TEST(test_transpose_8x8);
    RUN_TEST(test_transpose_basic);
    RUN_TEST(test_transpose_blocked_direct);
    RUN_TEST(test_transpose_blocked_non_multiple);
    RUN_TEST(test_transpose_invalid_args);
    RUN_TEST(test_transpose_large);
    RUN_TEST(test_transpose_large_blocked);
    RUN_TEST(test_transpose_large_for_blocked);
    RUN_TEST(test_transpose_medium_blocked);
    RUN_TEST(test_transpose_non_multiple_of_4);
    RUN_TEST(test_transpose_non_square);
    RUN_TEST(test_transpose_random);
    RUN_TEST(test_transpose_small_3x3);
    RUN_TEST(test_transpose_square_4x4);
}
