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

/*
 * Vector operations tests
*/


TEST(test_vec_distance_l1_basic) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 6.0, 3.0};
    double result;

    int status = fc_vec_distance_l1_f64(x, y, 3, &result);
    ASSERT_EQ(status, FC_OK);

    /* Expected: |4-1| + |6-2| + |3-3| = 3 + 4 + 0 = 7.0 */
    ASSERT_TRUE(double_equals(result, 7.0, TEST_EPSILON));
}

TEST(test_vec_distance_l1_invalid_args) {
    double x[] = {1.0, 2.0};
    double y[] = {3.0, 4.0};
    double result;

    ASSERT_EQ(fc_vec_distance_l1_f64(NULL, y, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_distance_l1_f64(x, NULL, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_distance_l1_f64(x, y, 0, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_distance_l1_f64(x, y, 2, NULL), FC_ERR_INVALID_ARG);
}

TEST(test_vec_distance_l1_size5) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double result;

    int status = fc_vec_distance_l1_f64(x, y, 5, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 12.0, TEST_EPSILON));
}

TEST(test_vec_distance_l1_size7) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double y[] = {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double result;

    int status = fc_vec_distance_l1_f64(x, y, 7, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 24.0, TEST_EPSILON));
}

TEST(test_vec_distance_l1_zero) {
    double x[] = {5.0, 10.0};
    double result;

    int status = fc_vec_distance_l1_f64(x, x, 2, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 0.0, TEST_EPSILON));
}

TEST(test_vec_distance_l2_basic) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 6.0, 3.0};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 3, &result);
    ASSERT_EQ(status, FC_OK);

    /* Expected: sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9 + 16 + 0) = 5.0 */
    ASSERT_TRUE(double_equals(result, 5.0, TEST_EPSILON));
}

TEST(test_vec_distance_l2_extreme) {
    /* Use values that will overflow without scaling */
    double x[] = {1e200, 2e200, 3e200};
    double y[] = {1e200, 2e200, 4e200};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(result > 0.0);
}

TEST(test_vec_distance_l2_invalid_args) {
    double x[] = {1.0, 2.0};
    double y[] = {3.0, 4.0};
    double result;

    ASSERT_EQ(fc_vec_distance_l2_f64(NULL, y, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_distance_l2_f64(x, NULL, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_distance_l2_f64(x, y, 0, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_distance_l2_f64(x, y, 2, NULL), FC_ERR_INVALID_ARG);
}

TEST(test_vec_distance_l2_large) {
    double x[] = {1e150, 2e150, 3e150};
    double y[] = {1e150, 2e150, 4e150};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(result > 0.0);
}

TEST(test_vec_distance_l2_mixed) {
    /* First element sets scale, second element triggers ratio calculation */
    double x[] = {1e100, 1e50};
    double y[] = {0.0, 0.0};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 2, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(result > 0.0);
}

TEST(test_vec_distance_l2_single) {
    /* Test single element path */
    double x[] = {10.0};
    double y[] = {3.0};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 1, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 7.0, TEST_EPSILON));
}

TEST(test_vec_distance_l2_size5) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 5, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(result > 6.3 && result < 6.4);
}

TEST(test_vec_distance_l2_size7) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double y[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0};
    double result;

    int status = fc_vec_distance_l2_f64(x, y, 7, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 1.0, TEST_EPSILON));
}

TEST(test_vec_distance_l2_zero) {
    /* Test identical vectors */
    double x[] = {1.0, 2.0, 3.0};
    double result;

    int status = fc_vec_distance_l2_f64(x, x, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 0.0, TEST_EPSILON));
}

TEST(test_vec_dot_basic) {
    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {5.0, 6.0, 7.0, 8.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 4, &result);
    ASSERT_EQ(status, FC_OK);

    /* Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70 */
    ASSERT_TRUE(double_equals(result, 70.0, TEST_EPSILON));
}

TEST(test_vec_dot_invalid_args) {
    double x[] = {1.0, 2.0};
    double result;

    /* NULL pointer */
    ASSERT_EQ(fc_vec_dot_f64(NULL, x, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_dot_f64(x, NULL, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_dot_f64(x, x, 2, NULL), FC_ERR_INVALID_ARG);

    /* Invalid length */
    ASSERT_EQ(fc_vec_dot_f64(x, x, 0, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_dot_f64(x, x, -1, &result), FC_ERR_INVALID_ARG);
}

TEST(test_vec_dot_negative) {
    double x[] = {-1.0, 2.0, -3.0, 4.0};
    double y[] = {5.0, -6.0, 7.0, -8.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 4, &result);
    ASSERT_EQ(status, FC_OK);

    /* Expected: -1*5 + 2*(-6) + (-3)*7 + 4*(-8) = -5 - 12 - 21 - 32 = -70 */
    ASSERT_TRUE(double_equals(result, -70.0, TEST_EPSILON));
}

TEST(test_vec_dot_single) {
    /* Test single element */
    double x[] = {3.5};
    double y[] = {2.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 1, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 7.0, TEST_EPSILON));
}

TEST(test_vec_dot_size3) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 32.0, TEST_EPSILON));
}

TEST(test_vec_dot_size5) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 5, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 15.0, TEST_EPSILON));
}

TEST(test_vec_dot_size7) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double y[] = {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 7, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 84.0, TEST_EPSILON));
}

TEST(test_vec_dot_size9) {
    double x[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double y[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 9, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 45.0, TEST_EPSILON));
}

TEST(test_vec_dot_zero) {
    /* Test zero vectors */
    double x[] = {0.0, 0.0, 0.0};
    double y[] = {1.0, 2.0, 3.0};
    double result;

    int status = fc_vec_dot_f64(x, y, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 0.0, TEST_EPSILON));
}

TEST(test_vec_norm_l1_basic) {
    double x[] = {-1.0, 2.0, -3.0, 4.0};
    double result;

    int status = fc_vec_norm_l1_f64(x, 4, &result);
    ASSERT_EQ(status, FC_OK);

    /* Expected: |−1| + |2| + |−3| + |4| = 1 + 2 + 3 + 4 = 10 */
    ASSERT_TRUE(double_equals(result, 10.0, TEST_EPSILON));
}

TEST(test_vec_norm_l1_invalid_args) {
    double x[] = {1.0, 2.0};
    double result;

    ASSERT_EQ(fc_vec_norm_l1_f64(NULL, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_norm_l1_f64(x, 0, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_norm_l1_f64(x, 2, NULL), FC_ERR_INVALID_ARG);
}

TEST(test_vec_norm_l1_large_values) {
    double x[] = {1e100, 2e100, 3e100, 4e100};
    double result;

    int status = fc_vec_norm_l1_f64(x, 4, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(result > 9.9e100 && result < 10.1e100);
}

TEST(test_vec_norm_l1_size3) {
    double x[] = {1.0, -2.0, 3.0};
    double result;

    int status = fc_vec_norm_l1_f64(x, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 6.0, TEST_EPSILON));
}

TEST(test_vec_norm_l1_size5) {
    double x[] = {1.0, -2.0, 3.0, -4.0, 5.0};
    double result;

    int status = fc_vec_norm_l1_f64(x, 5, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 15.0, TEST_EPSILON));
}

TEST(test_vec_norm_l1_size7) {
    double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double result;

    int status = fc_vec_norm_l1_f64(x, 7, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 7.0, TEST_EPSILON));
}

TEST(test_vec_norm_l1_zero) {
    double x[] = {0.0, 0.0};
    double result;

    int status = fc_vec_norm_l1_f64(x, 2, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 0.0, TEST_EPSILON));
}

TEST(test_vec_norm_l2_basic) {
    double x[] = {3.0, 4.0};
    double result;

    int status = fc_vec_norm_l2_f64(x, 2, &result);
    ASSERT_EQ(status, FC_OK);

    /* Expected: sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5.0 */
    ASSERT_TRUE(double_equals(result, 5.0, TEST_EPSILON));
}

TEST(test_vec_norm_l2_invalid_args) {
    double x[] = {1.0, 2.0};
    double result;

    ASSERT_EQ(fc_vec_norm_l2_f64(NULL, 2, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_norm_l2_f64(x, 0, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_norm_l2_f64(x, -1, &result), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_vec_norm_l2_f64(x, 2, NULL), FC_ERR_INVALID_ARG);
}

TEST(test_vec_norm_l2_large) {
    /* Test overflow protection with large values */
    double x[] = {1e150, 1e150, 1e150};
    double result;

    int status = fc_vec_norm_l2_f64(x, 3, &result);
    ASSERT_EQ(status, FC_OK);
    /* Expected: sqrt(3) * 1e150 ≈ 1.732e150 */
    ASSERT_TRUE(result > 1.7e150 && result < 1.8e150);
}

TEST(test_vec_norm_l2_single) {
    double x[] = {-7.5};
    double result;

    int status = fc_vec_norm_l2_f64(x, 1, &result);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(result, 7.5, TEST_EPSILON));
}

TEST(test_vec_norm_l2_size3) {
    double x[] = {1.0, 2.0, 2.0};
    double result;

    int status = fc_vec_norm_l2_f64(x, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 3.0, TEST_EPSILON));
}

TEST(test_vec_norm_l2_size5) {
    double x[] = {1.0, 2.0, 2.0, 0.0, 0.0};
    double result;

    int status = fc_vec_norm_l2_f64(x, 5, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 3.0, TEST_EPSILON));
}

TEST(test_vec_norm_l2_size7) {
    double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double result;

    int status = fc_vec_norm_l2_f64(x, 7, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, sqrt(7.0), TEST_EPSILON));
}

TEST(test_vec_norm_l2_small) {
    /* Test underflow protection with small values */
    double x[] = {1e-150, 1e-150, 1e-150};
    double result;

    int status = fc_vec_norm_l2_f64(x, 3, &result);
    ASSERT_EQ(status, FC_OK);
    /* Expected: sqrt(3) * 1e-150 ≈ 1.732e-150 */
    ASSERT_TRUE(result > 1.7e-150 && result < 1.8e-150);
}

TEST(test_vec_norm_l2_zero) {
    /* Test zero vector */
    double x[] = {0.0, 0.0, 0.0};
    double result;

    int status = fc_vec_norm_l2_f64(x, 3, &result);
    ASSERT_EQ(status, FC_OK);
    ASSERT_TRUE(double_equals(result, 0.0, TEST_EPSILON));
}

void test_vector_ops_register(void) {
    RUN_TEST(test_vec_distance_l1_basic);
    RUN_TEST(test_vec_distance_l1_invalid_args);
    RUN_TEST(test_vec_distance_l1_size5);
    RUN_TEST(test_vec_distance_l1_size7);
    RUN_TEST(test_vec_distance_l1_zero);
    RUN_TEST(test_vec_distance_l2_basic);
    RUN_TEST(test_vec_distance_l2_extreme);
    RUN_TEST(test_vec_distance_l2_invalid_args);
    RUN_TEST(test_vec_distance_l2_large);
    RUN_TEST(test_vec_distance_l2_mixed);
    RUN_TEST(test_vec_distance_l2_single);
    RUN_TEST(test_vec_distance_l2_size5);
    RUN_TEST(test_vec_distance_l2_size7);
    RUN_TEST(test_vec_distance_l2_zero);
    RUN_TEST(test_vec_dot_basic);
    RUN_TEST(test_vec_dot_invalid_args);
    RUN_TEST(test_vec_dot_negative);
    RUN_TEST(test_vec_dot_single);
    RUN_TEST(test_vec_dot_size3);
    RUN_TEST(test_vec_dot_size5);
    RUN_TEST(test_vec_dot_size7);
    RUN_TEST(test_vec_dot_size9);
    RUN_TEST(test_vec_dot_zero);
    RUN_TEST(test_vec_norm_l1_basic);
    RUN_TEST(test_vec_norm_l1_invalid_args);
    RUN_TEST(test_vec_norm_l1_large_values);
    RUN_TEST(test_vec_norm_l1_size3);
    RUN_TEST(test_vec_norm_l1_size5);
    RUN_TEST(test_vec_norm_l1_size7);
    RUN_TEST(test_vec_norm_l1_zero);
    RUN_TEST(test_vec_norm_l2_basic);
    RUN_TEST(test_vec_norm_l2_invalid_args);
    RUN_TEST(test_vec_norm_l2_large);
    RUN_TEST(test_vec_norm_l2_single);
    RUN_TEST(test_vec_norm_l2_size3);
    RUN_TEST(test_vec_norm_l2_size5);
    RUN_TEST(test_vec_norm_l2_size7);
    RUN_TEST(test_vec_norm_l2_small);
    RUN_TEST(test_vec_norm_l2_zero);
}
