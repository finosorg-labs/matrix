/**
 * @file test_matrix.c
 * @brief Matrix module test entry point
 *
 * This file serves as the main test registration point for the matrix module.
 * Individual test modules are in separate files:
 * - test_vector_ops.c: Vector operations tests
 * - test_gemm.c: GEMM tests
 * - test_gemv.c: GEMV tests
 * - test_transpose.c: Transpose tests
 * - test_decompose.c: Decomposition tests (LU/QR/Cholesky)
 * - test_solve.c: Linear solver and inverse tests
 * - test_tridiag.c: Tridiagonal solver tests
 * - test_precision.c: Precision tests
 */

#include "test_framework.h"

/* External test registration functions from sub-modules */
extern void test_vector_ops_register(void);
extern void test_gemm_register(void);
extern void test_gemv_register(void);
extern void test_transpose_register(void);
extern void test_decompose_register(void);
extern void test_solve_register(void);
extern void register_tridiag_tests(void);
extern void register_precision_tests(void);

void register_matrix_tests(void) {
    /* Register all sub-module tests */
    test_vector_ops_register();
    test_gemm_register();
    test_gemv_register();
    test_transpose_register();
    test_decompose_register();
    test_solve_register();
    register_tridiag_tests();
    register_precision_tests();
}
