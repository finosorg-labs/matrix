/**
 * @file bench_matrix.c
 * @brief Matrix module benchmark entry point
 *
 * This file serves as the main benchmark registration point for the matrix module.
 * Individual benchmark modules are in separate files:
 * - bench_vector_ops.c: Vector operations benchmarks
 * - bench_gemm.c: GEMM benchmarks
 * - bench_gemv.c: GEMV benchmarks
 * - bench_transpose.c: Transpose benchmarks
 * - bench_decompose.c: Decomposition benchmarks (LU/QR/Cholesky)
 * - bench_solve.c: Linear solver and inverse benchmarks
 * - bench_tridiag.c: Tridiagonal solver benchmarks
 */

#include "bench_framework.h"
#include <simd_detect.h>
#include <stdio.h>

/* External benchmark functions from sub-modules */
extern void bench_vector_ops_run(void);
extern void bench_gemm_run(void);
extern void bench_gemv_run(void);
extern void bench_transpose_run(void);
extern void bench_decompose_run(void);
extern void bench_solve_run(void);
extern void bench_tridiag_run(void);

/* Entry point for matrix benchmarks */
void bench_matrix_run(void) {
    printf("\n");
    printf("============================================================\n");
    printf("  Matrix Module Performance Benchmarks\n");
    printf("  SIMD level: %s\n", fc_simd_level_string(fc_get_simd_level()));
    printf("============================================================\n");

    /* Run all sub-module benchmarks */
    bench_vector_ops_run();
    bench_gemm_run();
    bench_gemv_run();
    bench_transpose_run();
    bench_decompose_run();
    bench_solve_run();
    bench_tridiag_run();

    printf("\n");
    printf("============================================================\n");
    printf("  Matrix benchmarks complete\n");
    printf("============================================================\n");
}
