/**
 * @file matrix_internal.h
 * @brief Internal matrix implementation functions for testing
 *
 * This header exposes internal implementation details for unit testing purposes.
 * These functions should NOT be used by external code.
 * Only include this header when FC_ENABLE_INTERNAL_TESTS is defined.
 */

#ifndef FC_MATRIX_INTERNAL_H
#define FC_MATRIX_INTERNAL_H

#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef FC_ENABLE_INTERNAL_TESTS

/**
 * @brief Internal GEMM implementations (for testing)
 *
 * These functions are exposed for unit testing individual SIMD variants.
 * Production code should use fc_mat_gemm_f64() which automatically dispatches.
 */

int fc_mat_gemm_f64_scalar(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc
);

int fc_mat_gemm_f64_sse42(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc
);

int fc_mat_gemm_f64_avx2(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc
);

int fc_mat_gemm_f64_avx512(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc
);

/* GEMV internal implementations */
void fc_mat_gemv_f64_scalar(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    double beta,
    double* y
);

void fc_mat_gemv_f64_sse42(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    double beta,
    double* y
);

void fc_mat_gemv_f64_avx2(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    double beta,
    double* y
);

void fc_mat_gemv_f64_avx512(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    double beta,
    double* y
);

/* Transpose internal implementations */
void transpose_blocked(int rows, int cols, const double* src, int ld_src, double* dst, int ld_dst);

#endif /* FC_ENABLE_INTERNAL_TESTS */

#ifdef __cplusplus
}
#endif

#endif /* FC_MATRIX_INTERNAL_H */
