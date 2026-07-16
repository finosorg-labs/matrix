/**
 * @file gemm.h
 * @brief General Matrix Multiply (GEMM) operations
 *
 * Provides high-performance matrix multiplication with SIMD acceleration.
 * Automatically selects optimal implementation based on CPU capabilities.
 */

#ifndef FC_GEMM_H
#define FC_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief General Matrix Multiply (GEMM) for double-precision matrices
 *
 * Computes: C = alpha * A * B + beta * C
 * where A is m×k, B is k×n, C is m×n
 *
 * Matrices are stored in row-major order.
 * Automatically selects optimal SIMD implementation based on CPU capabilities.
 *
 * Time complexity: O(m * n * k)
 * Space complexity: O(1) (in-place update of C)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    m      Number of rows in A and C
 * @param[in]    n      Number of columns in B and C
 * @param[in]    k      Number of columns in A and rows in B
 * @param[in]    alpha  Scalar multiplier for A*B
 * @param[in]    A      Matrix A (m×k, row-major)
 * @param[in]    lda    Leading dimension of A (stride, typically k)
 * @param[in]    B      Matrix B (k×n, row-major)
 * @param[in]    ldb    Leading dimension of B (stride, typically n)
 * @param[in]    beta   Scalar multiplier for C
 * @param[inout] C      Matrix C (m×n, row-major), updated in-place
 * @param[in]    ldc    Leading dimension of C (stride, typically n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_mat_gemm_f64(
    int64_t m,
    int64_t n,
    int64_t k,
    double alpha,
    const double* A,
    int64_t lda,
    const double* B,
    int64_t ldb,
    double beta,
    double* C,
    int64_t ldc
);

/**
 * @brief Extended GEMM with transpose support for double-precision matrices
 *
 * Computes: C = alpha * op(A) * op(B) + beta * C
 * where op(X) = X if trans='N', or op(X) = X^T if trans='T'
 *
 * This function supports transposing A and/or B without explicit materialization.
 * Useful for operations like A^T*A, A*A^T commonly used in SVD and least squares.
 *
 * Dimension requirements:
 * - If transa='N': A is m×k, lda >= k
 * - If transa='T': A is k×m, lda >= m (logically transposed to m×k)
 * - If transb='N': B is k×n, ldb >= n
 * - If transb='T': B is n×k, ldb >= k (logically transposed to k×n)
 * - C is always m×n, ldc >= n
 *
 * Time complexity: O(m * n * k)
 * Space complexity: O(1) (in-place update of C, no temporary transpose)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    transa Transpose flag for A: 'N' (no transpose) or 'T' (transpose)
 * @param[in]    transb Transpose flag for B: 'N' (no transpose) or 'T' (transpose)
 * @param[in]    m      Number of rows in op(A) and C
 * @param[in]    n      Number of columns in op(B) and C
 * @param[in]    k      Number of columns in op(A) and rows in op(B)
 * @param[in]    alpha  Scalar multiplier for op(A)*op(B)
 * @param[in]    A      Matrix A (row-major storage)
 * @param[in]    lda    Leading dimension of A
 * @param[in]    B      Matrix B (row-major storage)
 * @param[in]    ldb    Leading dimension of B
 * @param[in]    beta   Scalar multiplier for C
 * @param[inout] C      Matrix C (m×n, row-major), updated in-place
 * @param[in]    ldc    Leading dimension of C
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_mat_gemm_f64_ex(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    double alpha,
    const double* A,
    int64_t lda,
    const double* B,
    int64_t ldb,
    double beta,
    double* C,
    int64_t ldc
);

#ifdef __cplusplus
}
#endif

#endif /* FC_GEMM_H */
