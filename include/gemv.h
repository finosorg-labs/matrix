/**
 * @file gemv.h
 * @brief General Matrix-Vector Multiply (GEMV) operations
 *
 * Provides high-performance matrix-vector multiplication.
 */

#ifndef FC_GEMV_H
#define FC_GEMV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief General Matrix-Vector Multiply (GEMV) for double-precision
 *
 * Computes: y = alpha * A * x + beta * y
 * where A is m×n, x is n×1, y is m×1
 *
 * Matrix A is stored in row-major order.
 *
 * Time complexity: O(m * n)
 * Space complexity: O(1) (in-place update of y)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    m      Number of rows in A
 * @param[in]    n      Number of columns in A
 * @param[in]    alpha  Scalar multiplier for A*x
 * @param[in]    A      Matrix A (m×n, row-major)
 * @param[in]    lda    Leading dimension of A (stride, typically n)
 * @param[in]    x      Vector x (length n)
 * @param[in]    beta   Scalar multiplier for y
 * @param[inout] y      Vector y (length m), updated in-place
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_mat_gemv_f64(
    int64_t m,
    int64_t n,
    double alpha,
    const double* A,
    int64_t lda,
    const double* x,
    double beta,
    double* y
);

#ifdef __cplusplus
}
#endif

#endif /* FC_GEMV_H */
