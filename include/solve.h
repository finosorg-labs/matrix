/**
 * @file solve.h
 * @brief Linear system solvers
 *
 * Provides linear equation solving and matrix inversion.
 */

#ifndef FC_SOLVE_H
#define FC_SOLVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solve linear system A * X = B using LU decomposition
 *
 * Computes: X = A^(-1) * B
 * where A is n×n, B is n×nrhs, X is n×nrhs
 *
 * The matrix A is overwritten with its LU decomposition.
 * The matrix B is overwritten with the solution X.
 *
 * Time complexity: O(n^3 + n^2 * nrhs)
 * Space complexity: O(n) for pivot array
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    n     Matrix dimension (must be > 0)
 * @param[in]    nrhs  Number of right-hand sides (must be > 0)
 * @param[inout] A     Coefficient matrix (n×n, row-major), overwritten with LU factors
 * @param[in]    lda   Leading dimension of A (stride, typically n)
 * @param[inout] B     Right-hand side matrix (n×nrhs, row-major), overwritten with solution
 * @param[in]    ldb   Leading dimension of B (stride, typically nrhs)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_SINGULAR_MATRIX if matrix is singular
 */
int fc_mat_solve_linear_f64(
    int64_t n,
    int64_t nrhs,
    double* A,
    int64_t lda,
    double* B,
    int64_t ldb
);

/**
 * @brief Compute matrix inverse using LU decomposition
 *
 * Computes: A_inv = A^(-1)
 * where A is n×n non-singular
 *
 * The input matrix A is overwritten with its inverse.
 *
 * Time complexity: O(n^3)
 * Space complexity: O(n^2) for workspace
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    n     Matrix dimension (must be > 0)
 * @param[inout] A     Input matrix (n×n, row-major), overwritten with inverse
 * @param[in]    lda   Leading dimension of A (stride, typically n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_SINGULAR_MATRIX if matrix is singular
 */
int fc_mat_inverse_f64(int64_t n, double* A, int64_t lda);

#ifdef __cplusplus
}
#endif

#endif /* FC_SOLVE_H */
