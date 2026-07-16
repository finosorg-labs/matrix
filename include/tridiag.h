/**
 * @file tridiag.h
 * @brief Tridiagonal matrix solver
 *
 * Provides efficient O(n) solver for tridiagonal linear systems using
 * the Thomas algorithm, commonly used in finite difference methods.
 */

#ifndef FC_MATRIX_TRIDIAG_H
#define FC_MATRIX_TRIDIAG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solve tridiagonal system using Thomas algorithm
 *
 * Solves: A*x = d
 * where A is tridiagonal with lower diagonal a, main diagonal b, upper diagonal c
 *
 * The Thomas algorithm is O(n) specialized Gaussian elimination for tridiagonal systems,
 * commonly used in finite difference methods for PDEs.
 *
 * Time complexity: O(n)
 * Space complexity: O(n) for workspace
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]  n     Size of the system (must be > 1)
 * @param[in]  a     Lower diagonal (length n-1), a[0] is ignored
 * @param[in]  b     Main diagonal (length n)
 * @param[in]  c     Upper diagonal (length n-1), c[n-1] is ignored
 * @param[in]  d     Right-hand side vector (length n)
 * @param[out] x     Solution vector (length n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_SINGULAR_MATRIX if matrix is singular
 */
int fc_mat_tridiag_solve_f64(
    int64_t n,
    const double* a,
    const double* b,
    const double* c,
    const double* d,
    double* x
);

/**
 * @brief Solve tridiagonal system with multiple right-hand sides
 *
 * Solves: A*X = D
 * where A is tridiagonal (n×n) and D is n×nrhs
 *
 * More efficient than calling single RHS version multiple times.
 *
 * @param[in]  n     Size of the system (must be > 1)
 * @param[in]  nrhs  Number of right-hand sides (must be > 0)
 * @param[in]  a     Lower diagonal (length n-1)
 * @param[in]  b     Main diagonal (length n)
 * @param[in]  c     Upper diagonal (length n-1)
 * @param[in]  D     Right-hand side matrix (n×nrhs, row-major)
 * @param[in]  ldd   Leading dimension of D (stride, typically nrhs)
 * @param[out] X     Solution matrix (n×nrhs, row-major)
 * @param[in]  ldx   Leading dimension of X (stride, typically nrhs)
 *
 * @return FC_OK on success, error code otherwise
 */
int fc_mat_tridiag_solve_multi_f64(
    int64_t n,
    int64_t nrhs,
    const double* a,
    const double* b,
    const double* c,
    const double* D,
    int64_t ldd,
    double* X,
    int64_t ldx
);

#ifdef __cplusplus
}
#endif

#endif /* FC_MATRIX_TRIDIAG_H */
