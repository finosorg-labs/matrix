/**
 * @file decompose.h
 * @brief Matrix decomposition operations
 *
 * Provides LU, QR, and Cholesky decompositions.
 */

#ifndef FC_DECOMPOSE_H
#define FC_DECOMPOSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LU decomposition with partial pivoting
 *
 * Computes: P * A = L * U
 * where A is n×n, L is lower triangular with unit diagonal,
 * U is upper triangular, P is permutation matrix
 *
 * The L and U factors are stored in-place in matrix A:
 * - Upper triangle (including diagonal) contains U
 * - Strict lower triangle contains L (diagonal of L is implicitly 1)
 *
 * Time complexity: O(n^3)
 * Space complexity: O(n) for pivot array
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    n     Matrix dimension (must be > 0)
 * @param[inout] A     Input matrix (n×n, row-major), overwritten with L and U
 * @param[in]    lda   Leading dimension of A (stride, typically n)
 * @param[out]   ipiv  Pivot indices array (length n), ipiv[i] = row swapped with row i
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_SINGULAR_MATRIX if matrix is singular
 */
int fc_mat_lu_decompose_f64(int64_t n, double* A, int64_t lda, int64_t* ipiv);

/**
 * @brief QR decomposition using blocked Householder algorithm
 *
 * Computes: A = Q * R
 * where A is m×n (m >= n), Q is m×m orthogonal, R is m×n upper triangular
 *
 * Implementation uses blocked algorithm with WY representation for better cache
 * performance. Processes matrix in blocks (block_size=32), accumulating Householder
 * transformations and applying them via GEMM operations. This provides 2-3× better
 * performance than classical sequential Householder for medium to large matrices.
 *
 * The R factor is stored in the upper triangle of A.
 * The Householder vectors are stored in the lower triangle of A.
 * The tau array contains scaling factors for the Householder reflectors.
 *
 * Time complexity: O(m * n^2)
 * Space complexity: O(block_size^2 + m*block_size) for workspace
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    m     Number of rows in A (must be >= n)
 * @param[in]    n     Number of columns in A (must be > 0)
 * @param[inout] A     Input matrix (m×n, row-major), overwritten with Q and R factors
 * @param[in]    lda   Leading dimension of A (stride, typically n)
 * @param[out]   tau   Householder scaling factors (length n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_DIMENSION_MISMATCH if m < n, FC_ERR_OUT_OF_MEMORY if allocation fails
 */
int fc_mat_qr_decompose_f64(int64_t m, int64_t n, double* A, int64_t lda, double* tau);

/**
 * @brief Cholesky decomposition for symmetric positive definite matrices
 *
 * Computes: A = L * L^T
 * where A is n×n symmetric positive definite, L is lower triangular
 *
 * Only the lower triangle of A is accessed.
 * The L factor is stored in the lower triangle of A (upper triangle unchanged).
 *
 * Time complexity: O(n^3)
 * Space complexity: O(1) (in-place)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    n     Matrix dimension (must be > 0)
 * @param[inout] A     Input matrix (n×n, row-major), lower triangle overwritten with L
 * @param[in]    lda   Leading dimension of A (stride, typically n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_NOT_POSITIVE_DEF if matrix is not positive definite
 */
int fc_mat_cholesky_decompose_f64(int64_t n, double* A, int64_t lda);

/**
 * @brief Singular Value Decomposition (SVD)
 *
 * Computes: A = U * Sigma * V^T
 * where A is m×n, U is m×m orthogonal, Sigma is m×n diagonal (singular values),
 * V is n×n orthogonal
 *
 * Implementation: Uses eigenvalue decomposition of A^T*A (for m>=n) or A*A^T (for m<n).
 * This approach is simpler and more maintainable than Golub-Kahan bidiagonalization,
 * with acceptable numerical stability for well-conditioned matrices.
 *
 * Singular values are returned in descending order: s[0] >= s[1] >= ... >= s[min(m,n)-1] >= 0
 *
 * Time complexity: O(min(m,n)^3) dominated by eigenvalue decomposition
 * Space complexity: O(m*m + n*n) for U and V matrices
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    m      Number of rows in A (must be > 0)
 * @param[in]    n      Number of columns in A (must be > 0)
 * @param[inout] A      Input matrix (m×n, row-major), destroyed on output
 * @param[in]    lda    Leading dimension of A (stride, typically n)
 * @param[out]   s      Singular values (length min(m,n)), in descending order
 * @param[out]   U      Left singular vectors (m×m, row-major), can be NULL if not needed
 * @param[in]    ldu    Leading dimension of U (stride, typically m)
 * @param[out]   VT     Right singular vectors transposed (n×n, row-major), can be NULL if not
 * needed
 * @param[in]    ldvt   Leading dimension of VT (stride, typically n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_OUT_OF_MEMORY if memory allocation fails
 */
int fc_mat_svd_f64(
    int64_t m,
    int64_t n,
    double* A,
    int64_t lda,
    double* s,
    double* U,
    int64_t ldu,
    double* VT,
    int64_t ldvt
);

/**
 * @brief Eigenvalue decomposition for symmetric matrices
 *
 * Computes: A = Q * Lambda * Q^T
 * where A is n×n symmetric, Q is n×n orthogonal (eigenvectors),
 * Lambda is n×n diagonal (eigenvalues)
 *
 * Uses the QR algorithm with implicit shifts.
 * Only the lower triangle of A is accessed.
 * Eigenvalues are returned in ascending order.
 *
 * Time complexity: O(n^3)
 * Space complexity: O(n^2) for eigenvector matrix
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    n      Matrix dimension (must be > 0)
 * @param[inout] A      Input symmetric matrix (n×n, row-major), destroyed on output
 * @param[in]    lda    Leading dimension of A (stride, typically n)
 * @param[out]   w      Eigenvalues (length n), in ascending order
 * @param[out]   Q      Eigenvectors (n×n, row-major), column i is eigenvector for w[i]
 * @param[in]    ldq    Leading dimension of Q (stride, typically n)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_CONVERGENCE if iteration fails to converge
 */
int fc_mat_eig_sym_f64(int64_t n, double* A, int64_t lda, double* w, double* Q, int64_t ldq);

/**
 * @brief Apply Q^T to a vector using Householder reflectors from QR decomposition
 *
 * Applies the transpose of the orthogonal matrix Q from QR decomposition to a vector.
 * The Q matrix is represented implicitly by Householder vectors stored in the lower
 * triangle of A (from fc_mat_qr_decompose_f64) and the tau array.
 *
 * This function computes: b := Q^T * b
 * where Q = H_1 * H_2 * ... * H_k (product of Householder reflectors)
 *
 * Used in least squares solvers to transform the right-hand side vector before
 * solving the triangular system.
 *
 * Time complexity: O(m*n)
 * Space complexity: O(1) (in-place operation)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    m      Number of rows in original matrix (must be >= n)
 * @param[in]    n      Number of Householder reflectors (must be > 0)
 * @param[in]    A      Matrix containing Householder vectors from QR decomposition
 *                      (m×n, row-major), lower triangle contains v_k with implicit unit diagonal
 * @param[in]    lda    Leading dimension of A (stride, typically n)
 * @param[in]    tau    Householder scaling factors from QR decomposition (length n)
 * @param[inout] b      Input vector (length m), overwritten with Q^T * b
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_mat_apply_qt_vector_f64(
    int64_t m,
    int64_t n,
    const double* A,
    int64_t lda,
    const double* tau,
    double* b
);

/**
 * @brief Solve upper triangular system Rx = b
 *
 * Solves the upper triangular linear system R*x = b using backward substitution.
 * The solution x overwrites the input vector b.
 *
 * The matrix R must be non-singular (all diagonal elements non-zero).
 * Only the upper triangle of R is accessed.
 *
 * Used in least squares solvers after QR decomposition to solve R*x = Q^T*b.
 *
 * Time complexity: O(n^2)
 * Space complexity: O(1) (in-place operation)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]    n      Dimension of the system (must be > 0)
 * @param[in]    R      Upper triangular matrix (n×n, row-major)
 * @param[in]    ldr    Leading dimension of R (stride, typically n)
 * @param[inout] b      Input right-hand side (length n), overwritten with solution x
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid,
 *         FC_ERR_SINGULAR_MATRIX if diagonal element is zero
 */
int fc_mat_solve_triangular_upper_f64(int64_t n, const double* R, int64_t ldr, double* b);

#ifdef __cplusplus
}
#endif

#endif /* FC_DECOMPOSE_H */
