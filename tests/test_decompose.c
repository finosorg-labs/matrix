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
#include <inttypes.h>
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


TEST(test_cholesky_decompose_basic) {
    /* Symmetric positive definite matrix */
    double A[] = {
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0
    };
    double A_original[] = {
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0
    };

    int status = fc_mat_cholesky_decompose_f64(3, A, 3);
    ASSERT_EQ(status, FC_OK);

    /* Verify L * L^T = original A
     * After decomposition, A contains L in lower triangle
     * Reconstruct and verify against original */
    double reconstructed[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double sum = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++) {
                double L_ik = (k <= i) ? A[i * 3 + k] : 0.0;
                double L_jk = (k <= j) ? A[j * 3 + k] : 0.0;
                sum += L_ik * L_jk;
            }
            reconstructed[i * 3 + j] = sum;
        }
    }

    /* Verify reconstruction matches original */
    for (int i = 0; i < 9; i++) {
        ASSERT_TRUE(fabs(reconstructed[i] - A_original[i]) < 1e-10);
    }
}

TEST(test_cholesky_decompose_identity) {
    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    int status = fc_mat_cholesky_decompose_f64(3, A, 3);
    ASSERT_EQ(status, FC_OK);

    /* Identity should remain identity */
    ASSERT_TRUE(double_equals(A[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(A[4], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(A[8], 1.0, TEST_EPSILON));
}

TEST(test_cholesky_decompose_invalid_args) {
    double A[9];

    ASSERT_EQ(fc_mat_cholesky_decompose_f64(3, NULL, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_cholesky_decompose_f64(0, A, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_cholesky_decompose_f64(-1, A, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_cholesky_decompose_f64(3, A, 2), FC_ERR_INVALID_ARG);
}

TEST(test_cholesky_decompose_not_positive_def) {
    /* Not positive definite (negative eigenvalue) */
    double A[] = {
        1.0, 2.0, 3.0,
        2.0, 1.0, 4.0,
        3.0, 4.0, 1.0
    };

    int status = fc_mat_cholesky_decompose_f64(3, A, 3);
    ASSERT_EQ(status, FC_ERR_NOT_POSITIVE_DEF);
}

TEST(test_lu_decompose_basic) {
    /* Test 3x3 matrix */
    double A[] = {
        2.0, 1.0, 1.0,
        4.0, 3.0, 3.0,
        8.0, 7.0, 9.0
    };
    int64_t ipiv[3];

    int status = fc_mat_lu_decompose_f64(3, A, 3, ipiv);
    ASSERT_EQ(status, FC_OK);

    /* Check that diagonal elements are non-zero */
    ASSERT_TRUE(fabs(A[0]) > TEST_EPSILON);
    ASSERT_TRUE(fabs(A[4]) > TEST_EPSILON);
    ASSERT_TRUE(fabs(A[8]) > TEST_EPSILON);
}

TEST(test_lu_decompose_identity) {
    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    int64_t ipiv[3];

    int status = fc_mat_lu_decompose_f64(3, A, 3, ipiv);
    ASSERT_EQ(status, FC_OK);

    /* Identity should remain identity */
    ASSERT_TRUE(double_equals(A[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(A[4], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(A[8], 1.0, TEST_EPSILON));
}

TEST(test_lu_decompose_invalid_args) {
    double A[9];
    int64_t ipiv[3];

    ASSERT_EQ(fc_mat_lu_decompose_f64(3, NULL, 3, ipiv), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_lu_decompose_f64(3, A, 3, NULL), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_lu_decompose_f64(0, A, 3, ipiv), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_lu_decompose_f64(-1, A, 3, ipiv), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_lu_decompose_f64(3, A, 2, ipiv), FC_ERR_INVALID_ARG);
}

TEST(test_lu_decompose_singular) {
    /* Singular matrix (row 2 = row 1) */
    double A[] = {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    int64_t ipiv[3];

    int status = fc_mat_lu_decompose_f64(3, A, 3, ipiv);
    ASSERT_EQ(status, FC_ERR_SINGULAR_MATRIX);
}

TEST(test_qr_decompose_basic) {
    /* Test 3x2 matrix with known result (column-major)
     * A = [[1, 1],
     *      [1, 2],
     *      [1, 3]]
     * This should give solution x=[1,2] for b=[3,5,7]
     */
    double A[] = {
        1.0, 1.0, 1.0,  /* Column 0 */
        1.0, 2.0, 3.0   /* Column 1 */
    };
    double tau[2];

    int status = fc_mat_qr_decompose_f64(3, 2, A, 3, tau);
    ASSERT_EQ(status, FC_OK);

    /* R diagonal should be non-zero */
    ASSERT_TRUE(fabs(A[0]) > TEST_EPSILON);
    ASSERT_TRUE(fabs(A[3]) > TEST_EPSILON);

    /* Verify by solving a known least squares problem */
    double b[] = {3.0, 5.0, 7.0};
    status = fc_mat_apply_qt_vector_f64(3, 2, A, 3, tau, b);
    ASSERT_EQ(status, FC_OK);

    status = fc_mat_solve_triangular_upper_f64(2, A, 3, b);
    ASSERT_EQ(status, FC_OK);

    /* The solution should be x=[1,2] */
    ASSERT_TRUE(fabs(b[0] - 1.0) < 1e-10);
    ASSERT_TRUE(fabs(b[1] - 2.0) < 1e-10);
}

TEST(test_qr_decompose_invalid_args) {
    double A[12];
    double tau[3];

    ASSERT_EQ(fc_mat_qr_decompose_f64(4, 3, NULL, 3, tau), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_qr_decompose_f64(4, 3, A, 3, NULL), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_qr_decompose_f64(0, 3, A, 3, tau), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_qr_decompose_f64(4, 0, A, 3, tau), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_qr_decompose_f64(2, 3, A, 3, tau), FC_ERR_DIMENSION_MISMATCH);
    ASSERT_EQ(fc_mat_qr_decompose_f64(4, 3, A, 2, tau), FC_ERR_INVALID_ARG);
}

TEST(test_qr_decompose_tall) {
    /* Test 4x2 tall matrix (column-major)
     * Using perfect linear data: y = 1 + 2*x
     * Data points: (1,3), (2,5), (3,7), (4,9)
     * Expected exact solution: a=1.0, b=2.0
     */
    double A[] = {
        1.0, 1.0, 1.0, 1.0,  /* Column 0: intercept */
        1.0, 2.0, 3.0, 4.0   /* Column 1: x values */
    };
    double tau[2];

    int status = fc_mat_qr_decompose_f64(4, 2, A, 4, tau);
    ASSERT_EQ(status, FC_OK);

    /* Verify by solving least squares */
    double b[] = {3.0, 5.0, 7.0, 9.0};
    status = fc_mat_apply_qt_vector_f64(4, 2, A, 4, tau, b);
    ASSERT_EQ(status, FC_OK);

    status = fc_mat_solve_triangular_upper_f64(2, A, 4, b);
    ASSERT_EQ(status, FC_OK);

    /* Expected exact solution: a=1.0, b=2.0 */
    ASSERT_TRUE(fabs(b[0] - 1.0) < 1e-10);
    ASSERT_TRUE(fabs(b[1] - 2.0) < 1e-10);
}

TEST(test_svd_basic) {
    /* Test 3x3 matrix */
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double s[3];
    double U[9];
    double VT[9];

    int status = fc_mat_svd_f64(3, 3, A, 3, s, U, 3, VT, 3);
    ASSERT_EQ(status, FC_OK);

    /* Singular values should be non-negative and in descending order */
    ASSERT_TRUE(s[0] >= 0.0);
    ASSERT_TRUE(s[1] >= 0.0);
    ASSERT_TRUE(s[2] >= 0.0);
    ASSERT_TRUE(s[0] >= s[1]);
    ASSERT_TRUE(s[1] >= s[2]);
}

TEST(test_svd_identity) {
    /* Identity matrix should have singular values all equal to 1 */
    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    double s[3];
    double U[9];
    double VT[9];

    int status = fc_mat_svd_f64(3, 3, A, 3, s, U, 3, VT, 3);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(s[0], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(s[1], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(s[2], 1.0, TEST_EPSILON_RELAXED));
}

TEST(test_svd_diagonal) {
    /* Diagonal matrix - singular values should be absolute values of diagonal */
    double A[] = {
        3.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0
    };
    double s[3];
    double U[9];
    double VT[9];

    int status = fc_mat_svd_f64(3, 3, A, 3, s, U, 3, VT, 3);
    ASSERT_EQ(status, FC_OK);

    /* Singular values should be in descending order */
    ASSERT_TRUE(double_equals(s[0], 3.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(s[1], 2.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(s[2], 1.0, TEST_EPSILON_RELAXED));
}

TEST(test_svd_invalid_args) {
    double A[9], s[3], U[9], VT[9];

    ASSERT_EQ(fc_mat_svd_f64(3, 3, NULL, 3, s, U, 3, VT, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_svd_f64(3, 3, A, 3, NULL, U, 3, VT, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_svd_f64(0, 3, A, 3, s, U, 3, VT, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_svd_f64(3, 0, A, 3, s, U, 3, VT, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_svd_f64(3, 3, A, 2, s, U, 3, VT, 3), FC_ERR_INVALID_ARG);
}

TEST(test_svd_rectangular) {
    /* Test 4x3 tall matrix (column-major order) */
    double A[] = {
        1.0, 4.0, 7.0, 10.0,   /* Column 1 */
        2.0, 5.0, 8.0, 11.0,   /* Column 2 */
        3.0, 6.0, 9.0, 12.0    /* Column 3 */
    };
    double s[3];
    double U[12];  /* U is 4×3 */
    double VT[9];  /* VT is 3×3 */

    int status = fc_mat_svd_f64(4, 3, A, 4, s, U, 4, VT, 3);
    ASSERT_EQ(status, FC_OK);

    /* Check singular values are non-negative and sorted */
    ASSERT_TRUE(s[0] >= s[1]);
    ASSERT_TRUE(s[1] >= s[2]);
}

TEST(test_svd_wide_matrix) {
    /* Test 3x4 wide matrix (m < n) (column-major order) */
    double A[] = {
        1.0, 5.0, 9.0,    /* Column 1 */
        2.0, 6.0, 10.0,   /* Column 2 */
        3.0, 7.0, 11.0,   /* Column 3 */
        4.0, 8.0, 12.0    /* Column 4 */
    };
    double s[3];
    double U[9];    /* U is 3×3 */
    double VT[12];  /* VT is 3×4 */

    int status = fc_mat_svd_f64(3, 4, A, 3, s, U, 3, VT, 3);
    ASSERT_EQ(status, FC_OK);

    /* Check singular values are non-negative and sorted */
    ASSERT_TRUE(s[0] >= 0.0);
    ASSERT_TRUE(s[1] >= 0.0);
    ASSERT_TRUE(s[2] >= 0.0);
    ASSERT_TRUE(s[0] >= s[1]);
    ASSERT_TRUE(s[1] >= s[2]);

    /* TODO: Reconstruction test requires GEMM with transpose support
     * or manual transpose of VT before multiplication */
}

TEST(test_svd_reconstruction) {
    /* Test A = U*Σ*V^T reconstruction (column-major order) */
    double A[] = {
        1.0, 4.0, 7.0,  /* Column 1 */
        2.0, 5.0, 8.0,  /* Column 2 */
        3.0, 6.0, 9.0   /* Column 3 */
    };
    double s[3];
    double U[9];
    double VT[9];

    int status = fc_mat_svd_f64(3, 3, A, 3, s, U, 3, VT, 3);
    ASSERT_EQ(status, FC_OK);

    /* Reconstruct A = U*Σ*V^T (all matrices in column-major) */
    /* Compute (U*Σ)*V^T manually for column-major matrices */
    double A_reconstructed[9];
    for (int j = 0; j < 3; j++) {  /* for each column of result */
        for (int i = 0; i < 3; i++) {  /* for each row of result */
            double sum = 0.0;
            for (int k = 0; k < 3; k++) {
                sum += U[k * 3 + i] * s[k] * VT[j * 3 + k];  /* A = U * Σ * V^T */
            }
            A_reconstructed[j * 3 + i] = sum;
        }
    }

    /* Debug output (print in row-major for readability) */
    printf("\n  Original A:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f]\n", A[i+0*3], A[i+1*3], A[i+2*3]);
    }
    printf("  Singular values: [%.6f, %.6f, %.6f]\n", s[0], s[1], s[2]);
    printf("  Reconstructed A:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f]\n",
               A_reconstructed[i+0*3], A_reconstructed[i+1*3], A_reconstructed[i+2*3]);
    }
    printf("  Differences:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.2e, %.2e, %.2e]\n",
               fabs(A[i+0*3] - A_reconstructed[i+0*3]),
               fabs(A[i+1*3] - A_reconstructed[i+1*3]),
               fabs(A[i+2*3] - A_reconstructed[i+2*3]));
    }

    /* Verify reconstruction */
    for (int i = 0; i < 9; i++) {
        ASSERT_TRUE(double_equals(A[i], A_reconstructed[i], TEST_EPSILON_RELAXED));
    }
}

TEST(test_eig_sym_basic) {
    /* Symmetric matrix (column-major) */
    double A[] = {
        4.0, 1.0, 2.0,  /* Column 1 */
        1.0, 5.0, 3.0,  /* Column 2 */
        2.0, 3.0, 6.0   /* Column 3 */
    };
    double w[3];
    double Q[9];

    int status = fc_mat_eig_sym_f64(3, A, 3, w, Q, 3);
    ASSERT_EQ(status, FC_OK);

    /* Eigenvalues should be in ascending order */
    ASSERT_TRUE(w[0] <= w[1]);
    ASSERT_TRUE(w[1] <= w[2]);
}

TEST(test_eig_sym_identity) {
    /* Identity matrix - all eigenvalues should be 1 (column-major) */
    double A[] = {
        1.0, 0.0, 0.0,  /* Column 1 */
        0.0, 1.0, 0.0,  /* Column 2 */
        0.0, 0.0, 1.0   /* Column 3 */
    };
    double w[3];
    double Q[9];

    int status = fc_mat_eig_sym_f64(3, A, 3, w, Q, 3);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(w[0], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(w[1], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(w[2], 1.0, TEST_EPSILON_RELAXED));
}

TEST(test_eig_sym_diagonal) {
    /* Diagonal matrix - eigenvalues are diagonal elements (column-major) */
    double A[] = {
        3.0, 0.0, 0.0,  /* Column 1 */
        0.0, 2.0, 0.0,  /* Column 2 */
        0.0, 0.0, 1.0   /* Column 3 */
    };
    double w[3];
    double Q[9];

    int status = fc_mat_eig_sym_f64(3, A, 3, w, Q, 3);
    ASSERT_EQ(status, FC_OK);

    /* Eigenvalues should be sorted in ascending order */
    ASSERT_TRUE(double_equals(w[0], 1.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(w[1], 2.0, TEST_EPSILON_RELAXED));
    ASSERT_TRUE(double_equals(w[2], 3.0, TEST_EPSILON_RELAXED));
}

TEST(test_eig_sym_invalid_args) {
    double A[9], w[3], Q[9];

    ASSERT_EQ(fc_mat_eig_sym_f64(3, NULL, 3, w, Q, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_eig_sym_f64(3, A, 3, NULL, Q, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_eig_sym_f64(3, A, 3, w, NULL, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_eig_sym_f64(0, A, 3, w, Q, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_eig_sym_f64(-1, A, 3, w, Q, 3), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_eig_sym_f64(3, A, 2, w, Q, 3), FC_ERR_INVALID_ARG);
}

TEST(test_eig_sym_orthogonality) {
    /* Test Q^T * Q = I (orthogonality) */
    double A[] = {
        4.0, 1.0, 2.0,
        1.0, 5.0, 3.0,
        2.0, 3.0, 6.0
    };
    double w[3];
    double Q[9];

    int status = fc_mat_eig_sym_f64(3, A, 3, w, Q, 3);
    ASSERT_EQ(status, FC_OK);

    /* Compute Q^T * Q */
    double QTQ[9];
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 3; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < 3; k++) {
                sum += Q[k * 3 + i] * Q[k * 3 + j];
            }
            QTQ[i * 3 + j] = sum;
        }
    }

    /* Q^T * Q should be identity */
    for (int64_t i = 0; i < 3; i++) {
        for (int64_t j = 0; j < 3; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_TRUE(double_equals(QTQ[i * 3 + j], expected, TEST_EPSILON_RELAXED));
        }
    }
}

TEST(test_eig_sym_reconstruction) {
    /* Test A * v = lambda * v for each eigenpair */
    double A_orig[] = {
        4.0, 1.0, 2.0,  /* Column 1 */
        1.0, 5.0, 3.0,  /* Column 2 */
        2.0, 3.0, 6.0   /* Column 3 (column-major) */
    };
    double A[] = {
        4.0, 1.0, 2.0,
        1.0, 5.0, 3.0,
        2.0, 3.0, 6.0
    };
    double w[3];
    double Q[9];

    int status = fc_mat_eig_sym_f64(3, A, 3, w, Q, 3);
    ASSERT_EQ(status, FC_OK);

    /* Debug: print eigenvalues and eigenvectors (print as rows for readability) */
    printf("\n  Eigenvalues: [%.6f, %.6f, %.6f]\n", w[0], w[1], w[2]);
    printf("  Q matrix:\n");
    for (int64_t i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f]\n", Q[0*3+i], Q[1*3+i], Q[2*3+i]);
    }

    /* For each eigenvector, verify A * v = lambda * v */
    for (int64_t i = 0; i < 3; i++) {
        double v[3];
        double Av[3];

        /* Extract eigenvector i (column i of Q in column-major) */
        for (int64_t j = 0; j < 3; j++) {
            v[j] = Q[i * 3 + j];  /* Column-major: Q[j,i] = Q[i*3 + j] */
        }

        /* Compute A * v (A_orig is column-major) */
        for (int64_t j = 0; j < 3; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < 3; k++) {
                sum += A_orig[k * 3 + j] * v[k];  /* Column-major: A[j,k] = A[k*3 + j] */
            }
            Av[j] = sum;
        }

        /* Debug: print verification */
        printf("  Eigenpair %" PRId64 ": lambda=%.6f\n", i, w[i]);
        for (int64_t j = 0; j < 3; j++) {
            printf("    A*v[%" PRId64 "]=%.6f, lambda*v[%" PRId64 "]=%.6f, diff=%.2e\n",
                   j, Av[j], j, w[i] * v[j], fabs(Av[j] - w[i] * v[j]));
        }

        /* Verify A * v = lambda * v */
        for (int64_t j = 0; j < 3; j++) {
            ASSERT_TRUE(double_equals(Av[j], w[i] * v[j], TEST_EPSILON_RELAXED));
        }
    }
}

TEST(test_apply_qt_vector_basic) {
    /* Test Q^T * b where Q is from QR decomposition
     * Use a simple matrix with known QR decomposition result */
    /* Matrix [[1,1],[1,2],[1,3]] in column-major format:
     * Column 0: [1, 1, 1]
     * Column 1: [1, 2, 3] */
    double A[] = {
        1.0, 1.0, 1.0,  /* Column 0 */
        1.0, 2.0, 3.0   /* Column 1 */
    };
    double tau[2];
    double b[] = {3.0, 5.0, 7.0};

    /* Perform QR decomposition (column-major, lda=3) */
    int status = fc_mat_qr_decompose_f64(3, 2, A, 3, tau);
    ASSERT_EQ(status, FC_OK);

    /* Apply Q^T to b */
    status = fc_mat_apply_qt_vector_f64(3, 2, A, 3, tau, b);
    ASSERT_EQ(status, FC_OK);

    /* For this specific matrix and vector, we can verify the result
     * by solving the least squares problem manually:
     * The matrix [[1,1],[1,2],[1,3]] with y=[3,5,7] has exact solution beta=[1,2]
     * After Q^T, the first two elements should equal R*beta
     * R = [[-1.732, -3.464], [0, -1.633]] (approximately)
     * Q^T*y should have first two elements that when solved give beta=[1,2]
     */

    /* Verify: solve R*x = Q^T*y for first 2 elements */
    double x[2];
    x[0] = b[0];
    x[1] = b[1];

    printf("  Q^T*y = [%.6f, %.6f, %.6f]\n", b[0], b[1], b[2]);

    status = fc_mat_solve_triangular_upper_f64(2, A, 3, x);
    ASSERT_EQ(status, FC_OK);

    printf("  Solution x = [%.6f, %.6f]\n", x[0], x[1]);
    printf("  Expected: [1.0, 2.0]\n");

    /* The solution should be approximately [1.0, 2.0] */
    ASSERT_TRUE(fabs(x[0] - 1.0) < 1e-6);
    ASSERT_TRUE(fabs(x[1] - 2.0) < 1e-6);
}

TEST(test_apply_qt_vector_identity) {
    /* Test with identity matrix (special case) */
    double A[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    double tau[2];
    double b[] = {3.0, 4.0};

    int status = fc_mat_qr_decompose_f64(2, 2, A, 2, tau);
    ASSERT_EQ(status, FC_OK);

    status = fc_mat_apply_qt_vector_f64(2, 2, A, 2, tau, b);
    ASSERT_EQ(status, FC_OK);

    /* For near-identity, result should be close to original */
    ASSERT_TRUE(fabs(b[0]) > 0.0);
    ASSERT_TRUE(fabs(b[1]) > 0.0);
}

TEST(test_apply_qt_vector_invalid_args) {
    double A[6], tau[2] = {0}, b[3];

    ASSERT_EQ(fc_mat_apply_qt_vector_f64(3, 2, NULL, 2, tau, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_apply_qt_vector_f64(3, 2, A, 2, NULL, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_apply_qt_vector_f64(3, 2, A, 2, tau, NULL), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_apply_qt_vector_f64(0, 2, A, 2, tau, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_apply_qt_vector_f64(3, 0, A, 2, tau, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_apply_qt_vector_f64(2, 3, A, 2, tau, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_apply_qt_vector_f64(3, 2, A, 1, tau, b), FC_ERR_INVALID_ARG);
}

TEST(test_solve_triangular_upper_basic) {
    /* Simple upper triangular system:
     * 2x + 3y = 8  =>  x = 1, y = 2
     *      y  = 2
     * Matrix in column-major format:
     * Column 0: [2.0, 0.0]
     * Column 1: [3.0, 1.0]
     */
    double R[] = {
        2.0, 0.0,  /* Column 0 */
        3.0, 1.0   /* Column 1 */
    };
    double b[] = {8.0, 2.0};

    int status = fc_mat_solve_triangular_upper_f64(2, R, 2, b);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(b[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[1], 2.0, TEST_EPSILON));
}

TEST(test_solve_triangular_upper_identity) {
    /* Identity matrix: solution should equal right-hand side */
    double R[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    double b[] = {5.0, 7.0, 9.0};

    int status = fc_mat_solve_triangular_upper_f64(3, R, 3, b);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(b[0], 5.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[1], 7.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[2], 9.0, TEST_EPSILON));
}

TEST(test_solve_triangular_upper_3x3) {
    /* 3x3 upper triangular system:
     * 2x + 3y + 1z = 18  =>  x = 1, y = 2, z = 10
     *      4y + 2z = 28
     *           z  = 10
     * Matrix in column-major format:
     * Column 0: [2.0, 0.0, 0.0]
     * Column 1: [3.0, 4.0, 0.0]
     * Column 2: [1.0, 2.0, 1.0]
     */
    double R[] = {
        2.0, 0.0, 0.0,  /* Column 0 */
        3.0, 4.0, 0.0,  /* Column 1 */
        1.0, 2.0, 1.0   /* Column 2 */
    };
    double b[] = {18.0, 28.0, 10.0};

    int status = fc_mat_solve_triangular_upper_f64(3, R, 3, b);
    ASSERT_EQ(status, FC_OK);

    ASSERT_TRUE(double_equals(b[0], 1.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[1], 2.0, TEST_EPSILON));
    ASSERT_TRUE(double_equals(b[2], 10.0, TEST_EPSILON));
}

TEST(test_solve_triangular_upper_invalid_args) {
    double R[4], b[2];

    ASSERT_EQ(fc_mat_solve_triangular_upper_f64(2, NULL, 2, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_triangular_upper_f64(2, R, 2, NULL), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_triangular_upper_f64(0, R, 2, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_triangular_upper_f64(-1, R, 2, b), FC_ERR_INVALID_ARG);
    ASSERT_EQ(fc_mat_solve_triangular_upper_f64(2, R, 1, b), FC_ERR_INVALID_ARG);
}

TEST(test_solve_triangular_upper_singular) {
    /* Singular matrix (zero diagonal element) */
    double R[] = {
        2.0, 3.0,
        0.0, 0.0
    };
    double b[] = {8.0, 2.0};

    int status = fc_mat_solve_triangular_upper_f64(2, R, 2, b);
    ASSERT_EQ(status, FC_ERR_SINGULAR_MATRIX);
}

TEST(test_qr_least_squares_integration) {
    /* Integration test: solve least squares using QR decomposition
     * System: A*x = b where A is 4x2 (overdetermined)
     * Using perfect linear data: y = 1 + 2*x
     * Data points: (1,3), (2,5), (3,7), (4,9)
     * Expected exact solution: a=1.0, b=2.0
     */
    double A[] = {
        1.0, 1.0, 1.0, 1.0,  /* Column 0: intercept */
        1.0, 2.0, 3.0, 4.0   /* Column 1: x values */
    };
    double b[] = {3.0, 5.0, 7.0, 9.0};  /* y values */
    double tau[2];

    /* Step 1: QR decomposition (column-major, lda=4) */
    int status = fc_mat_qr_decompose_f64(4, 2, A, 4, tau);
    ASSERT_EQ(status, FC_OK);

    /* Step 2: Apply Q^T to b */
    status = fc_mat_apply_qt_vector_f64(4, 2, A, 4, tau, b);
    ASSERT_EQ(status, FC_OK);

    /* Debug: print intermediate values */
    printf("  After Q^T * b: [%.6f, %.6f, %.6f, %.6f]\n", b[0], b[1], b[2], b[3]);
    printf("  R matrix (upper triangle):\n");
    printf("    [%.6f, %.6f]\n", A[0], A[4]);
    printf("    [%.6f, %.6f]\n", A[1], A[5]);

    /* Step 3: Solve R*x = Q^T*b (use first 2 elements of b) */
    status = fc_mat_solve_triangular_upper_f64(2, A, 4, b);
    ASSERT_EQ(status, FC_OK);

    /* Debug: print solution */
    printf("  Solution: x = [%.6f, %.6f]\n", b[0], b[1]);

    /* Verify solution completed without error and produces finite values */
    ASSERT_TRUE(isfinite(b[0]));
    ASSERT_TRUE(isfinite(b[1]));

    /* Verify the expected exact solution: a=1.0, b=2.0 */
    ASSERT_TRUE(fabs(b[0] - 1.0) < 1e-10);
    ASSERT_TRUE(fabs(b[1] - 2.0) < 1e-10);
}

void test_decompose_register(void) {
    RUN_TEST(test_cholesky_decompose_basic);
    RUN_TEST(test_cholesky_decompose_identity);
    RUN_TEST(test_cholesky_decompose_invalid_args);
    RUN_TEST(test_cholesky_decompose_not_positive_def);
    RUN_TEST(test_lu_decompose_basic);
    RUN_TEST(test_lu_decompose_identity);
    RUN_TEST(test_lu_decompose_invalid_args);
    RUN_TEST(test_lu_decompose_singular);
    RUN_TEST(test_qr_decompose_basic);
    RUN_TEST(test_qr_decompose_invalid_args);
    RUN_TEST(test_qr_decompose_tall);
    RUN_TEST(test_svd_basic);
    RUN_TEST(test_svd_identity);
    RUN_TEST(test_svd_diagonal);
    RUN_TEST(test_svd_invalid_args);
    RUN_TEST(test_svd_rectangular);
    RUN_TEST(test_svd_wide_matrix);
    RUN_TEST(test_svd_reconstruction);
    RUN_TEST(test_eig_sym_basic);
    RUN_TEST(test_eig_sym_identity);
    RUN_TEST(test_eig_sym_diagonal);
    RUN_TEST(test_eig_sym_invalid_args);
    RUN_TEST(test_eig_sym_orthogonality);
    RUN_TEST(test_eig_sym_reconstruction);
    RUN_TEST(test_apply_qt_vector_basic);
    RUN_TEST(test_apply_qt_vector_identity);
    RUN_TEST(test_apply_qt_vector_invalid_args);
    RUN_TEST(test_solve_triangular_upper_basic);
    RUN_TEST(test_solve_triangular_upper_identity);
    RUN_TEST(test_solve_triangular_upper_3x3);
    RUN_TEST(test_solve_triangular_upper_invalid_args);
    RUN_TEST(test_solve_triangular_upper_singular);
    RUN_TEST(test_qr_least_squares_integration);
}
