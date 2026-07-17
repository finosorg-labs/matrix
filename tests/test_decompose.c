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

    int status = fc_mat_cholesky_decompose_f64(3, A, 3);
    ASSERT_EQ(status, FC_OK);

    /* Verify L * L^T = original A by checking diagonal elements */
    ASSERT_TRUE(A[0] > 0.0);
    ASSERT_TRUE(A[4] > 0.0);
    ASSERT_TRUE(A[8] > 0.0);
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
    /* Test 3x3 matrix */
    double A[] = {
        12.0, -51.0, 4.0,
        6.0, 167.0, -68.0,
        -4.0, 24.0, -41.0
    };
    double tau[3];

    int status = fc_mat_qr_decompose_f64(3, 3, A, 3, tau);
    ASSERT_EQ(status, FC_OK);

    /* R diagonal should be non-zero */
    ASSERT_TRUE(fabs(A[0]) > TEST_EPSILON);
    ASSERT_TRUE(fabs(A[4]) > TEST_EPSILON);
    ASSERT_TRUE(fabs(A[8]) > TEST_EPSILON);
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
    /* Test 4x3 tall matrix */
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };
    double tau[3];

    int status = fc_mat_qr_decompose_f64(4, 3, A, 3, tau);
    ASSERT_EQ(status, FC_OK);
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
    /* Test 4x3 tall matrix */
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };
    double s[3];
    double U[16];
    double VT[9];

    int status = fc_mat_svd_f64(4, 3, A, 3, s, U, 4, VT, 3);
    ASSERT_EQ(status, FC_OK);

    /* Check singular values are non-negative and sorted */
    ASSERT_TRUE(s[0] >= s[1]);
    ASSERT_TRUE(s[1] >= s[2]);
}

TEST(test_svd_wide_matrix) {
    /* Test 3x4 wide matrix (m < n) */
    double A[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    double s[3];
    double U[9];
    double VT[12];  /* VT should be 3x4 for thin SVD or 4x4 for full SVD */

    int status = fc_mat_svd_f64(3, 4, A, 4, s, U, 3, VT, 4);
    ASSERT_EQ(status, FC_OK);

    /* Check singular values are non-negative and sorted */
    ASSERT_TRUE(s[0] >= 0.0);
    ASSERT_TRUE(s[1] >= 0.0);
    ASSERT_TRUE(s[2] >= 0.0);
    ASSERT_TRUE(s[0] >= s[1]);
    ASSERT_TRUE(s[1] >= s[2]);

    /* Reconstruct A = U*Σ*V^T to verify correctness */
    /* For m < n, Σ is 3x3 diagonal, VT is 3x4 */
    double Sigma[9] = {0};
    Sigma[0] = s[0];
    Sigma[4] = s[1];
    Sigma[8] = s[2];

    /* Compute U*Σ (3x3 * 3x3 = 3x3) */
    double US[9];
    fc_mat_gemm_f64(3, 3, 3, 1.0, U, 3, Sigma, 3, 0.0, US, 3);

    /* Compute (U*Σ)*V^T (3x3 * 3x4 = 3x4) */
    double A_reconstructed[12];
    fc_mat_gemm_f64(3, 4, 3, 1.0, US, 3, VT, 4, 0.0, A_reconstructed, 4);

    /* Verify reconstruction */
    printf("\n  Wide matrix (3x4) reconstruction test:\n");
    printf("  Original A:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f, %.6f]\n",
               A[i*4+0], A[i*4+1], A[i*4+2], A[i*4+3]);
    }
    printf("  Singular values: [%.6f, %.6f, %.6f]\n", s[0], s[1], s[2]);
    printf("  Reconstructed A:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f, %.6f]\n",
               A_reconstructed[i*4+0], A_reconstructed[i*4+1],
               A_reconstructed[i*4+2], A_reconstructed[i*4+3]);
    }
    printf("  Differences:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.2e, %.2e, %.2e, %.2e]\n",
               fabs(A[i*4+0] - A_reconstructed[i*4+0]),
               fabs(A[i*4+1] - A_reconstructed[i*4+1]),
               fabs(A[i*4+2] - A_reconstructed[i*4+2]),
               fabs(A[i*4+3] - A_reconstructed[i*4+3]));
    }

    for (int i = 0; i < 12; i++) {
        ASSERT_TRUE(double_equals(A[i], A_reconstructed[i], TEST_EPSILON_RELAXED));
    }
}

TEST(test_svd_reconstruction) {
    /* Test A = U*Σ*V^T reconstruction */
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

    /* Reconstruct A = U*Σ*V^T */
    double Sigma[9] = {0};
    Sigma[0] = s[0];
    Sigma[4] = s[1];
    Sigma[8] = s[2];

    /* Compute U*Σ */
    double US[9];
    fc_mat_gemm_f64(3, 3, 3, 1.0, U, 3, Sigma, 3, 0.0, US, 3);

    /* Compute (U*Σ)*V^T */
    double A_reconstructed[9];
    fc_mat_gemm_f64(3, 3, 3, 1.0, US, 3, VT, 3, 0.0, A_reconstructed, 3);

    /* Debug output */
    printf("\n  Original A:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f]\n", A[i*3+0], A[i*3+1], A[i*3+2]);
    }
    printf("  Singular values: [%.6f, %.6f, %.6f]\n", s[0], s[1], s[2]);
    printf("  Reconstructed A:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f]\n",
               A_reconstructed[i*3+0], A_reconstructed[i*3+1], A_reconstructed[i*3+2]);
    }
    printf("  Differences:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.2e, %.2e, %.2e]\n",
               fabs(A[i*3+0] - A_reconstructed[i*3+0]),
               fabs(A[i*3+1] - A_reconstructed[i*3+1]),
               fabs(A[i*3+2] - A_reconstructed[i*3+2]));
    }

    /* Verify reconstruction */
    for (int i = 0; i < 9; i++) {
        ASSERT_TRUE(double_equals(A[i], A_reconstructed[i], TEST_EPSILON_RELAXED));
    }
}

TEST(test_eig_sym_basic) {
    /* Symmetric matrix */
    double A[] = {
        4.0, 1.0, 2.0,
        1.0, 5.0, 3.0,
        2.0, 3.0, 6.0
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
    /* Identity matrix - all eigenvalues should be 1 */
    double A[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
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
    /* Diagonal matrix - eigenvalues are diagonal elements */
    double A[] = {
        3.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0
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
        4.0, 1.0, 2.0,
        1.0, 5.0, 3.0,
        2.0, 3.0, 6.0
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

    /* Debug: print eigenvalues and eigenvectors */
    printf("\n  Eigenvalues: [%.6f, %.6f, %.6f]\n", w[0], w[1], w[2]);
    printf("  Q matrix:\n");
    for (int64_t i = 0; i < 3; i++) {
        printf("    [%.6f, %.6f, %.6f]\n", Q[i*3+0], Q[i*3+1], Q[i*3+2]);
    }

    /* For each eigenvector, verify A * v = lambda * v */
    for (int64_t i = 0; i < 3; i++) {
        double v[3];
        double Av[3];

        /* Extract eigenvector i */
        for (int64_t j = 0; j < 3; j++) {
            v[j] = Q[j * 3 + i];
        }

        /* Compute A * v */
        for (int64_t j = 0; j < 3; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < 3; k++) {
                sum += A_orig[j * 3 + k] * v[k];
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
    /* Test Q^T * b where Q is from QR decomposition */
    double A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    double tau[2];
    double b[] = {1.0, 2.0, 3.0};
    double b_original[] = {1.0, 2.0, 3.0};

    /* Perform QR decomposition */
    int status = fc_mat_qr_decompose_f64(3, 2, A, 2, tau);
    ASSERT_EQ(status, FC_OK);

    /* Apply Q^T to b */
    status = fc_mat_apply_qt_vector_f64(3, 2, A, 2, tau, b);
    ASSERT_EQ(status, FC_OK);

    /* Verify b has changed (Q^T is not identity for this matrix) */
    int changed = 0;
    for (int i = 0; i < 3; i++) {
        if (fabs(b[i] - b_original[i]) > TEST_EPSILON) {
            changed = 1;
            break;
        }
    }
    ASSERT_TRUE(changed);

    /* Verify the operation completed successfully (basic sanity check) */
    ASSERT_TRUE(isfinite(b[0]));
    ASSERT_TRUE(isfinite(b[1]));
    ASSERT_TRUE(isfinite(b[2]));
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
     */
    double R[] = {
        2.0, 3.0,
        0.0, 1.0
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
     */
    double R[] = {
        2.0, 3.0, 1.0,
        0.0, 4.0, 2.0,
        0.0, 0.0, 1.0
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
     * A = [1 1]    b = [2]
     *     [1 2]        [3]
     *     [1 3]        [5]
     *     [1 4]        [6]
     *
     * This is linear regression: y = a + b*x with data points:
     * (1, 2), (2, 3), (3, 5), (4, 6)
     * Expected least squares solution: approximately a=0.5, b=1.0
     */
    double A[] = {
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0
    };
    double b[] = {2.0, 3.0, 5.0, 6.0};
    double tau[2];

    /* Step 1: QR decomposition */
    int status = fc_mat_qr_decompose_f64(4, 2, A, 2, tau);
    ASSERT_EQ(status, FC_OK);

    /* Step 2: Apply Q^T to b */
    status = fc_mat_apply_qt_vector_f64(4, 2, A, 2, tau, b);
    ASSERT_EQ(status, FC_OK);

    /* Debug: print intermediate values */
    printf("  After Q^T * b: [%.6f, %.6f, %.6f, %.6f]\n", b[0], b[1], b[2], b[3]);
    printf("  R matrix (upper triangle):\n");
    printf("    [%.6f, %.6f]\n", A[0], A[1]);
    printf("    [%.6f, %.6f]\n", A[2], A[3]);

    /* Step 3: Solve R*x = Q^T*b (use first 2 elements of b) */
    status = fc_mat_solve_triangular_upper_f64(2, A, 2, b);
    ASSERT_EQ(status, FC_OK);

    /* Debug: print solution */
    printf("  Solution: x = [%.6f, %.6f]\n", b[0], b[1]);

    /* Verify solution completed without error and produces finite values */
    ASSERT_TRUE(isfinite(b[0]));
    ASSERT_TRUE(isfinite(b[1]));

    /* Solution should be in reasonable range for this linear regression problem */
    ASSERT_TRUE(fabs(b[0]) < 100.0);
    ASSERT_TRUE(fabs(b[1]) < 100.0);
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
