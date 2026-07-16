package matrix

import (
	"math"
	"testing"

)

func TestLUDecompose(t *testing.T) {

	// Test 3x3 matrix
	A := []float64{
		2, 1, 1,
		4, 3, 3,
		8, 7, 9,
	}
	ipiv := make([]int64, 3)

	err := LUDecompose(3, A, 3, ipiv)
	if err != nil {
		t.Fatalf("LUDecompose() failed: %v", err)
	}

	// Check that diagonal elements are non-zero
	if math.Abs(A[0]) < epsilon {
		t.Errorf("A[0,0] should be non-zero")
	}
	if math.Abs(A[4]) < epsilon {
		t.Errorf("A[1,1] should be non-zero")
	}
	if math.Abs(A[8]) < epsilon {
		t.Errorf("A[2,2] should be non-zero")
	}
}

func TestQRDecompose(t *testing.T) {

	// Test 3x3 matrix
	A := []float64{
		12, -51, 4,
		6, 167, -68,
		-4, 24, -41,
	}
	tau := make([]float64, 3)

	err := QRDecompose(3, 3, A, 3, tau)
	if err != nil {
		t.Fatalf("QRDecompose() failed: %v", err)
	}

	// Check that R diagonal elements are non-zero
	if math.Abs(A[0]) < epsilon {
		t.Errorf("R[0,0] should be non-zero")
	}
	if math.Abs(A[4]) < epsilon {
		t.Errorf("R[1,1] should be non-zero")
	}
	if math.Abs(A[8]) < epsilon {
		t.Errorf("R[2,2] should be non-zero")
	}
}

func TestCholeskyDecompose(t *testing.T) {

	// Test symmetric positive definite matrix
	A := []float64{
		4, 12, -16,
		12, 37, -43,
		-16, -43, 98,
	}

	err := CholeskyDecompose(3, A, 3)
	if err != nil {
		t.Fatalf("CholeskyDecompose() failed: %v", err)
	}

	// Check that L diagonal elements are positive
	if A[0] <= 0 {
		t.Errorf("L[0,0] should be positive, got %f", A[0])
	}
	if A[4] <= 0 {
		t.Errorf("L[1,1] should be positive, got %f", A[4])
	}
	if A[8] <= 0 {
		t.Errorf("L[2,2] should be positive, got %f", A[8])
	}
}

func TestSVD(t *testing.T) {

	// Test 3x3 matrix
	A := []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	s := make([]float64, 3)
	U := make([]float64, 9)
	VT := make([]float64, 9)

	err := SVD(3, 3, A, 3, s, U, 3, VT, 3)
	if err != nil {
		t.Fatalf("SVD() failed: %v", err)
	}

	// Check singular values are non-negative and in descending order
	if s[0] < 0 {
		t.Errorf("Singular value s[0] should be non-negative, got %f", s[0])
	}
	if s[1] < 0 {
		t.Errorf("Singular value s[1] should be non-negative, got %f", s[1])
	}
	if s[2] < 0 {
		t.Errorf("Singular value s[2] should be non-negative, got %f", s[2])
	}
	if s[0] < s[1] {
		t.Errorf("Singular values should be in descending order: s[0]=%f < s[1]=%f", s[0], s[1])
	}
	if s[1] < s[2] {
		t.Errorf("Singular values should be in descending order: s[1]=%f < s[2]=%f", s[1], s[2])
	}
}

func TestSVDIdentity(t *testing.T) {

	// Identity matrix should have singular values all equal to 1
	A := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	s := make([]float64, 3)

	err := SVD(3, 3, A, 3, s, nil, 3, nil, 3)
	if err != nil {
		t.Fatalf("SVD() failed: %v", err)
	}

	// Check singular values are approximately 1
	for i := 0; i < 3; i++ {
		if math.Abs(s[i]-1.0) > 1e-10 {
			t.Errorf("Singular value s[%d] should be 1.0, got %f", i, s[i])
		}
	}
}

func TestEigenSymmetric(t *testing.T) {

	// Symmetric matrix
	A := []float64{
		4, 1, 2,
		1, 5, 3,
		2, 3, 6,
	}
	w := make([]float64, 3)
	Q := make([]float64, 9)

	err := EigenSymmetric(3, A, 3, w, Q, 3)
	if err != nil {
		t.Fatalf("EigenSymmetric() failed: %v", err)
	}

	// Check eigenvalues are in ascending order
	if w[0] > w[1] {
		t.Errorf("Eigenvalues should be in ascending order: w[0]=%f > w[1]=%f", w[0], w[1])
	}
	if w[1] > w[2] {
		t.Errorf("Eigenvalues should be in ascending order: w[1]=%f > w[2]=%f", w[1], w[2])
	}
}

func TestEigenSymmetricIdentity(t *testing.T) {

	// Identity matrix - all eigenvalues should be 1
	A := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	w := make([]float64, 3)
	Q := make([]float64, 9)

	err := EigenSymmetric(3, A, 3, w, Q, 3)
	if err != nil {
		t.Fatalf("EigenSymmetric() failed: %v", err)
	}

	// Check eigenvalues are approximately 1
	for i := 0; i < 3; i++ {
		if math.Abs(w[i]-1.0) > 1e-10 {
			t.Errorf("Eigenvalue w[%d] should be 1.0, got %f", i, w[i])
		}
	}
}
