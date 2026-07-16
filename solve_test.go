package matrix

import (
	"math"
	"testing"

)

func TestSolveLinear(t *testing.T) {

	// Test solving A*x = b
	// A = [[2, 1], [1, 2]], b = [3, 3], solution x = [1, 1]
	A := []float64{
		2, 1,
		1, 2,
	}
	b := []float64{3, 3}

	err := SolveLinear(2, 1, A, 2, b, 1)
	if err != nil {
		t.Fatalf("SolveLinear() failed: %v", err)
	}

	// Check solution
	expected := []float64{1, 1}
	for i := range b {
		if math.Abs(b[i]-expected[i]) > epsilon {
			t.Errorf("x[%d] = %f, want %f", i, b[i], expected[i])
		}
	}
}

func TestInverse(t *testing.T) {

	// Test 2x2 matrix inversion
	// A = [[4, 7], [2, 6]], A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
	A := []float64{
		4, 7,
		2, 6,
	}

	err := Inverse(2, A, 2)
	if err != nil {
		t.Fatalf("Inverse() failed: %v", err)
	}

	// Check inverse
	expected := []float64{0.6, -0.7, -0.2, 0.4}
	for i := range A {
		if math.Abs(A[i]-expected[i]) > epsilon {
			t.Errorf("A_inv[%d] = %f, want %f", i, A[i], expected[i])
		}
	}
}
