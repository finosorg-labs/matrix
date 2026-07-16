package matrix

import (
	"math"
	"testing"

)

func TestTranspose(t *testing.T) {

	// Test case: B = A^T
	// A: 2x3, B: 3x2
	A := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	B := make([]float64, 6)

	err := Transpose(2, 3, A, 3, B, 2)
	if err != nil {
		t.Fatalf("Transpose() failed: %v", err)
	}

	// Expected: B = [[1, 4], [2, 5], [3, 6]]
	expected := []float64{1, 4, 2, 5, 3, 6}
	for i := range B {
		if math.Abs(B[i]-expected[i]) > epsilon {
			t.Errorf("B[%d] = %f, want %f", i, B[i], expected[i])
		}
	}
}
