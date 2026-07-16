package matrix

import (
	"math"
	"testing"

)

func TestGEMV(t *testing.T) {

	// Test case: y = A * x
	// A: 2x3, x: 3x1, y: 2x1
	A := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	x := []float64{1, 2, 3}
	y := make([]float64, 2)

	err := GEMV(2, 3, 1.0, A, 3, x, 0.0, y)
	if err != nil {
		t.Fatalf("GEMV() failed: %v", err)
	}

	// Expected: y = [14, 32]
	expected := []float64{14, 32}
	for i := range y {
		if math.Abs(y[i]-expected[i]) > epsilon {
			t.Errorf("y[%d] = %f, want %f", i, y[i], expected[i])
		}
	}
}
