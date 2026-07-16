package matrix

import (
	"math"
	"testing"
)

func TestSolveTridiag(t *testing.T) {
	// Test simple tridiagonal system
	lower := []float64{0.0, 1.0, 1.0, 1.0, 1.0}
	diag := []float64{4.0, 4.0, 4.0, 4.0, 4.0}
	upper := []float64{1.0, 1.0, 1.0, 1.0, 0.0}
	d := []float64{5.0, 6.0, 6.0, 6.0, 5.0}

	x, err := SolveTridiag(lower, diag, upper, d)
	if err != nil {
		t.Fatalf("SolveTridiag failed: %v", err)
	}

	if len(x) != 5 {
		t.Fatalf("Expected solution length 5, got %d", len(x))
	}

	// Verify solution by computing residual
	for i := 0; i < 5; i++ {
		var residual float64
		if i == 0 {
			residual = diag[0]*x[0] + upper[0]*x[1] - d[0]
		} else if i == 4 {
			residual = lower[4]*x[3] + diag[4]*x[4] - d[4]
		} else {
			residual = lower[i]*x[i-1] + diag[i]*x[i] + upper[i]*x[i+1] - d[i]
		}
		if math.Abs(residual) > 1e-10 {
			t.Errorf("Residual at index %d too large: %e", i, residual)
		}
	}
}

func TestSolveTridiagMulti(t *testing.T) {
	// Test tridiagonal system with multiple RHS
	n := 4
	nrhs := 2
	lower := []float64{0.0, 1.0, 1.0, 1.0}
	diag := []float64{3.0, 3.0, 3.0, 3.0}
	upper := []float64{1.0, 1.0, 1.0, 0.0}

	// Two right-hand sides
	D := []float64{
		4.0, 8.0, // row 0
		5.0, 10.0, // row 1
		5.0, 10.0, // row 2
		4.0, 8.0, // row 3
	}

	X, err := SolveTridiagMulti(lower, diag, upper, D, nrhs)
	if err != nil {
		t.Fatalf("SolveTridiagMulti failed: %v", err)
	}

	if len(X) != n*nrhs {
		t.Fatalf("Expected solution length %d, got %d", n*nrhs, len(X))
	}

	// Verify each solution
	for k := 0; k < nrhs; k++ {
		for i := 0; i < n; i++ {
			var residual float64
			if i == 0 {
				residual = diag[0]*X[0*nrhs+k] + upper[0]*X[1*nrhs+k] - D[0*nrhs+k]
			} else if i == n-1 {
				residual = lower[i]*X[(i-1)*nrhs+k] + diag[i]*X[i*nrhs+k] - D[i*nrhs+k]
			} else {
				residual = lower[i]*X[(i-1)*nrhs+k] + diag[i]*X[i*nrhs+k] + upper[i]*X[(i+1)*nrhs+k] - D[i*nrhs+k]
			}
			if math.Abs(residual) > 1e-10 {
				t.Errorf("RHS %d: Residual at index %d too large: %e", k, i, residual)
			}
		}
	}
}
