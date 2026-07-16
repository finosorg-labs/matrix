package matrix

import (
	"math"
	"testing"

)

func TestGEMM(t *testing.T) {

	// Test case: C = A * B
	// A: 2x3, B: 3x2, C: 2x2
	A := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	B := []float64{
		7, 8,
		9, 10,
		11, 12,
	}
	C := make([]float64, 4)

	err := GEMM(2, 2, 3, 1.0, A, 3, B, 2, 0.0, C, 2)
	if err != nil {
		t.Fatalf("GEMM() failed: %v", err)
	}

	// Expected: C = [[58, 64], [139, 154]]
	expected := []float64{58, 64, 139, 154}
	for i := range C {
		if math.Abs(C[i]-expected[i]) > epsilon {
			t.Errorf("C[%d] = %f, want %f", i, C[i], expected[i])
		}
	}
}

func BenchmarkGEMM(b *testing.B) {

	sizes := []int{64, 128, 256, 512}
	for _, n := range sizes {
		b.Run(string(rune(n)), func(b *testing.B) {
			A := make([]float64, n*n)
			B := make([]float64, n*n)
			C := make([]float64, n*n)

			for i := range A {
				A[i] = float64(i)
				B[i] = float64(i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = GEMM(int64(n), int64(n), int64(n), 1.0, A, int64(n), B, int64(n), 0.0, C, int64(n))
			}
		})
	}
}
