package matrix

import (
	"math"
	"testing"

)

const epsilon = 1e-9

func TestVecDot(t *testing.T) {

	x := []float64{1, 2, 3}
	y := []float64{4, 5, 6}

	result, err := VecDot(x, y)
	if err != nil {
		t.Fatalf("VecDot() failed: %v", err)
	}

	// Expected: 1*4 + 2*5 + 3*6 = 32
	expected := 32.0
	if math.Abs(result-expected) > epsilon {
		t.Errorf("VecDot() = %f, want %f", result, expected)
	}
}

func TestVecNormL2(t *testing.T) {

	x := []float64{3, 4}

	result, err := VecNormL2(x)
	if err != nil {
		t.Fatalf("VecNormL2() failed: %v", err)
	}

	// Expected: sqrt(3^2 + 4^2) = 5
	expected := 5.0
	if math.Abs(result-expected) > epsilon {
		t.Errorf("VecNormL2() = %f, want %f", result, expected)
	}
}

func TestVecNormL1(t *testing.T) {

	x := []float64{-3, 4, -5}

	result, err := VecNormL1(x)
	if err != nil {
		t.Fatalf("VecNormL1() failed: %v", err)
	}

	// Expected: |−3| + |4| + |−5| = 12
	expected := 12.0
	if math.Abs(result-expected) > epsilon {
		t.Errorf("VecNormL1() = %f, want %f", result, expected)
	}
}

func TestVecDistanceL2(t *testing.T) {

	x := []float64{1, 2, 3}
	y := []float64{4, 6, 8}

	result, err := VecDistanceL2(x, y)
	if err != nil {
		t.Fatalf("VecDistanceL2() failed: %v", err)
	}

	// Expected: sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9+16+25) = sqrt(50)
	expected := math.Sqrt(50)
	if math.Abs(result-expected) > epsilon {
		t.Errorf("VecDistanceL2() = %f, want %f", result, expected)
	}
}

func TestVecDistanceL1(t *testing.T) {

	x := []float64{1, 2, 3}
	y := []float64{4, 6, 8}

	result, err := VecDistanceL1(x, y)
	if err != nil {
		t.Fatalf("VecDistanceL1() failed: %v", err)
	}

	// Expected: |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
	expected := 12.0
	if math.Abs(result-expected) > epsilon {
		t.Errorf("VecDistanceL1() = %f, want %f", result, expected)
	}
}

func BenchmarkVecDot(b *testing.B) {

	sizes := []int{1000, 10000, 100000}
	for _, n := range sizes {
		b.Run(string(rune(n)), func(b *testing.B) {
			x := make([]float64, n)
			y := make([]float64, n)

			for i := range x {
				x[i] = float64(i)
				y[i] = float64(i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = VecDot(x, y)
			}
		})
	}
}
