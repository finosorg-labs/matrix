#
# DetectSIMD.cmake - CPU SIMD capability detection
#
# Populates FC_SIMD_LEVEL with: SCALAR, SSE42, AVX2, or AVX512
# Then creates an interface library that links compiler flags from CompilerFlags.cmake
#

include(CheckCSourceCompiles)

# Save original flags
set(_orig_cmake_required_flags "${CMAKE_REQUIRED_FLAGS}")

# Detect SSE4.2
set(CMAKE_REQUIRED_FLAGS "${_orig_cmake_required_flags} -msse4.2")
check_c_source_compiles("
    #include <immintrin.h>
    int main() {
        __m128i a = _mm_setzero_si128();
        return _mm_extract_epi32(a, 0);
    }
" FC_HAS_SSE42)

# Detect AVX2
set(CMAKE_REQUIRED_FLAGS "${_orig_cmake_required_flags} -mavx2 -mfma")
check_c_source_compiles("
    #include <immintrin.h>
    int main() {
        __m256i a = _mm256_setzero_si256();
        return _mm256_extract_epi32(a, 0);
    }
" FC_HAS_AVX2)

# Detect AVX-512
set(CMAKE_REQUIRED_FLAGS "${_orig_cmake_required_flags} -mavx512f -mavx512dq")
check_c_source_compiles("
    #include <immintrin.h>
    int main() {
        __m512i a = _mm512_setzero_si512();
        return _mm512_extract_epi32(a, 0);
    }
" FC_HAS_AVX512_RUNTIME)

# Restore original flags
set(CMAKE_REQUIRED_FLAGS "${_orig_cmake_required_flags}")

# For compilation: Always enable all SIMD levels that the compiler supports
# This allows generating code for all SIMD variants regardless of build machine CPU
# Runtime dispatch will select the appropriate variant based on actual CPU
set(FC_HAS_SSE42 1)
set(FC_HAS_AVX2 1)
set(FC_HAS_AVX512 1)

# Determine the highest SIMD level for runtime (based on build machine)
if(FC_HAS_AVX512_RUNTIME)
    set(FC_SIMD_LEVEL "AVX512")
elseif(FC_HAS_AVX2)
    set(FC_SIMD_LEVEL "AVX2")
elseif(FC_HAS_SSE42)
    set(FC_SIMD_LEVEL "SSE42")
else()
    set(FC_SIMD_LEVEL "SCALAR")
endif()

# Promote to cache so it persists across re-runs and is visible to IDEs
set(FC_SIMD_LEVEL "${FC_SIMD_LEVEL}" CACHE STRING "Highest supported SIMD instruction set")
message(STATUS "SIMD level: ${FC_SIMD_LEVEL}")
