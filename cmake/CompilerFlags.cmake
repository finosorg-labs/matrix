#
# CompilerFlags.cmake - Compiler-specific flags for different SIMD levels
#

set(FC_AVX512_FLAGS "-mavx512f -mavx512dq -mavx512bw")
set(FC_AVX2_FLAGS   "-mavx2 -mfma")
set(FC_SSE42_FLAGS  "-msse4.2")

# Common flags shared by all SIMD levels (as list for proper CMake handling)
add_compile_options(-Wall -Wextra -Wpedantic -ffp-contract=off)

# Enable x86-64-v3 baseline (AVX2, BMI2, FMA) for better auto-vectorization
# This allows compiler auto-vectorization to use AVX2 instructions
# Manual SIMD code still uses runtime dispatch for optimal performance
if(NOT CMAKE_CROSSCOMPILING AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"))
    add_compile_options(-march=x86-64-v3)
endif()

# Suppress GNU extension warning for ##__VA_ARGS__ (widely supported by GCC, Clang, MSVC)
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-gnu-zero-variadic-macro-arguments)
endif()

# Coverage flags
if(FC_ENABLE_COVERAGE)
    message(STATUS "Enabling code coverage")
    add_compile_options(-fprofile-arcs -ftest-coverage -O0 -g)
    add_link_options(-fprofile-arcs -ftest-coverage)
endif()
