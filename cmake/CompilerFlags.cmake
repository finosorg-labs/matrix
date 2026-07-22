#
# CompilerFlags.cmake - Compiler-specific flags for different SIMD levels
#

set(FC_AVX512_FLAGS "-mavx512f -mavx512dq -mavx512bw")
# Prevent GCC from using AVX-512 extended registers (xmm16-xmm31) in AVX2/SSE code
# when -mavx512f is enabled at file level
set(FC_AVX2_FLAGS   "-mavx2 -mfma -ffixed-xmm16 -ffixed-xmm17 -ffixed-xmm18 -ffixed-xmm19 -ffixed-xmm20 -ffixed-xmm21 -ffixed-xmm22 -ffixed-xmm23 -ffixed-xmm24 -ffixed-xmm25 -ffixed-xmm26 -ffixed-xmm27 -ffixed-xmm28 -ffixed-xmm29 -ffixed-xmm30 -ffixed-xmm31")
set(FC_SSE42_FLAGS  "-msse4.2 -ffixed-xmm16 -ffixed-xmm17 -ffixed-xmm18 -ffixed-xmm19 -ffixed-xmm20 -ffixed-xmm21 -ffixed-xmm22 -ffixed-xmm23 -ffixed-xmm24 -ffixed-xmm25 -ffixed-xmm26 -ffixed-xmm27 -ffixed-xmm28 -ffixed-xmm29 -ffixed-xmm30 -ffixed-xmm31")

# Common flags shared by all SIMD levels (as list for proper CMake handling)
add_compile_options(-Wall -Wextra -Wpedantic -ffp-contract=off)

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
