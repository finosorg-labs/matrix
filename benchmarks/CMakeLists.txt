#
# benchmarks/CMakeLists.txt - Benchmark configuration for matrix module
#

# Benchmark framework library
add_library(bench_framework OBJECT
    bench_framework.c
)

target_include_directories(bench_framework
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/platform/include
)

# Benchmark executable
add_executable(matrix_benchmarks
    bench_main.c
    bench_matrix.c
    bench_decompose.c
    bench_gemm.c
    bench_gemv.c
    bench_precision.c
    bench_solve.c
    bench_transpose.c
    bench_tridiag.c
    bench_vector_ops.c
)

# Link with benchmark framework and merged matrix library
target_link_libraries(matrix_benchmarks
    PRIVATE
        bench_framework
        finkit_matrix_static_interface
        pthread
)

target_compile_definitions(matrix_benchmarks PRIVATE
    FC_STATIC_BUILD
)

# Include directories
target_include_directories(matrix_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/platform/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags for benchmarks
target_compile_options(matrix_benchmarks PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

# Installation
install(TARGETS matrix_benchmarks
    RUNTIME DESTINATION bin
)
