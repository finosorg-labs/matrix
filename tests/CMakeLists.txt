#
# tests/CMakeLists.txt - Test configuration for matrix module
#

# Test executable
add_executable(matrix_tests
    test_framework.c
    test_main.c
    test_matrix.c
    test_decompose.c
    test_gemm.c
    test_gemv.c
    test_precision.c
    test_solve.c
    test_transpose.c
    test_tridiag.c
    test_vector_ops.c
)

# Link with merged matrix library (includes all dependencies)
target_link_libraries(matrix_tests
    PRIVATE
        finkit_matrix_static_interface
        pthread
)

target_compile_definitions(matrix_tests PRIVATE
    FC_STATIC_BUILD
    FC_ENABLE_INTERNAL_TESTS
)

# Include directories
target_include_directories(matrix_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/platform/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags for tests
target_compile_options(matrix_tests PRIVATE -Wall -Wextra -Wpedantic)

# Run tests with ctest
add_test(
    NAME "matrix_tests"
    COMMAND matrix_tests
)

# Installation
install(TARGETS matrix_tests
    RUNTIME DESTINATION bin
)
