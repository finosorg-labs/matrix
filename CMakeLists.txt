cmake_minimum_required(VERSION 3.16)
project(finkit-matrix VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include SIMD detection and compiler flags
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DetectSIMD.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerFlags.cmake)

# Sanitizer configuration
if(FC_ENABLE_SANITIZERS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers.cmake)
    endif()
endif()

# Generate cgo_source.go with detected SIMD flags
if(FC_SIMD_LEVEL STREQUAL "AVX512")
    set(FC_SIMD_CFLAGS "${FC_AVX512_FLAGS}")
elseif(FC_SIMD_LEVEL STREQUAL "AVX2")
    set(FC_SIMD_CFLAGS "${FC_AVX2_FLAGS}")
elseif(FC_SIMD_LEVEL STREQUAL "SSE42")
    set(FC_SIMD_CFLAGS "${FC_SSE42_FLAGS}")
else()
    set(FC_SIMD_CFLAGS "")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cgo_source.go.in
    ${CMAKE_CURRENT_SOURCE_DIR}/cgo_source.go
    @ONLY
)

# Use prebuilt platform library
if(NOT TARGET finkit::platform)
    # Detect platform
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(PLATFORM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/build/darwin_arm64")
        else()
            set(PLATFORM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/build/darwin_x86_64")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(PLATFORM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/build/linux_amd64")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(PLATFORM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/build/windows_amd64")
    else()
        message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
    endif()

    # Import prebuilt platform library
    add_library(finkit_platform_static STATIC IMPORTED)
    set_target_properties(finkit_platform_static PROPERTIES
        IMPORTED_LOCATION "${PLATFORM_BUILD_DIR}/libfinkit_platform_static.a"
    )
    target_include_directories(finkit_platform_static INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/include"
    )
    add_library(finkit::platform ALIAS finkit_platform_static)
endif()

# Determine matrix build directory (needed for OpenBLAS path detection)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(MATRIX_BUILD_DIR "darwin_arm64")
    else()
        set(MATRIX_BUILD_DIR "darwin_x86_64")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(MATRIX_BUILD_DIR "linux_amd64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(MATRIX_BUILD_DIR "windows_amd64")
endif()

# Try to find OpenBLAS library in build directory first, then fall back to source directory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/${MATRIX_BUILD_DIR}/third_party/libopenblas.a)
    set(FC_OPENBLAS_STATIC_LIB ${CMAKE_CURRENT_SOURCE_DIR}/build/${MATRIX_BUILD_DIR}/third_party/libopenblas.a)
elseif(EXISTS ${CMAKE_BINARY_DIR}/third_party/libopenblas.a)
    set(FC_OPENBLAS_STATIC_LIB ${CMAKE_BINARY_DIR}/third_party/libopenblas.a)
else()
    set(FC_OPENBLAS_STATIC_LIB ${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenBLAS/libopenblas.a)
endif()


# Matrix module sources
set(MATRIX_SOURCES
    matrix-c/decompose.c
    matrix-c/gemm.c
    matrix-c/gemv.c
    matrix-c/solve.c
    matrix-c/transpose.c
    matrix-c/tridiag.c
    matrix-c/vector_ops.c
)

# Create object library
add_library(finkit_matrix OBJECT ${MATRIX_SOURCES})

target_include_directories(finkit_matrix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenBLAS
)

target_compile_definitions(finkit_matrix PRIVATE FC_STATIC_BUILD FC_ENABLE_INTERNAL_TESTS)

# For files with multiple SIMD paths, use the highest available
if(FC_SIMD_LEVEL STREQUAL "AVX512")
    set_source_files_properties(matrix-c/gemm.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/gemv.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/transpose.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/vector_ops.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/decompose.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/solve.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/tridiag.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
elseif(FC_SIMD_LEVEL STREQUAL "AVX2")
    set_source_files_properties(matrix-c/gemm.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/gemv.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/transpose.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/vector_ops.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/decompose.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/solve.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/tridiag.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
elseif(FC_SIMD_LEVEL STREQUAL "SSE42")
    set_source_files_properties(matrix-c/gemm.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/gemv.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/transpose.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/vector_ops.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/decompose.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/solve.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/tridiag.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
endif()

# Create base static library (without platform)
add_library(finkit_matrix_static_base STATIC $<TARGET_OBJECTS:finkit_matrix>)
target_include_directories(finkit_matrix_static_base
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Merge static libraries into a single library
set(MERGED_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/${MATRIX_BUILD_DIR}/libfinkit_matrix_static.a)
set(PLATFORM_LIB_PATH "${PLATFORM_BUILD_DIR}/libfinkit_platform_static.a")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Windows: use lib.exe to merge libraries
    add_custom_command(
        OUTPUT ${MERGED_LIB_PATH}
     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/build/${MATRIX_BUILD_DIR}
     COMMAND lib.exe /OUT:${MERGED_LIB_PATH}
            $<TARGET_FILE:finkit_matrix_static_base>
            ${PLATFORM_LIB_PATH}
            ${FC_OPENBLAS_STATIC_LIB}
        DEPENDS finkit_matrix_static_base ${PLATFORM_LIB_PATH} ${FC_OPENBLAS_STATIC_LIB}
        COMMENT "Merging static libraries into ${MERGED_LIB_PATH}"
        VERBATIM
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS: use libtool to merge libraries
    add_custom_command(
        OUTPUT ${MERGED_LIB_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/build/${MATRIX_BUILD_DIR}
        COMMAND libtool -static -o ${MERGED_LIB_PATH}
            $<TARGET_FILE:finkit_matrix_static_base>
            ${PLATFORM_LIB_PATH}
            ${FC_OPENBLAS_STATIC_LIB}
            2>&1 | grep -v "warning duplicate member name" || true
        DEPENDS finkit_matrix_static_base ${PLATFORM_LIB_PATH} ${FC_OPENBLAS_STATIC_LIB}
      COMMENT "Merging static libraries into ${MERGED_LIB_PATH}"
        VERBATIM
    )
else()
    # Linux: use ar with MRI script to merge libraries
    set(MRI_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/merge_libs.mri)
    file(GENERATE OUTPUT ${MRI_SCRIPT} CONTENT
"CREATE ${MERGED_LIB_PATH}
ADDLIB $<TARGET_FILE:finkit_matrix_static_base>
ADDLIB ${PLATFORM_LIB_PATH}
ADDLIB ${FC_OPENBLAS_STATIC_LIB}
SAVE
END
")

    add_custom_command(
        OUTPUT ${MERGED_LIB_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/build/${MATRIX_BUILD_DIR}
        COMMAND ${CMAKE_AR} -M < ${MRI_SCRIPT}
        DEPENDS finkit_matrix_static_base ${PLATFORM_LIB_PATH} ${FC_OPENBLAS_STATIC_LIB} ${MRI_SCRIPT}
        COMMENT "Merging static libraries into ${MERGED_LIB_PATH}"
        VERBATIM
    )
endif()

# Create custom target to trigger the merge
add_custom_target(finkit_matrix_static ALL
    DEPENDS ${MERGED_LIB_PATH}
    COMMAND ${CMAKE_COMMAND} -E echo "Merged library created at: ${MERGED_LIB_PATH}"
)

# Create interface library for linking
add_library(finkit_matrix_static_interface INTERFACE)
target_include_directories(finkit_matrix_static_interface
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(finkit_matrix_static_interface
    INTERFACE
        m
        ${MERGED_LIB_PATH}
)

# Create alias
add_library(finkit::matrix ALIAS finkit_matrix_static_interface)

# Install merged static library
install(FILES ${MERGED_LIB_PATH}
    DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/matrix
    FILES_MATCHING PATTERN "*.h"
)

# Install interface library target
install(TARGETS finkit_matrix_static_interface
    EXPORT finkit-matrix-targets
)

install(EXPORT finkit-matrix-targets
    FILE finkit-matrixTargets.cmake
    NAMESPACE finkit::
    DESTINATION lib/cmake/finkit-matrix
)

# Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finkit-matrixConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfig.cmake"
    INSTALL_DESTINATION lib/cmake/finkit-matrix
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfigVersion.cmake"
    DESTINATION lib/cmake/finkit-matrix
)

# Tests
option(FC_BUILD_TESTS "Build tests" ON)
if(FC_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(FC_BUILD_BENCHMARKS "Build benchmarks" ON)
if(FC_BUILD_BENCHMARKS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    add_subdirectory(benchmarks)
endif()

# Print configuration
message(STATUS "")
message(STATUS "============================================================")
message(STATUS " finkit-matrix ${PROJECT_VERSION}")
message(STATUS "============================================================")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "============================================================")
message(STATUS "")
