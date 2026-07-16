#
# cmake/toolchain/aarch64-apple-darwin.cmake — macOS ARM64 cross-compilation toolchain
#
#
# Prerequisites (Linux):
#   - clang with arm64-apple-darwin target support
#   - lld (LLVM linker) for macOS target
#   - macOS SDK (optional, for linking and running executables)
#     Download from: https://github.com/phracker/MacOSX-SDKs
#     Extract to: ~/.local/share/sdk/MacOSX.sdk
#
# Usage:
#   cmake -B build-darwin \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/aarch64-apple-darwin.cmake \
#     -DCMAKE_OSX_SYSROOT=~/.local/share/sdk/MacOSX.sdk \
#     -DCMAKE_BUILD_TYPE=Release

set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR arm64)

# Use the wrapper script so Clang applies Darwin target semantics
# (no __linux__, no GNU extensions by default, correct predefined macros)
set(CMAKE_C_COMPILER "${CMAKE_CURRENT_LIST_DIR}/clang-wrapper")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_LIST_DIR}/clang-wrapper")

# Target platform
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
set(CMAKE_OSX_ARCHITECTURES arm64)

# macOS sysroot (optional — required for linking and execution)
set(CMAKE_OSX_SYSROOT "" CACHE STRING "macOS SDK sysroot path")
if(CMAKE_OSX_SYSROOT)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Override OS detection macros so platform.h uses target OS (macOS), not host OS (Linux).
# The #ifndef guards in platform.h ensure toolchain-provided values take precedence.
add_compile_definitions(
    FC_OS_MACOS=1
    FC_OS_LINUX=0
    FC_OS_WINDOWS=0
)
