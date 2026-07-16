#
# cmake/sanitizers.cmake - CMake sanitizer configuration for matrix library
#
#
# Provides unified sanitizer configuration that works across:
#   - GCC / Clang (Linux, macOS, Windows via MinGW)
#
# Usage:
#   cmake -B build -DFC_ENABLE_SANITIZERS=ON -DFC_SANITIZER_TYPE=address
#   cmake -B build -DFC_ENABLE_SANITIZERS=ON -DFC_SANITIZER_TYPE=all
#
# Available sanitizer types:
#   address  - AddressSanitizer (memory errors: leaks, OOB, use-after-free)
#   undefined - UndefinedBehaviorSanitizer (UB: overflow, null deref, etc.)
#   thread   - ThreadSanitizer (data races)
#   memory   - MemorySanitizer (uninitialized memory reads, Clang only)
#   all      - All sanitizers (matrix-dependent)
#
# NOTE: Sanitizers are only effective in Debug or RelWithDebInfo builds.
#       They impose significant runtime overhead and are NOT suitable for
#       production Release builds.
#

include_guard(GLOBAL)

# -----------------------------------------------------------------------------
# Sanitizer type selection
# -----------------------------------------------------------------------------
if(NOT DEFINED FC_SANITIZER_TYPE)
    set(FC_SANITIZER_TYPE "address" CACHE STRING
        "Sanitizer type: address|undefined|thread|memory|all")
endif()

# Normalize to lowercase
string(TOLOWER "${FC_SANITIZER_TYPE}" FC_SANITIZER_TYPE)

# Validate type
set(_valid_types "address" "undefined" "thread" "memory" "all")
list(FIND _valid_types "${FC_SANITIZER_TYPE}" _type_idx)
if(_type_idx EQUAL -1)
    message(FATAL_ERROR "FC_SANITIZER_TYPE must be one of: ${_valid_types}, got '${FC_SANITIZER_TYPE}'")
endif()

# -----------------------------------------------------------------------------
# Build-type guard
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND
   NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" AND
   NOT CMAKE_BUILD_TYPE STREQUAL "Coverage" AND
   NOT CMAKE_BUILD_TYPE STREQUAL "" )
    message(WARNING "Sanitizers are only effective in Debug/RelWithDebInfo builds. "
                    "Current build type: ${CMAKE_BUILD_TYPE}. "
                    "Sanitizer flags will be NOT applied.")
    return()
endif()

# -----------------------------------------------------------------------------
# Compiler detection
# -----------------------------------------------------------------------------
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Arm")
    set(_COMPILER_IS_GNUC TRUE)
    set(_COMPILER_IS_CLANG FALSE)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(_COMPILER_IS_GNUC FALSE)
    set(_COMPILER_IS_CLANG TRUE)
else()
    message(AUTHOR_WARNING "Unknown compiler '${CMAKE_C_COMPILER_ID}'. "
                           "Sanitizer support may be incomplete.")
    set(_COMPILER_IS_GNUC TRUE)
    set(_COMPILER_IS_CLANG FALSE)
endif()

# -----------------------------------------------------------------------------
# Helper: check if a sanitizer is applicable
# -----------------------------------------------------------------------------
function(_fc_sanitizer_check_available sanitizer_name check_var)
    if(DEFINED ${check_var})
        if(NOT ${check_var})
            message(STATUS "fin-kit sanitizers: ${sanitizer_name} is not available "
                           "on this compiler/platform, skipping.")
        endif()
    endif()
endfunction()

# -----------------------------------------------------------------------------
# GCC/Clang sanitizer configuration
# -----------------------------------------------------------------------------
if(_COMPILER_IS_GNUC OR _COMPILER_IS_CLANG)

    # Sanitizer blacklist to exclude third-party code (Clang only)
    set(_BLACKLIST_FLAG "")
    if(_COMPILER_IS_CLANG)
        set(_SANITIZER_BLACKLIST "${CMAKE_CURRENT_SOURCE_DIR}/.github/sanitizer-blacklist.txt")
        if(EXISTS "${_SANITIZER_BLACKLIST}")
            set(_BLACKLIST_FLAG "-fsanitize-blacklist=${_SANITIZER_BLACKLIST}")
        endif()
    endif()

    # Compiler flags for each sanitizer type
    set(_ASAN_FLAGS    "-fsanitize=address" "-fno-omit-frame-pointer")
    set(_USAN_FLAGS    "-fsanitize=undefined" "-fno-omit-frame-pointer")
    set(_TSAN_FLAGS    "-fsanitize=thread" "-fno-omit-frame-pointer" "-g")
    set(_MSAN_FLAGS    "-fsanitize=memory" "-fno-omit-frame-pointer" "-fsanitize-memory-track-origins=2")

    # Add blacklist flag if available
    if(_BLACKLIST_FLAG)
        list(APPEND _ASAN_FLAGS ${_BLACKLIST_FLAG})
        list(APPEND _USAN_FLAGS ${_BLACKLIST_FLAG})
        list(APPEND _TSAN_FLAGS ${_BLACKLIST_FLAG})
        list(APPEND _MSAN_FLAGS ${_BLACKLIST_FLAG})
    endif()

    # Build combined flags
    set(_all_flags "")
    set(_link_flags "")

    # Determine which sanitizers to enable
    set(_enable_asan  FALSE)
    set(_enable_usan  FALSE)
    set(_enable_tsan  FALSE)
    set(_enable_msan  FALSE)

    if(FC_SANITIZER_TYPE STREQUAL "all")
        set(_enable_asan  TRUE)
        set(_enable_usan  TRUE)
        set(_enable_tsan  TRUE)
        # MSan requires Clang - check below
        if(_COMPILER_IS_CLANG)
            set(_enable_msan TRUE)
        else()
            message(STATUS "fin-kit sanitizers: MemorySanitizer requires Clang, skipping MSan.")
        endif()
    elseif(FC_SANITIZER_TYPE STREQUAL "address")
        set(_enable_asan TRUE)
    elseif(FC_SANITIZER_TYPE STREQUAL "undefined")
        set(_enable_usan TRUE)
    elseif(FC_SANITIZER_TYPE STREQUAL "thread")
        set(_enable_tsan TRUE)
    elseif(FC_SANITIZER_TYPE STREQUAL "memory")
        if(_COMPILER_IS_CLANG)
            set(_enable_msan TRUE)
        else()
            message(FATAL_ERROR "MemorySanitizer (MSan) requires Clang. "
                               "Current compiler: ${CMAKE_C_COMPILER_ID}")
        endif()
    endif()

    # Collect compile/link flags
    if(_enable_asan)
        list(APPEND _all_flags   ${_ASAN_FLAGS})
        list(APPEND _link_flags ${_ASAN_FLAGS})
    endif()

    if(_enable_usan)
        list(APPEND _all_flags   ${_USAN_FLAGS})
        list(APPEND _link_flags ${_USAN_FLAGS})
    endif()

    if(_enable_tsan)
        list(APPEND _all_flags   ${_TSAN_FLAGS})
        list(APPEND _link_flags ${_TSAN_FLAGS})
    endif()

    if(_enable_msan)
        list(APPEND _all_flags   ${_MSAN_FLAGS})
        list(APPEND _link_flags ${_MSAN_FLAGS})
    endif()

    # Remove duplicates
    if(_all_flags)
        list(REMOVE_DUPLICATES _all_flags)
    endif()
    if(_link_flags)
        list(REMOVE_DUPLICATES _link_flags)
    endif()

    # Apply flags to all targets
    if(_all_flags)
        string(REPLACE ";" " " _flag_str "${_all_flags}")
        add_compile_options(${_all_flags})

        if(_link_flags)
            add_link_options(${_link_flags})
        endif()

        # Extra restrictions for MSan (must be pure C, no external deps with UB)
        if(_enable_msan)
            add_compile_options(-Wno-unused-command-line-argument)
        endif()

        message(STATUS "fin-kit sanitizers: enabled ${FC_SANITIZER_TYPE} "
                       "(${_flag_str}) for ${CMAKE_BUILD_TYPE} build")
    endif()

endif()

# -----------------------------------------------------------------------------
# Sanitizer-specific environment / runtime options
# -----------------------------------------------------------------------------

# Common ASan options
set(ASAN_OPTIONS
    "detect_leaks=1"
    "strict_string_checks=1"
    "detect_container_overflow=1"
    "detect_odr_violation=0"
    "new_delete_type_mismatch=1"
    PARAMS
        "halt_on_error=0"
        "print_summary=1"
)

# Common USan options
set(UBSAN_OPTIONS
    "print_stacktrace=1"
    "halt_on_error=0"
    "print_summary=1"
)

# Common TSan options
set(TSAN_OPTIONS
    "halt_on_error=0"
    "print_stacktrace=1"
    "history_size=7"
)

# MemorySanitizer options
set(MSAN_OPTIONS
    "print_stats=1"
    "print_stacktrace=1"
    "halt_on_error=0"
)

# Export options to a file that can be sourced or used by tests
file(WRITE "${CMAKE_BINARY_DIR}/sanitizer_env.sh" "")
file(APPEND "${CMAKE_BINARY_DIR}/sanitizer_env.sh" "# fin-kit sanitizer environment variables\n")

if(_COMPILER_IS_GNUC OR _COMPILER_IS_CLANG)
    if(FC_SANITIZER_TYPE STREQUAL "address" OR FC_SANITIZER_TYPE STREQUAL "all")
        string(REPLACE ";" ":" _asan_opts "${ASAN_OPTIONS}")
        file(APPEND "${CMAKE_BINARY_DIR}/sanitizer_env.sh"
            "export ASAN_OPTIONS=\"${_asan_opts}\"\n")
    endif()

    if(FC_SANITIZER_TYPE STREQUAL "undefined" OR FC_SANITIZER_TYPE STREQUAL "all")
        string(REPLACE ";" ":" _ubsan_opts "${UBSAN_OPTIONS}")
        file(APPEND "${CMAKE_BINARY_DIR}/sanitizer_env.sh"
            "export UBSAN_OPTIONS=\"${_ubsan_opts}\"\n")
    endif()

    if(FC_SANITIZER_TYPE STREQUAL "thread" OR FC_SANITIZER_TYPE STREQUAL "all")
        string(REPLACE ";" ":" _tsan_opts "${TSAN_OPTIONS}")
        file(APPEND "${CMAKE_BINARY_DIR}/sanitizer_env.sh"
            "export TSAN_OPTIONS=\"${_tsan_opts}\"\n")
    endif()

    if(FC_SANITIZER_TYPE STREQUAL "memory" OR FC_SANITIZER_TYPE STREQUAL "all")
        string(REPLACE ";" ":" _msan_opts "${MSAN_OPTIONS}")
        file(APPEND "${CMAKE_BINARY_DIR}/sanitizer_env.sh"
            "export MSAN_OPTIONS=\"${_msan_opts}\"\n")
    endif()

    message(STATUS "platform sanitizers: runtime options written to ${CMAKE_BINARY_DIR}/sanitizer_env.sh")
endif()

# -----------------------------------------------------------------------------
# Cleanup helper variables
# -----------------------------------------------------------------------------
unset(_COMPILER_IS_GNUC)
unset(_COMPILER_IS_CLANG)
unset(_valid_types)
unset(_type_idx)
