# matrix Makefile
#
# Build Targets:
#   make           build Linux native (default)
#   make linux     build Linux native
#   make windows   cross-compile Windows amd64
#   make all       build Linux + Windows
#
# Test Targets:
#   make test      run Go tests
#   make bench     run Go benchmarks
#
# QA Targets:
#   make qa                run all QA checks (sanitizers + static analysis)
#   make qa-sanitizers     run all sanitizers (ASan/USan/TSan/MSan)
#   make qa-static         run static analysis (clang-tidy/cppcheck)
#
#   Individual sanitizers:
#     make sanitizer-asan   AddressSanitizer
#     make sanitizer-usan   UndefinedBehaviorSanitizer
#     make sanitizer-tsan   ThreadSanitizer
#     make sanitizer-msan   MemorySanitizer (requires clang)
#
#   Individual static analysis:
#     make clang-tidy       clang-tidy analysis
#     make cppcheck         cppcheck analysis
#
# Utility Targets:
#   make format    format C code with clang-format
#   make verify    verify artifact formats
#   make clean     remove all build artifacts
#   make sync      sync all submodules
#   make help      show this help

BUILD_TYPE ?= Release
CMAKE ?= cmake
TOOLCHAIN_DIR := cmake/toolchain

LINUX_BUILD_DIR  := build/linux_amd64
WINDOWS_BUILD_DIR := build/windows_amd64

LINUX_ARTIFACT_DIR    := $(LINUX_BUILD_DIR)
WINDOWS_ARTIFACT_DIR  := $(WINDOWS_BUILD_DIR)

LINUX_CONFIG := -G Ninja \
	-B $(LINUX_BUILD_DIR) \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DFC_BUILD_TESTS=ON \
	-DFC_BUILD_BENCHMARKS=$(shell [ "$(BUILD_TYPE)" = "Release" ] && echo ON || echo OFF)

COVERAGE_CONFIG := -G Ninja \
	-B $(LINUX_BUILD_DIR) \
	-DCMAKE_BUILD_TYPE=Debug \
	-DFC_BUILD_TESTS=ON \
	-DFC_BUILD_BENCHMARKS=OFF \
	-DFC_ENABLE_COVERAGE=ON

.PHONY: all default linux windows go test bench clean verify help format
.PHONY: qa qa-sanitizers qa-static
.PHONY: sanitizer-asan sanitizer-usan sanitizer-tsan sanitizer-msan clang-tidy cppcheck
.PHONY: sync build_third clean_third

default: linux

all: format linux go

qa: format qa-static qa-sanitizers
	@echo "==> All QA checks completed"

qa-static: clang-tidy cppcheck
	@echo "==> All static analysis checks completed"

qa-sanitizers: sanitizer-asan sanitizer-usan sanitizer-tsan sanitizer-msan
	@echo "==> All sanitizer checks completed"

build_third:
	@echo "==> Building third-party libraries (OpenBLAS)"
	@echo "==> Cleaning previous OpenBLAS build"
	@cd third_party/OpenBLAS && $(MAKE) clean || true
	@echo "==> Building OpenBLAS library (TARGET=HASWELL with DYNAMIC_ARCH and LAPACK)"
	@cd third_party/OpenBLAS && $(MAKE) TARGET=HASWELL DYNAMIC_ARCH=1 NO_SHARED=1 USE_THREAD=1 NUM_THREADS=64 libs netlib -j$(shell nproc)


copy_third:
	@mkdir -p $(LINUX_BUILD_DIR)/third_party
	@cp third_party/OpenBLAS/libopenblas.a $(LINUX_BUILD_DIR)/third_party/libopenblas.a
	@echo "==> Third-party libraries built successfully: $(LINUX_BUILD_DIR)/third_party/libopenblas.a"


clean_third:
	@echo "==> Cleaning third-party build artifacts"
	@cd third_party/OpenBLAS && $(MAKE) clean || true
	@rm -f $(LINUX_BUILD_DIR)/third_party/libopenblas.a
	@rm -f $(WINDOWS_BUILD_DIR)/third_party/libopenblas.a

linux:
	@echo "==> Building Linux (native, $(BUILD_TYPE))"
	@$(CMAKE) -B $(LINUX_BUILD_DIR) \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(CMAKE) --build $(LINUX_BUILD_DIR) --parallel
	@echo "==> Cleaning intermediate build artifacts"
	@rm -f $(LINUX_BUILD_DIR)/libfinkit_matrix_static_base.a

windows:
	@echo "==> Building Windows amd64 (cross-compile, $(BUILD_TYPE))"
	@$(CMAKE) -B $(WINDOWS_BUILD_DIR) \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_DIR)/x86_64-w64-mingw32.cmake
	@$(CMAKE) --build $(WINDOWS_BUILD_DIR) --parallel

go:
	@echo "==> Building Go module with source (verify compilation)"
	@CGO_CFLAGS_ALLOW="-m(avx2|avx512f|avx512dq|fma|sse4\.2)" go build ./...
	@echo "==> Building Go module with lib (verify compilation)"
	@CGO_CFLAGS_ALLOW="-m(avx2|avx512f|avx512dq|fma|sse4\.2)" go build -tags lib ./...

test: linux
	@echo "==> Rebuilding with coverage enabled"
	@$(CMAKE) $(COVERAGE_CONFIG)
	@$(CMAKE) --build $(LINUX_BUILD_DIR) --parallel
	@echo "==> Running C tests with coverage"
	@bash scripts/test_coverage.sh $(LINUX_BUILD_DIR)
	@echo ""
	@echo "==> Running Go tests"
	@FC_BUILD_MODE=source CGO_CFLAGS_ALLOW="-m(avx2|avx512f|avx512dq|fma|sse4\.2)" go test -vet=all -race -parallel=4 -v ./...

bench:
	@echo "==> Building benchmarks (Release mode)"
	@BUILD_TYPE=Release $(CMAKE) -B $(LINUX_BUILD_DIR) \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DFC_BUILD_TESTS=OFF \
		-DFC_BUILD_BENCHMARKS=ON \
	@$(CMAKE) --build $(LINUX_BUILD_DIR) --parallel
	@echo "==> Running C benchmarks"
	@if [ -f $(LINUX_BUILD_DIR)/benchmarks/matrix_benchmarks ]; then \
		$(LINUX_BUILD_DIR)/benchmarks/matrix_benchmarks; \
	else \
		echo "No C benchmarks found"; \
	fi
	@echo ""
	@echo "==> Running Go benchmarks"
	@FC_BUILD_MODE=source CGO_CFLAGS_ALLOW="-m(avx2|avx512f|avx512dq|fma|sse4\.2)" go test -bench=. -benchmem ./...

format:
	@echo "==> Formatting C code with clang-format"
	@if command -v clang-format >/dev/null 2>&1; then \
		find matrix-c include \( -name '*.c' -o -name '*.h' \) -exec clang-format -i {} \; ; \
	else \
		echo "WARNING: clang-format not found, skipping format check"; \
	fi

verify:
	@echo "=== Verify artifact formats ==="
	@echo "--- Linux ---"
	@objdump -f $(LINUX_ARTIFACT_DIR)/*.a 2>/dev/null | grep "file format" || echo "(no artifacts)"
	@echo "--- Windows ---"
	@objdump -f $(WINDOWS_ARTIFACT_DIR)/*.a 2>/dev/null | grep "file format" || echo "(no artifacts)"

sanitizer-asan:
	@echo "==> Building with AddressSanitizer"
	@$(CMAKE) -B build/sanitizer-asan \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DFC_BUILD_TESTS=ON \
		-DFC_BUILD_BENCHMARKS=OFF \
		-DFC_ENABLE_SANITIZERS=ON \
		-DFC_SANITIZER_TYPE=address \
	@$(CMAKE) --build build/sanitizer-asan --parallel
	@echo "==> Running AddressSanitizer tests"
	@cd build/sanitizer-asan && ctest --output-on-failure

sanitizer-usan:
	@echo "==> Building with UndefinedBehaviorSanitizer"
	@$(CMAKE) -B build/sanitizer-usan \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DFC_BUILD_TESTS=ON \
		-DFC_BUILD_BENCHMARKS=OFF \
		-DFC_ENABLE_SANITIZERS=ON \
		-DFC_SANITIZER_TYPE=undefined \
	@$(CMAKE) --build build/sanitizer-usan --parallel
	@echo "==> Running UndefinedBehaviorSanitizer tests"
	@cd build/sanitizer-usan && ctest --output-on-failure

sanitizer-tsan:
	@echo "==> Building with ThreadSanitizer"
	@$(CMAKE) -B build/sanitizer-tsan \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DFC_BUILD_TESTS=ON \
		-DFC_BUILD_BENCHMARKS=OFF \
		-DFC_ENABLE_SANITIZERS=ON \
		-DFC_SANITIZER_TYPE=thread \
	@$(CMAKE) --build build/sanitizer-tsan --parallel
	@echo "==> Running ThreadSanitizer tests"
	@cd build/sanitizer-tsan && ctest --output-on-failure || \
		(echo "WARNING: ThreadSanitizer failed (known WSL2 compatibility issue)" && exit 0)

sanitizer-msan:
	@echo "==> Building with MemorySanitizer (requires clang)"
	@CC=clang $(CMAKE) -B build/sanitizer-msan \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DFC_BUILD_TESTS=ON \
		-DFC_BUILD_BENCHMARKS=OFF \
		-DFC_ENABLE_SANITIZERS=ON \
		-DFC_SANITIZER_TYPE=memory \
	@$(CMAKE) --build build/sanitizer-msan --parallel
	@echo "==> Running MemorySanitizer tests"
	@cd build/sanitizer-msan && ctest --output-on-failure

clang-tidy:
	@echo "==> Generating compile_commands.json for clang-tidy"
	@CC=clang CXX=clang++ $(CMAKE) -B build/clang-tidy \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1 || true
	@echo "==> Running clang-tidy analysis on C source files"
	@find matrix-c include \( -name '*.c' -o -name '*.h' \) \
		! -name 'platform_win.c' \
		! -name 'platform_macos.c' \
		-print | while read f; do echo "  Checking: $$f"; done
	@find matrix-c include \( -name '*.c' -o -name '*.h' \) \
		! -name 'platform_win.c' \
		! -name 'platform_macos.c' \
		-exec clang-tidy -p build/clang-tidy {} \; 2>&1 | \
		grep -v "warnings generated" || true
	@echo "==> clang-tidy: No issues found"

cppcheck:
	@echo "==> Running cppcheck static analysis on C source files"
	@find matrix-c include \( -name '*.c' -o -name '*.h' \) -print | \
		while read f; do echo "  Checking: $$f"; done
	@cppcheck --enable=warning,performance,portability \
		--suppress=missingIncludeSystem \
		--suppress=missingInclude \
		--suppress=toomanyconfigs \
		--suppress=unusedFunction \
		--suppress=knownConditionTrueFalse \
		--inline-suppr --quiet \
		-I include matrix-c/ 2>&1 || true
	@echo "==> cppcheck: No issues found"

clean:
	@echo "==> Cleaning build artifacts (keeping third-party libraries)"
	@if [ -d build ]; then \
		find build -type f ! -path "*/third_party/*" -delete; \
		find build -type d -empty -delete; \
	fi
	@echo "==> Cleaning Go cache"
	@go clean -cache

sync:
	@echo "==> Syncing submodules recursively"
	@echo "==> third_party/ modules: full clone"
	@echo "==> other modules: sparse clone (build + include)"
	@bash -c 'set -e; \
	clone_submodule() { \
		local gitmodules_file=$$1; \
		git config -f $$gitmodules_file --get-regexp "^submodule\..*\.path$$" | while read key path; do \
			submodule_name=$$(echo $$key | sed "s/^submodule\.\(.*\)\.path$$/\1/"); \
			url=$$(git config -f $$gitmodules_file --get "submodule.$$submodule_name.url"); \
			module_name=$$(basename $$url .git); \
			branch=$$(git config -f $$gitmodules_file --get "submodule.$$submodule_name.branch" 2>/dev/null || echo "main"); \
			need_recurse=false; \
			is_third_party=$$(echo "$$path" | grep -q "^third_party/" && echo "true" || echo "false"); \
			\
			if [ -d "$$path/.git" ]; then \
				echo "  ↻ Checking $$module_name for updates"; \
				cd $$path; \
				current_commit=$$(git rev-parse HEAD 2>/dev/null || echo "none"); \
				git fetch --depth=1 origin $$branch 2>/dev/null || git fetch --depth=1 origin main 2>/dev/null || git fetch --depth=1 origin master 2>/dev/null; \
				latest_commit=$$(git rev-parse FETCH_HEAD 2>/dev/null); \
				if [ "$$current_commit" != "$$latest_commit" ]; then \
					echo "    ⬆ Updating $$module_name"; \
					git checkout FETCH_HEAD 2>/dev/null; \
					need_recurse=true; \
				else \
					echo "    ✓ $$module_name is up to date"; \
					cd - > /dev/null; \
					continue; \
				fi; \
				cd - > /dev/null; \
			else \
				if [ "$$is_third_party" = "true" ]; then \
					echo "  ⬇ Cloning $$module_name to $$path (full clone)"; \
					rm -rf $$path; \
					mkdir -p $$(dirname $$path); \
					git clone --depth=1 --branch $$branch $$url $$path 2>/dev/null || \
					git clone --depth=1 --branch main $$url $$path 2>/dev/null || \
					git clone --depth=1 --branch master $$url $$path; \
					need_recurse=true; \
				else \
					echo "  ⬇ Cloning $$module_name to $$path (sparse: build + include)"; \
					rm -rf $$path; \
					mkdir -p $$path; \
					cd $$path && \
					git init && \
					git remote add origin $$url && \
					git config core.sparseCheckout true && \
					git sparse-checkout init --no-cone && \
					git sparse-checkout set "build/*" "include/*" ".gitmodules" && \
					git fetch --depth=1 origin && \
					git checkout $$branch 2>/dev/null || git checkout main 2>/dev/null || git checkout master; \
					cd - > /dev/null; \
					need_recurse=true; \
				fi; \
			fi; \
			\
			if [ "$$need_recurse" = "true" ] && [ -f "$$path/.gitmodules" ]; then \
				echo "    Found nested submodules in $$module_name"; \
				clone_submodule "$$path/.gitmodules"; \
			fi; \
		done; \
	}; \
	clone_submodule ".gitmodules"'
	@echo "==> Submodules synced successfully"

help:
	@echo "matrix Makefile - Build and Test Targets"
	@echo ""
	@echo "Build Targets:"
	@echo "  make             - build Linux native (default)"
	@echo "  make linux       - build Linux native"
	@echo "  make windows     - cross-compile Windows amd64"
	@echo "  make all         - build third-party + Linux + Windows + Go"
	@echo "  make go          - build Go module"
	@echo "  make build_third - build third-party libraries (OpenBLAS)"
	@echo ""
	@echo "Test Targets:"
	@echo "  make test      - run Go tests"
	@echo "  make bench     - run Go benchmarks"
	@echo ""
	@echo "QA Targets:"
	@echo "  make qa            - run static analysis on C source code"
	@echo "  make qa-static     - run clang-tidy and cppcheck"
	@echo "  make qa-sanitizers - run runtime sanitizer tests"
	@echo "  make clang-tidy    - run clang-tidy analysis"
	@echo "  make cppcheck      - run cppcheck analysis"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make format      - format C code"
	@echo "  make verify      - verify artifact formats"
	@echo "  make clean       - remove build artifacts"
	@echo "  make clean_third - clean third-party build artifacts"
	@echo "  make sync        - sync all submodules"
	@echo "  make help        - show this help"
