SHELL := /usr/bin/env bash

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=
NATIVE_BIN := air_llm/build/native/airllm
ROOT_DIR := air_llm
OUT_DIR := $(ROOT_DIR)/build/native
PYTHON_BIN ?= $(ROOT_DIR)/.venv/bin/python

.PHONY: all build clean install uninstall release-archive

all: build

build:
	@set -euo pipefail; \
	echo "Running from root directory: $(ROOT_DIR)"; \
	echo "Using Python executable: $(PYTHON_BIN)"; \
	if [[ ! -x "$(PYTHON_BIN)" ]]; then \
	  echo "Python executable not found: $(PYTHON_BIN)" >&2; \
	  exit 1; \
	fi; \
	if ! "$(PYTHON_BIN)" -m pip --version >/dev/null 2>&1; then \
	  echo "Bootstrapping pip in build environment..." >&2; \
	  "$(PYTHON_BIN)" -m ensurepip --upgrade; \
	fi; \
	if ! "$(PYTHON_BIN)" -c 'import nuitka, zstandard' >/dev/null 2>&1; then \
	  echo "Installing build dependency: nuitka[onefile]" >&2; \
	  "$(PYTHON_BIN)" -m pip install --upgrade "nuitka[onefile]>=1.8"; \
	fi; \
	PY_VERSION="$$($(PYTHON_BIN) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; \
	if [[ "$$PY_VERSION" != "3.13" && "$${AIRLLM_ALLOW_EXPERIMENTAL_PYTHON:-0}" != "1" ]]; then \
	  echo "Full standalone Nuitka builds are pinned to Python 3.13 for stability." >&2; \
	  echo "Detected Python $$PY_VERSION." >&2; \
	  echo "Use Python 3.13 or set AIRLLM_ALLOW_EXPERIMENTAL_PYTHON=1 to proceed at your own risk." >&2; \
	  exit 2; \
	fi; \
	mkdir -p "$(OUT_DIR)"; \
	"$(PYTHON_BIN)" -m nuitka \
	  --standalone \
	  --onefile \
	  --assume-yes-for-downloads \
	  --module-parameter=torch-disable-jit=no \
	  --include-distribution-metadata=huggingface_hub \
	  --include-distribution-metadata=hf-xet \
	  --include-distribution-metadata=pytest \
	  --include-distribution-metadata=httpx \
	  --include-distribution-metadata=kernels \
	  --include-distribution-metadata=triton \
	  --include-package=airllm \
	  --include-package=hf_xet \
	  --include-package=kernels \
	  --include-package=httpx \
	  --include-package=triton \
	  --include-package=triton.backends \
	  --include-package=triton.backends.amd \
	  --include-package=triton.backends.nvidia \
	  --include-package=_pytest \
	  --include-package-data=kernels \
	  --output-dir="$(OUT_DIR)" \
	  --output-filename="airllm" \
	  "$(ROOT_DIR)/airllm_server.py"; \
	echo "Standalone native executable created at $(OUT_DIR)/airllm"

clean:
	@rm -rf "$(OUT_DIR)/airllm" "$(OUT_DIR)/airllm.sh" "$(OUT_DIR)/airllm_server.build" "$(OUT_DIR)/airllm_server.dist" "$(OUT_DIR)/airllm_server.onefile-build"
	@echo "Cleaned native build artifacts"

install:
	@test -x "$(NATIVE_BIN)" || (echo "Build binary first: make build" >&2; exit 1)
	@install -d "$(DESTDIR)$(BINDIR)"
	@install -m 0755 "$(NATIVE_BIN)" "$(DESTDIR)$(BINDIR)/airllm"
	@echo "Installed $(DESTDIR)$(BINDIR)/airllm"

uninstall:
	@rm -f "$(DESTDIR)$(BINDIR)/airllm"
	@echo "Removed $(DESTDIR)$(BINDIR)/airllm"

release-archive:
	@test -n "$(VERSION)" || (echo "Set VERSION, e.g. make release-archive VERSION=v2.12.0 PLATFORM=linux-x86_64" >&2; exit 1)
	@test -n "$(PLATFORM)" || (echo "Set PLATFORM, e.g. PLATFORM=linux-x86_64" >&2; exit 1)
	@test -x "$(NATIVE_BIN)" || (echo "Build binary first: make build" >&2; exit 1)
	@mkdir -p dist/package
	@cp "$(NATIVE_BIN)" dist/package/airllm
	@cp LICENSE dist/package/LICENSE
	@cp Makefile dist/package/Makefile
	@cp air_llm/README.md dist/package/README.md
	@tar -C dist/package -czf "dist/airllm-$(VERSION)-$(PLATFORM).tar.gz" airllm LICENSE Makefile README.md
	@rm -rf dist/package
	@echo "Created dist/airllm-$(VERSION)-$(PLATFORM).tar.gz"
