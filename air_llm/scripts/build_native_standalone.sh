#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/build/native"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

echo "Running from root directory: ${ROOT_DIR}"

echo "Using Python executable: ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import nuitka, zstandard' >/dev/null 2>&1; then
  echo "Installing build dependency: nuitka[onefile]" >&2
  "${PYTHON_BIN}" -m pip install --upgrade "nuitka[onefile]>=1.8"
fi

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PY_VERSION}" != "3.13" && "${AIRLLM_ALLOW_EXPERIMENTAL_PYTHON:-0}" != "1" ]]; then
  echo "Full standalone Nuitka builds are pinned to Python 3.13 for stability." >&2
  echo "Detected Python ${PY_VERSION}." >&2
  echo "Use Python 3.13 or set AIRLLM_ALLOW_EXPERIMENTAL_PYTHON=1 to proceed at your own risk." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -m nuitka \
  --standalone \
  --onefile \
  --assume-yes-for-downloads \
  --module-parameter=torch-disable-jit=no \
  --disable-plugin=transformers \
  --include-package=transformers \
  --output-dir="${OUT_DIR}" \
  --output-filename="airllm" \
  "${ROOT_DIR}/airllm_server.py"

echo "Standalone native executable created at ${OUT_DIR}/airllm"
