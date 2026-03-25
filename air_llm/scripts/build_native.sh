#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/build/native"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -m nuitka \
  --nofollow-imports \
  --output-dir="${OUT_DIR}" \
  --output-filename="airllm" \
  "${ROOT_DIR}/airllm_server.py"

echo "Native executable created at ${OUT_DIR}/airllm"
