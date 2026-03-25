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

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  echo "Bootstrapping pip in build environment..." >&2
  "${PYTHON_BIN}" -m ensurepip --upgrade
fi

if ! "${PYTHON_BIN}" -c 'import nuitka, zstandard' >/dev/null 2>&1; then
  echo "Installing build dependency: nuitka[onefile]" >&2
  "${PYTHON_BIN}" -m pip install --upgrade "nuitka[onefile]>=1.8"
fi

TRANSFORMERS_VERSION="$(${PYTHON_BIN} -c 'import importlib.util as u; import importlib.metadata as m; print(m.version("transformers") if u.find_spec("transformers") else "")')"

# Optional escape hatch if users explicitly want to force the historical pin.
if [[ -n "${TRANSFORMERS_VERSION}" && "${AIRLLM_STANDALONE_PIN_TRANSFORMERS_443:-0}" == "1" ]]; then
  echo "Forcing transformers==4.43.3 for standalone build (AIRLLM_STANDALONE_PIN_TRANSFORMERS_443=1)." >&2
  "${PYTHON_BIN}" -m pip install --upgrade "transformers==4.43.3"
fi
# Work around a Nuitka optimization crash seen with transformers 5.x (walrus operator in generation/utils.py).
if [[ -n "${TRANSFORMERS_VERSION}" && "${TRANSFORMERS_VERSION}" =~ ^5\. && "${AIRLLM_STANDALONE_SKIP_TF5_PATCH:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -c 'from pathlib import Path; import transformers.generation.utils as gu; p = Path(gu.__file__); old = "        if (position_ids := kwargs.pop(position_ids_key, None)) is not None:"; new = "        position_ids = kwargs.pop(position_ids_key, None)\n        if position_ids is not None:"; s = p.read_text(encoding="utf-8");
if old in s:
    p.write_text(s.replace(old, new, 1), encoding="utf-8")'
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
  --output-dir="${OUT_DIR}" \
  --output-filename="airllm" \
  "${ROOT_DIR}/airllm_server.py"

echo "Standalone native executable created at ${OUT_DIR}/airllm"
