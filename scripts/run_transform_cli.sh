#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

exec "${PY_BIN}" "${ROOT_DIR}/transformer_service.py" "$@"
