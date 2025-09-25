#!/usr/bin/env bash
set -euo pipefail

# --- locate repo root ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

# --- settings ---
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
APP_ENTRY="${APP_ENTRY:-app.py}"

OPENAI_JSON="${OPENAI_JSON:-${SCRIPT_DIR}/openai_credentials.json}"
GOOGLE_JSON="${GOOGLE_JSON:-${SCRIPT_DIR}/google_credentials.json}"

# --- create venv if missing ---
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating venv at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# --- install deps ---
python -m pip install --upgrade pip >/dev/null
if [[ -f "$REQ_FILE" ]]; then
  echo "[setup] installing deps from $REQ_FILE"
  python -m pip install -r "$REQ_FILE"
fi

# --- credentials ---
# OpenAI: read api_key from JSON file if present and not already set
if [[ -z "${OPENAI_API_KEY:-}" && -f "$OPENAI_JSON" ]]; then
  OPENAI_API_KEY="$(python - <<PY
import json,sys
p = r"""$OPENAI_JSON"""
try:
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f) or {}
    v = j.get("api_key") or ""
    print(v)
except Exception:
    print("")
PY
)"
  if [[ -n "$OPENAI_API_KEY" ]]; then export OPENAI_API_KEY; fi
fi

# Google: point SDK to the JSON file if present
if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" && -f "$GOOGLE_JSON" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_JSON"
fi

# --- sanity output (non-secret) ---
echo "[env]  VENV: $VENV_DIR"
echo "[env]  OPENAI_API_KEY set: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo yes || echo no)"
echo "[env]  GOOGLE_APPLICATION_CREDENTIALS: ${GOOGLE_APPLICATION_CREDENTIALS:-<unset>}"
echo "[run]  python $APP_ENTRY"

# --- launch GUI ---
exec python "$APP_ENTRY"
