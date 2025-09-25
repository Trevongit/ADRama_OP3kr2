#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== ADRama TTS Studio — start ==="

# 1) Ensure venv exists
if [ ! -d ".venv" ]; then
  echo "No .venv found. Please run: ./install_adramaz.sh"
  exit 1
fi

# 2) Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) Load secrets from .env file if it exists
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# 4) Launch the app
echo "Running app.py ..."
exec python app.py