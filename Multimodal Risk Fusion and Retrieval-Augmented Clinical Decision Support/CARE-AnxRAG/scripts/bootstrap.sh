#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV=${VENV:-"$ROOT/.venv"}
EXTRAS=${EXTRAS:-production,dev}

"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -e "$ROOT[$EXTRAS]"

echo "Environment created at $VENV"
echo "Activate it with: source $VENV/bin/activate"
