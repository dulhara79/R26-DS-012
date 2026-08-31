#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
if [ ! -f reports/results/best_params.json ]; then
  echo "No tuned parameters found; running tuning first."
  PYTHONPATH=src python scripts/tune_model.py
else
  echo "Using existing tuned parameters from reports/results/best_params.json"
fi
PYTHONPATH=src python scripts/train_final_model.py
pytest -q
