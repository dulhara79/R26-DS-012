"""Convenience wrapper: tune first, then train/evaluate the final model."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src")

subprocess.run([sys.executable, str(ROOT / "scripts/tune_model.py")], cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/train_final_model.py")], cwd=ROOT, env=env, check=True)
