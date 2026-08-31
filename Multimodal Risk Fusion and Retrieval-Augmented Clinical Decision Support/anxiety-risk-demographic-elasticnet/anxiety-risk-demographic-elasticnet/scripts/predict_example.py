from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxiety_risk.predict import load_bundle, predict_patient  # noqa: E402

bundle = load_bundle(ROOT / "models/demographic_elasticnet_calibrated.joblib")
result = predict_patient(
    bundle,
    age=22,
    gender="female",
    education="bachelor's degree",
    smoking="current smoker (cumulative smoking >10 packs)",
    drinking="current regular drinker (more than once a week)",
)
print(json.dumps(result, indent=2))
