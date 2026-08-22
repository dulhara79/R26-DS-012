
"""
score_service.py
Component 2 scoring layer.

Expected local files:
    feature_extractor.py
    m2_mobile_model.joblib
    m2_mobile_model_metadata.json

This service:
1. Builds the 240-feature mobile-compatible vector
2. Checks data sufficiency / coverage
3. Loads the trained Logistic Regression pipeline
4. Produces an EXPERIMENTAL behavioral vulnerability score
5. Keeps the fusion gate explicit
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from feature_extractor import (
    build_feature_vector,
    validate_against_model_metadata,
)

MODEL_PATH = Path("m2_mobile_model.joblib")
METADATA_PATH = Path("m2_mobile_model_metadata.json")

# Engineering gates — not clinical thresholds.
MIN_HISTORY_DAYS = 14
MIN_DAILY_FEATURE_AVAILABILITY = 0.35

# Safe default: generate/store the experimental score, but do NOT allow it into
# active fusion until the team explicitly enables it after review.
ENABLE_C2_FUSION = os.getenv("ENABLE_C2_FUSION", "0").strip() == "1"

_model = None
_metadata = None


def load_artifacts():
    global _model, _metadata

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {MODEL_PATH.resolve()}"
            )
        _model = joblib.load(MODEL_PATH)

    if _metadata is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Missing metadata artifact: {METADATA_PATH.resolve()}"
            )
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

    return _model, _metadata


def _base_response(
    participant_id: str,
    status: str,
    *,
    score: Optional[float] = None,
    raw_score: Optional[float] = None,
    fusion_eligible: bool = False,
    coverage: Optional[Dict[str, Any]] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    _, metadata = load_artifacts()

    return {
        "subject_id": participant_id,
        "modality": "c2_behavioral",
        "score": score,
        "status": status,
        "fusion_eligible": fusion_eligible,
        "behavioral_vulnerability_score": raw_score,
        "data_coverage": coverage or {},
        "window_start": window_start,
        "window_end": window_end,
        "reason": reason,
        "model_version": metadata.get(
            "model_version",
            "M2_mobile_screen_location_v1",
        ),
        "validation_status": "experimental",
        "score_semantics": (
            "Experimental behavioral vulnerability signal; "
            "not a calibrated clinical anxiety probability."
        ),
    }


def score_participant_events(
    rows: pd.DataFrame | Iterable[Dict[str, Any]],
    participant_id: str,
    window_end_date: str,
    normalization_rows: Optional[
        pd.DataFrame | Iterable[Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """
    Score one participant using raw mobile sensor events.

    Important:
    - `rows` should contain at least the previous 28 days.
    - `window_end_date` is EXCLUSIVE.
      Example: 2026-08-20 uses data through 2026-08-19.
    """
    model, metadata = load_artifacts()

    result = build_feature_vector(
        rows=rows,
        participant_id=participant_id,
        window_end_date=window_end_date,
        normalization_rows=normalization_rows,
    )

    validate_against_model_metadata(result, metadata)

    coverage = result.coverage

    if not coverage.get("minimum_history_met", False):
        return _base_response(
            participant_id,
            "insufficient_data",
            score=None,
            raw_score=None,
            fusion_eligible=False,
            coverage=coverage,
            window_start=result.window_start,
            window_end=result.window_end,
            reason=(
                f"Only {coverage.get('days_with_any_data', 0)} days with data; "
                f"minimum required is {MIN_HISTORY_DAYS}."
            ),
        )

    if coverage.get("daily_feature_availability", 0.0) < MIN_DAILY_FEATURE_AVAILABILITY:
        return _base_response(
            participant_id,
            "poor_signal",
            score=None,
            raw_score=None,
            fusion_eligible=False,
            coverage=coverage,
            window_start=result.window_start,
            window_end=result.window_end,
            reason=(
                "Daily feature availability below engineering threshold "
                f"{MIN_DAILY_FEATURE_AVAILABILITY:.2f}."
            ),
        )

    X = result.as_model_input()
    raw_score = float(model.predict_proba(X)[0, 1])

    # The evaluation metadata says whether the model is a candidate for
    # experimental fusion. Active fusion still requires an explicit server-side
    # switch so it cannot be enabled accidentally.
    recommended = bool(
        metadata.get("fusion_eligible_recommendation", False)
    )

    if recommended and ENABLE_C2_FUSION:
        return _base_response(
            participant_id,
            "ok",
            score=raw_score,
            raw_score=raw_score,
            fusion_eligible=True,
            coverage=coverage,
            window_start=result.window_start,
            window_end=result.window_end,
            reason="Experimental C2 fusion explicitly enabled by backend configuration.",
        )

    return _base_response(
        participant_id,
        "not_validated",
        score=None,
        raw_score=raw_score,
        fusion_eligible=False,
        coverage=coverage,
        window_start=result.window_start,
        window_end=result.window_end,
        reason=(
            "Experimental behavioral score generated and stored, but active "
            "fusion is disabled. Set ENABLE_C2_FUSION=1 only after team review."
        ),
    )


if __name__ == "__main__":
    print("Component 2 scoring service")
    print("Model path   :", MODEL_PATH.resolve())
    print("Metadata path:", METADATA_PATH.resolve())
    print("Active C2 fusion enabled:", ENABLE_C2_FUSION)
