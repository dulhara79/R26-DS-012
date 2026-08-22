"""
DCAR — Demographic & Contextual Anxiety Risk service
Component 4 · R26-DS-012 · https://github.com/dulhara79/R26-DS-012

Consumes the five demographic fields the patient app collects at enrolment and
returns a calibrated P(GAD-7 >= 10) plus the reliability metadata the fusion
layer needs.

This model is a POPULATION PRIOR, not a diagnostic instrument. It is scored once
per patient, at first login, and never changes thereafter.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artefacts/dcar_model.joblib"))
API_TOKEN = os.getenv("DCAR_API_TOKEN")  # set as a Space secret; unset = open (dev only)

BUNDLE = joblib.load(MODEL_PATH)
THRESHOLDS: List[int] = BUNDLE["thresholds"]
CUTOFF: int = BUNDLE["cutoff"]
BANDS: List[str] = BUNDLE["band_names"]
COLUMNS: List[str] = BUNDLE["columns"]
MEDIANS = BUNDLE["medians"]
REF_SCORES = np.asarray(BUNDLE["reference_scores"])

app = FastAPI(title="DCAR — Demographic Anxiety Risk", version=BUNDLE["version"])
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── request / response ───────────────────────────────────────────────────────
class DemographicRequest(BaseModel):
    patient_id: str = Field(..., description="MRN or participant code — used only for the audit log")
    gender: Optional[str] = None
    age: Optional[float] = None
    edu: Optional[str] = None
    smoke: Optional[str] = None
    drink: Optional[str] = None
    submitted_at: Optional[str] = Field(
        None, description="when the patient submitted this profile (ISO 8601). "
                          "If omitted, captured_at falls back to computed_at — "
                          "the backend should always send this once available.")


class DcarResponse(BaseModel):
    # ── common envelope (R26-DS-012_service_contracts.md §1) — every service
    # response carries these, so the backend never has to special-case ours.
    subject_id: str
    modality: str = "c4_demographic"
    status: str          # ok | poor_signal | error  (DCAR's status vocabulary is
                         # a subset of the full one — see note in predict() below)
    captured_at: str     # when the demographic profile was submitted
    computed_at: str     # when this inference ran
    latency_ms: int
    # ── component-specific fields ────────────────────────────────────────────
    patient_id: str      # kept for backward compatibility; identical to subject_id
    score: Optional[float]
    percentile: Optional[float]
    risk_label: Optional[str]
    threshold: float
    severity_probs: dict
    most_likely_band: Optional[str]
    expected_gad7: Optional[float]
    confidence: float
    coverage: float
    available: bool
    model_version: str


# ── feature construction — must mirror §4/§14 of the notebook exactly ────────
def _ordinal(value, rules):
    if value is None:
        return np.nan
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return np.nan
    for keys, v in rules:
        if any(k in s for k in keys):
            return float(v)
    return np.nan


def build_features(req: DemographicRequest) -> pd.DataFrame:
    row = {}
    for field, rules, col in (
        ("edu", BUNDLE["edu_order"], "edu_ord"),
        ("smoke", BUNDLE["smoke_order"], "smoke_ord"),
        ("drink", BUNDLE["drink_order"], "drink_ord"),
    ):
        v = _ordinal(getattr(req, field), rules)
        row[col] = MEDIANS[col] if np.isnan(v) else v
        row[f"{col}_missing"] = int(np.isnan(v))

    age = req.age if req.age is not None else np.nan
    age = float(age) if age == age else np.nan  # NaN-safe
    row["age"] = MEDIANS["age"] if np.isnan(age) else age
    row["age_missing"] = int(np.isnan(age))

    g = (req.gender or "").strip().lower()
    g = g if g in ("female", "male") else "other_unknown"
    for lvl in ("female", "male", "other_unknown"):
        row[f"gender_{lvl}"] = int(g == lvl)

    return pd.DataFrame([row])[COLUMNS]


def cumulative(X: pd.DataFrame) -> np.ndarray:
    """Calibrated P(total >= t) for each threshold, monotonicity enforced."""
    cols = []
    for t in THRESHOLDS:
        model = BUNDLE["threshold_models"].get(t)
        const = BUNDLE["threshold_constants"].get(t)
        iso = BUNDLE["isotonics"].get(t)
        raw = model.predict_proba(X)[:, 1] if model is not None else np.full(len(X), const)
        cols.append(raw if iso is None else iso.predict(raw))
    return np.minimum.accumulate(np.column_stack(cols), axis=1)


def band_probabilities(P: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [1 - P[:, 0], P[:, 0] - P[:, 1], P[:, 1] - P[:, 2], P[:, 2]]
    ).clip(0, 1)


# ── endpoints ────────────────────────────────────────────────────────────────
def _check_auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@app.get("/health")
def health():
    """The clinician app's Settings screen renders this. Never hard-code metrics in the UI."""
    return {
        "status": "ok",
        "model_version": BUNDLE["version"],
        "features": BUNDLE["features"],
        "target": f"P(GAD-7 total >= {CUTOFF})",
        "operating_threshold": BUNDLE["threshold"],
        "reference_n": int(REF_SCORES.size),
        "role": "population prior — not a diagnostic instrument",
    }


@app.post("/predict", response_model=DcarResponse)
def predict(req: DemographicRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    t0 = perf_counter()
    computed_at = datetime.now(timezone.utc).isoformat()

    X = build_features(req)
    P = cumulative(X)
    bands = band_probabilities(P)[0]
    score = float(P[0, THRESHOLDS.index(CUTOFF)])

    # confidence: 1 - normalised entropy over the four severity bands.
    # A flat distribution means this profile is one the model cannot resolve,
    # and the fusion gate must down-weight it accordingly.
    b = np.clip(bands, 1e-12, 1.0)
    entropy = float(-(b * np.log(b)).sum())
    confidence = float(1 - entropy / np.log(len(BANDS)))

    supplied = sum(getattr(req, f) is not None for f in ("gender", "age", "edu", "smoke", "drink"))
    coverage = supplied / 5.0

    # percentile against the frozen reference distribution — this is what makes
    # the score comparable with the physiological and clinical-text streams.
    percentile = float(np.searchsorted(REF_SCORES, score) / max(REF_SCORES.size, 1))

    # DCAR's status vocabulary is a subset of the full one (see
    # modality_clients.py in the central backend for the complete list): a
    # demographic form is either usable or it isn't, there is no "warming_up"
    # or "insufficient_data" equivalent for a static prior.
    available = coverage >= 0.6   # below 3 of 5 fields the score is not trustworthy
    status = "ok" if available else "poor_signal"

    return DcarResponse(
        subject_id=req.patient_id, modality="c4_demographic", status=status,
        captured_at=req.submitted_at or computed_at, computed_at=computed_at,
        latency_ms=int((perf_counter() - t0) * 1000),
        patient_id=req.patient_id,
        score=round(score, 4) if status == "ok" else None,
        percentile=round(percentile, 4) if status == "ok" else None,
        risk_label=("elevated" if score >= BUNDLE["threshold"] else "not elevated") if status == "ok" else None,
        threshold=round(float(BUNDLE["threshold"]), 4),
        severity_probs={b_: round(float(v), 4) for b_, v in zip(BANDS, bands)},
        most_likely_band=BANDS[int(bands.argmax())] if status == "ok" else None,
        expected_gad7=round(float(bands @ np.asarray(BUNDLE["band_midpoints"])), 2) if status == "ok" else None,
        confidence=round(confidence, 4),
        coverage=round(coverage, 4),
        available=available,
        model_version=BUNDLE["version"],
    )


@app.post("/fusion_component")
def fusion_component(req: DemographicRequest, authorization: Optional[str] = Header(None)):
    """Exact shape the fusion service expects for `c4_demographic`."""
    r = predict(req, authorization)
    return {
        "c4_demographic": {
            "score": r.score,
            "status": r.status,
            "available": r.available,
            "confidence": r.confidence,
            "coverage": r.coverage,
            "captured_at": r.captured_at,
            "computed_at": r.computed_at,
            "model_version": r.model_version,
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
