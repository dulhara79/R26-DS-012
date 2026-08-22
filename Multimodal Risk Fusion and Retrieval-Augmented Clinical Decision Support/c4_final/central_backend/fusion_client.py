"""
Fusion client — step 29 of the sequence diagram: POST /fuse with the scores for
ONE subject only.

Two modes, because you need both:

  http        calls the separate Fusion Service. This is the deployed topology in
              the diagram, and it is what you should demonstrate.
  inprocess   imports fusion.py directly. Same maths, no network. Use it for
              tests and for reproducing any composite in your paper without
              standing up a second service.

The mode is set by FUSION_MODE. Default is `inprocess` so the backend runs out of
the box; switch to `http` once the Fusion Service is deployed.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Dict, Optional

import httpx

FUSION_MODE = os.getenv("FUSION_MODE", "inprocess").lower()
FUSION_URL = os.getenv("FUSION_URL", "http://127.0.0.1:7861").rstrip("/")
FUSION_TOKEN = os.getenv("FUSION_API_TOKEN", "")
FUSION_TIMEOUT_S = float(os.getenv("FUSION_TIMEOUT_S", "20"))

BAND_FOR_TIER = {"Low": "GREEN", "Medium": "AMBER", "High": "RED", None: "GREY"}


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        v = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        return v.isoformat()
    return str(value)


def fuse(subject_id: str, readings: Dict[str, dict]) -> dict:
    """readings: {modality: {raw_score, confidence, coverage, captured_at}}

    Returns the fusion service's wire response, with `band` guaranteed present.
    """
    components = {
        m: {
            "score": r.get("raw_score"),
            "available": True,
            "confidence": float(r.get("confidence", 0.5)),
            "coverage": float(r.get("coverage", 1.0)),
            "captured_at": _iso(r.get("captured_at")),
        }
        for m, r in readings.items()
    }

    if FUSION_MODE == "http":
        result = _fuse_http(subject_id, components)
    else:
        result = _fuse_inprocess(components)

    result.setdefault("band", BAND_FOR_TIER.get(result.get("tier"), "GREY"))
    result["subject_id"] = subject_id
    return result


def _fuse_http(subject_id: str, components: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if FUSION_TOKEN:
        headers["Authorization"] = f"Bearer {FUSION_TOKEN}"
    r = httpx.post(f"{FUSION_URL}/v1/fuse/manual", headers=headers,
                   json={"mrn": subject_id, "components": components,
                         "already_harmonised": False},
                   timeout=FUSION_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _fuse_inprocess(components: dict) -> dict:
    """Import the fusion service's own modules so the maths is IDENTICAL.

    Deliberately not a reimplementation — a second copy of the weighting logic
    would drift from the first and silently produce different numbers in tests
    than in deployment.
    """
    import sys
    from pathlib import Path

    fusion_dir = Path(os.getenv("FUSION_SERVICE_DIR",
                                Path(__file__).resolve().parent.parent / "fusion_service"))
    if str(fusion_dir) not in sys.path:
        sys.path.insert(0, str(fusion_dir))

    from fusion import MODALITIES, Reading, base_weights, fuse as fuse_core  # noqa: E402
    from harmonise import Harmoniser  # noqa: E402

    global _HARMONISER
    try:
        _HARMONISER
    except NameError:
        _HARMONISER = Harmoniser(fusion_dir / "reference")

    readings, audit = {}, {}
    for m in MODALITIES:
        c = components.get(m)
        if not c or c.get("score") is None:
            readings[m] = Reading(available=False)
            audit[m] = {"available": False}
            continue
        h = _HARMONISER.harmonise(m, c["score"])
        captured = c.get("captured_at")
        if isinstance(captured, str):
            captured = dt.datetime.fromisoformat(captured.replace("Z", "+00:00"))
        readings[m] = Reading(score=h.value, available=True,
                              confidence=c.get("confidence", 0.5),
                              coverage=c.get("coverage", 1.0),
                              captured_at=captured)
        audit[m] = {"raw": round(h.raw, 4), "harmonised": round(h.value, 4),
                    "drift": h.drift, "note": h.note}

    out = fuse_core(readings).to_wire()
    out["harmonisation"] = audit
    out["base_weights"] = {k: round(v, 4) for k, v in base_weights().items()}
    out["model_version"] = "ragf-v0.4-inprocess"
    return out
