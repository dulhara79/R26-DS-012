"""
Fusion orchestration service — Component 4 · R26-DS-012

Pipeline per request:

    1. fan out to the four component Spaces in parallel        (clients.py)
    2. harmonise each raw score to its percentile               (harmonise.py)
    3. weight by informativeness x recency x reliability        (fusion.py)
    4. renormalise over available modalities, band, return

Run locally:   uvicorn app:app --reload --port 7861
Docs:          http://127.0.0.1:7861/docs
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import clients
from fusion import LiveFusion, MODALITIES, Reading, base_weights, fuse
from harmonise import Harmoniser

API_TOKEN = os.getenv("FUSION_API_TOKEN")
harmoniser = Harmoniser()

app = FastAPI(title="Multimodal Risk Fusion", version="ragf-v0.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# In-memory state. FINE FOR THE PILOT, NOT FOR DEPLOYMENT — restart wipes it and
# a second worker process gets its own copy. Replace with Postgres before NHSL.
_live: Dict[str, LiveFusion] = {}
_last: Dict[str, Dict[str, Reading]] = {}


def _auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "invalid or missing bearer token")


# ── request models ───────────────────────────────────────────────────────────
class FuseRequest(BaseModel):
    mrn: str
    demographics: Optional[dict] = Field(
        None, description="passed through to C4 if you want it rescored; normally omitted")
    explain: bool = True


class ManualComponent(BaseModel):
    score: Optional[float] = None
    available: bool = True
    confidence: float = 0.5
    coverage: float = 1.0
    captured_at: Optional[datetime] = None


class ManualFuseRequest(BaseModel):
    mrn: str
    components: Dict[str, ManualComponent]
    already_harmonised: bool = False


class PhysioTick(BaseModel):
    mrn: str
    score: float
    confidence: float = 0.7
    coverage: float = 1.0
    captured_at: Optional[datetime] = None


# ── core ─────────────────────────────────────────────────────────────────────
def _harmonise_all(readings: Dict[str, Reading], skip: bool = False):
    """Percentile-map every score. Returns (readings, audit trail)."""
    audit, out = {}, {}
    for m in MODALITIES:
        r = readings.get(m, Reading())
        if not r.available or r.score is None:
            out[m] = r
            audit[m] = {"available": False, "note": r.note}
            continue
        if skip:
            out[m] = r
            audit[m] = {"raw": r.score, "harmonised": r.score, "note": "harmonisation skipped"}
            continue

        h = harmoniser.harmonise(m, r.score)
        out[m] = Reading(score=h.value, available=True, confidence=r.confidence,
                         coverage=r.coverage, captured_at=r.captured_at, note=r.note)
        audit[m] = {"raw": round(h.raw, 4), "harmonised": round(h.value, 4),
                    "drift": h.drift, "note": h.note}
    return out, audit


def _respond(readings: Dict[str, Reading], audit: dict, mrn: str, live: bool = False):
    if live:
        out = _live.setdefault(mrn, LiveFusion()).update(readings)
    else:
        out = fuse(readings)

    wire = out.to_wire()
    wire["mrn"] = mrn
    wire["harmonisation"] = audit
    wire["base_weights"] = {k: round(v, 4) for k, v in base_weights().items()}
    wire["model_version"] = app.version
    wire["computed_locally"] = False
    return wire


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": app.version,
        "base_weights": {k: round(v, 4) for k, v in base_weights().items()},
        "component_endpoints": {m: bool(u) for m, u in clients.ENDPOINTS.items()},
        "reference_distributions": harmoniser.available(),
        "component_timeout_s": clients.TIMEOUT_S,
        "warning": ("no reference distribution loaded for some modalities — "
                    "cross-modality comparison is not yet valid")
                   if len(harmoniser.available()) < 3 else None,
        "state_backend": "in-memory (pilot only — replace with Postgres)",
    }


@app.post("/v1/fuse")
async def fuse_live(req: FuseRequest, authorization: Optional[str] = Header(None)):
    """Call all four Spaces, harmonise, fuse. This is what the clinician app hits."""
    _auth(authorization)
    extra = {"c4_demographic": req.demographics} if req.demographics else {}
    readings = await clients.collect(req.mrn, extra)
    _last[req.mrn] = readings
    harmonised, audit = _harmonise_all(readings)
    return _respond(harmonised, audit, req.mrn)


@app.post("/v1/fuse/manual")
def fuse_manual(req: ManualFuseRequest, authorization: Optional[str] = Header(None)):
    """Fuse scores you supply directly. Use this to test before the Spaces exist,
    and to reproduce any composite from your paper without a network call."""
    _auth(authorization)
    readings = {
        m: Reading(score=c.score, available=c.available and c.score is not None,
                   confidence=c.confidence, coverage=c.coverage,
                   captured_at=c.captured_at or datetime.now(timezone.utc))
        for m, c in req.components.items()
    }
    harmonised, audit = _harmonise_all(readings, skip=req.already_harmonised)
    return _respond(harmonised, audit, req.mrn)


@app.post("/v1/physio/tick")
def physio_tick(tick: PhysioTick, authorization: Optional[str] = Header(None)):
    """C1 posts here every minute. Smoothed and hysteresis-gated so the ward
    display does not flicker; the other three streams are reused from the last
    full fuse rather than re-fetched 1,440 times a day."""
    _auth(authorization)
    readings = dict(_last.get(tick.mrn, {}))
    readings["c1_physiological"] = Reading(
        score=tick.score, available=True, confidence=tick.confidence,
        coverage=tick.coverage,
        captured_at=tick.captured_at or datetime.now(timezone.utc))
    _last[tick.mrn] = readings
    harmonised, audit = _harmonise_all(readings)
    return _respond(harmonised, audit, tick.mrn, live=True)


@app.get("/v1/patients/{mrn}/state")
def patient_state(mrn: str, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    readings = _last.get(mrn)
    if not readings:
        raise HTTPException(404, f"no readings cached for {mrn}")
    harmonised, audit = _harmonise_all(readings)
    return _respond(harmonised, audit, mrn)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7861)))
