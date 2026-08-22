"""
Fake teammate components — run this so you can test the fusion service TODAY,
before C1, C2 and C3 exist.

    uvicorn mock_components:app --reload --port 7900

Then point your .env at it:

    C1_URL=http://127.0.0.1:7900/c1/predict
    C2_URL=http://127.0.0.1:7900/c2/predict
    C3_URL=http://127.0.0.1:7900/c3/predict

Each mock deliberately behaves like a DIFFERENT badly-behaved service, because
that is what you will actually get:

  C1 — raw score on an unbounded scale (reconstruction error), full metadata
  C2 — returns a score but is meant to be ignored (zero weight)
  C3 — irregular timestamps, different field name ("risk_score"), no coverage
  /slow  — takes 20 seconds, to prove your timeout works
  /broken — returns HTTP 500, to prove one dead Space does not kill the fuse
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock components (TESTING ONLY)")
random.seed(7)


class Req(BaseModel):
    patient_id: str | None = None
    mrn: str | None = None


def _now():
    return datetime.now(timezone.utc).isoformat()


@app.post("/c1/predict")
def c1(req: Req):
    """Physiological. Raw anomaly score on an unbounded scale — NOT a probability.
    This is exactly why harmonisation exists."""
    return {
        "score": round(random.uniform(0.02, 0.35), 4),   # reconstruction error
        "available": True,
        "confidence": round(random.uniform(0.6, 0.9), 3),
        "coverage": round(random.uniform(0.5, 1.0), 3),  # wear time
        "captured_at": _now(),
        "model_version": "c1-lstm-ae-v1.3",
    }


@app.post("/c2/predict")
def c2(req: Req):
    """Behavioural. Answers confidently. Gets zero weight anyway, because it did
    not clear its permutation null. Good test that the rule actually holds."""
    return {
        "score": round(random.uniform(0.7, 0.99), 4),
        "available": True,
        "confidence": 0.95,
        "coverage": 1.0,
        "captured_at": _now(),
        "model_version": "c2-gatv2-v0.9",
    }


@app.post("/c3/predict")
def c3(req: Req):
    """Clinical notes. Different field name, no coverage field, note is days old.
    The client has to cope with all three."""
    age_days = random.choice([1, 5, 30, 95])
    return {
        "risk_score": round(random.uniform(0.3, 0.85), 4),
        "confidence": round(random.uniform(0.7, 0.9), 3),
        "computed_at": (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat(),
        "model_version": "tcwpn-v2.1",
        "_note_age_days": age_days,
    }


@app.post("/slow/predict")
async def slow(req: Req):
    await asyncio.sleep(20)
    return {"score": 0.5}


@app.post("/broken/predict")
def broken(req: Req):
    raise HTTPException(500, "simulated component failure")
