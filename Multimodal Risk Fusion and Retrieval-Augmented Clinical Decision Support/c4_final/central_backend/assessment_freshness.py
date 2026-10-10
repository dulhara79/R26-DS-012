"""Read-time validity of a persisted fusion snapshot (handbook section 19).

Never recompute a stored composite with fresh modality data: a fusion result is
immutable evidence. A current projection is unavailable when its source
snapshot no longer meets the original model's eligibility gate.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select

import gate
from db_models import ModalityReading


def current_status(db, fusion, *, as_of=None):
    """Return (status, reason) without modifying historical fusion rows."""
    if fusion is None:
        return "unavailable", "no fusion assessment"
    now = as_of or dt.datetime.now(dt.timezone.utc)
    snap = ((fusion.harmonisation or {}).get("source_reading_ids"))
    recorded = ((fusion.harmonisation or {}).get("assessment") or {}).get("status")
    base = {"complete": "complete", "provisional": "partial",
            "insufficient": "unavailable"}.get(
                recorded, "partial" if fusion.composite is not None else "unavailable")
    # Historical pre-snapshot assessments cannot be proven current. They may
    # be displayed as history, but must not be called current risk.
    if not isinstance(snap, dict) or not snap:
        return "unavailable", "source reading provenance unavailable"
    ids = [v for v in snap.values() if isinstance(v, int)]
    rows = db.scalars(select(ModalityReading).where(
        ModalityReading.subject_id == fusion.subject_id,
        ModalityReading.id.in_(ids))).all() if ids else []
    readings = {r.modality: {
        "raw_score": r.raw_score, "status": r.status,
        "confidence": r.confidence, "coverage": r.coverage,
        "captured_at": r.captured_at,
    } for r in rows if snap.get(r.modality) == r.id}
    decision = gate.evaluate(readings, now=now)
    if not decision.passed:
        return "unavailable", "fusion evidence no longer usable: " + (decision.reason or "gate failed")
    # The snapshot is a historical immutable composite. If any included
    # component aged out, its stored weighting is no longer valid. Fail closed.
    originally = {m for m in (fusion.weights or {}) if m not in gate.EXCLUDED_MODALITIES}
    if not originally.issubset(decision.usable):
        return "unavailable", "one or more original fusion contributors expired"
    return base, None
