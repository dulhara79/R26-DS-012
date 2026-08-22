"""
⚠️ PARTIALLY SUPERSEDED — /v1/fuse (which uses this module) is no longer the
primary path. The Central Backend now owns calling C1/C3/C4 and stores every
reading (central_backend/modality_clients.py); it calls THIS service's
/v1/fuse/manual endpoint with scores it already has, in FUSION_MODE=http. Keep
this file for standalone testing of the fusion service, but new integration
work should go through the backend, not through clients.py.

Fan-out to the four component services.

Design rule: A COMPONENT THAT DOES NOT ANSWER IN TIME IS MISSING, NOT ZERO.

Every call gets its own timeout. A slow or sleeping Hugging Face Space must never
block the clinician's screen, and it must never be silently scored as 0.0 — that
would read as "this patient has no physiological risk", which is a dangerous lie.
It becomes an unavailable modality, the masked renormalisation in fusion.py
redistributes its weight, and the response says `renormalised: true`.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx

from fusion import Reading

# ── endpoint configuration ───────────────────────────────────────────────────
# Set these in .env or as Space secrets. Leave a URL blank and that component is
# treated as permanently unavailable, which is exactly right for C2.
ENDPOINTS = {
    "c1_physiological": os.getenv("C1_URL", ""),
    "c2_behavioral":    os.getenv("C2_URL", ""),
    "c3_clinical_nlp":  os.getenv("C3_URL", ""),
    "c4_demographic":   os.getenv("C4_URL", ""),
}

TOKENS = {
    "c1_physiological": os.getenv("C1_TOKEN", ""),
    "c2_behavioral":    os.getenv("C2_TOKEN", ""),
    "c3_clinical_nlp":  os.getenv("C3_TOKEN", ""),
    "c4_demographic":   os.getenv("C4_TOKEN", ""),
}

TIMEOUT_S = float(os.getenv("COMPONENT_TIMEOUT_S", "8"))


def _parse_time(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def to_reading(modality: str, payload: dict) -> Reading:
    """Accept the several shapes teammates might send, without arguing about it.

    Tolerated: {"score": x}, {"risk_score": x}, {"c1_physiological": {...}},
    {"result": {"score": x}}. Missing metadata falls back to conservative
    defaults rather than optimistic ones — an unspecified confidence becomes
    0.5, not 1.0, because a component that does not report its uncertainty has
    not earned full weight.
    """
    if not isinstance(payload, dict):
        return Reading(available=False, note="response was not a JSON object")

    body = payload
    for key in (modality, "result", "data", "output"):
        if isinstance(body.get(key), dict):
            body = body[key]
            break

    score = None
    for key in ("score", "risk_score", "value", "probability", "risk"):
        if isinstance(body.get(key), (int, float)):
            score = float(body[key])
            break
    if score is None:
        return Reading(available=False, note=f"no numeric score field in {list(body)[:6]}")

    available = bool(body.get("available", True))
    return Reading(
        score=score,
        available=available,
        confidence=float(body.get("confidence", 0.5)),
        coverage=float(body.get("coverage", 1.0)),
        captured_at=_parse_time(body.get("captured_at") or body.get("computed_at")
                                or body.get("timestamp")) or datetime.now(timezone.utc),
        note=body.get("model_version"),
    )


async def _fetch_one(client: httpx.AsyncClient, modality: str, payload: dict) -> Reading:
    url = ENDPOINTS.get(modality, "")
    if not url:
        return Reading(available=False, note="no endpoint configured")

    headers = {"Content-Type": "application/json"}
    if TOKENS.get(modality):
        headers["Authorization"] = f"Bearer {TOKENS[modality]}"

    try:
        r = await client.post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
        if r.status_code != 200:
            return Reading(available=False, note=f"HTTP {r.status_code}")
        return to_reading(modality, r.json())
    except httpx.TimeoutException:
        return Reading(available=False, note=f"timeout after {TIMEOUT_S}s")
    except Exception as exc:                       # noqa: BLE001 — never let one Space crash the fuse
        return Reading(available=False, note=f"{type(exc).__name__}: {exc}"[:120])


async def collect(mrn: str, extra: Optional[Dict[str, dict]] = None) -> Dict[str, Reading]:
    """Call all four components in parallel. Total wall time ~= the slowest one."""
    extra = extra or {}
    base = {"patient_id": mrn, "mrn": mrn}

    async with httpx.AsyncClient() as client:
        modalities = list(ENDPOINTS)
        results = await asyncio.gather(*[
            _fetch_one(client, m, {**base, **extra.get(m, {})}) for m in modalities
        ])
    return dict(zip(modalities, results))
