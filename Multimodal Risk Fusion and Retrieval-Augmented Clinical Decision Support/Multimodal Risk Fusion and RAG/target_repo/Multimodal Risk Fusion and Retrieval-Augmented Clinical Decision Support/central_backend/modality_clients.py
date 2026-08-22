"""
Modality clients — the Central Backend's adapters to the four component Spaces.

Moved here from the fusion service, following the sequence diagram: the BACKEND
owns ingestion and calls each component's /predict (steps 11, 15, 19, 24); the
Fusion Service only fuses. That separation matters because it means the fusion
layer can be re-run over stored readings at any time without re-calling anyone.

Two contract generations are handled for C1 and C3, because the frozen service
contract (R26-DS-012_service_contracts.md) describes the TARGET shape each Space
is meant to converge on, which is not necessarily what is live today:

    c1_physiological  target: POST /predict {window, features} -> score [0,1]
                              + status (ok|warming_up|poor_signal|error)
                      legacy: GET  /predict/{user_id} -> current_risk_index [0,100]

    c2_behavioral     excluded — stored, never fused, no Space called

    c3_clinical_nlp   target: POST /predict {note_text,...} -> calibrated_probability
                      legacy: POST /predict {note_text,...} -> risk_score
                      (same endpoint, superset of fields — no fallback branch needed,
                       just a field-preference order)

    c4_demographic    POST /fusion_component -> score [0,1] + confidence + coverage
                      (fully under our control — see hf_space/app.py)

Each adapter tries the target contract first and falls back to the legacy shape
only where the two are structurally different (C1's GET vs POST). Once a
component ships the target contract, its legacy branch becomes dead code that
can be deleted — it is kept isolated in its own function for exactly that reason.

Rule that governs all of them: A COMPONENT THAT DOES NOT ANSWER IS MISSING, NOT
ZERO. A timeout is recorded with status='error' and no score, never as 0.0 —
scoring a sleeping Space as zero would read as "this patient has no risk", which
is a dangerous lie.

Status vocabulary (R26-DS-012_service_contracts.md, appendix), passed through
verbatim wherever a component reports its own status rather than inferred:

    ok               valid reading                          -> in composite
    warming_up       C1 personal baseline not learned yet    -> excluded
    insufficient_data C2 has fewer than 42 days of history   -> excluded
    poor_signal      input quality below threshold           -> excluded
    no_support_set   C3 has zero support examples (K=0)      -> excluded
    not_validated    model not yet clinically validated (C2) -> excluded
    error            service failed, timed out, or malformed -> excluded

gate.py rejects anything that is not exactly "ok"; it does not need to know this
vocabulary, so adding a new status value here needs no gate change.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional

import httpx

C1_BASE = os.getenv("C1_URL", "").rstrip("/")
C3_BASE = os.getenv("C3_URL", "").rstrip("/")
C4_BASE = os.getenv("C4_URL", "").rstrip("/")

C1_TOKEN = os.getenv("C1_TOKEN", "")
C3_TOKEN = os.getenv("C3_TOKEN", "")
C4_TOKEN = os.getenv("C4_TOKEN", "")

TIMEOUT_S = float(os.getenv("COMPONENT_TIMEOUT_S", "30"))

VALID_STATUSES = {"ok", "warming_up", "insufficient_data", "poor_signal",
                  "no_support_set", "not_validated", "error"}


@dataclass
class ComponentResult:
    """What one component said. Mirrors a row of modality_readings."""
    raw_score: Optional[float] = None
    status: str = "error"                 # see VALID_STATUSES above
    confidence: float = 0.5
    coverage: float = 1.0
    model_version: Optional[str] = None
    detail: Optional[dict] = None
    note: Optional[str] = None
    captured_at: dt.datetime = None       # type: ignore[assignment]

    def __post_init__(self):
        if self.captured_at is None:
            self.captured_at = dt.datetime.now(dt.timezone.utc)
        if self.status not in VALID_STATUSES:
            # Don't hide an unrecognised status — record it as an error with the
            # original value preserved, rather than silently coercing it to "ok".
            self.note = f"unrecognised status '{self.status}' from component ({self.note or ''})".strip()
            self.status = "error"


def _headers(token: str) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_captured_at(value, fallback: dt.datetime) -> dt.datetime:
    if not value:
        return fallback
    try:
        v = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return fallback


# ── C1 physiological ─────────────────────────────────────────────────────────
def _call_c1_target_contract(client: httpx.Client, subject_id: str,
                             window: dict) -> Optional[ComponentResult]:
    """POST /predict per the frozen contract. Returns None (not an error result)
    on HTTP 404 specifically, which is the signal to try the legacy endpoint —
    every other failure is a real error and is returned as such."""
    r = client.post(f"{C1_BASE}/predict", headers=_headers(C1_TOKEN),
                    json={"subject_id": subject_id, **window}, timeout=TIMEOUT_S)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        return ComponentResult(status="error", note=f"HTTP {r.status_code}")

    body = r.json()
    now = dt.datetime.now(dt.timezone.utc)
    status = body.get("status", "ok" if body.get("score") is not None else "error")

    # baseline_ready is the authoritative override: a personal baseline that
    # hasn't converged makes the score noise regardless of what `status` says.
    if body.get("baseline_ready") is False:
        status = "warming_up"

    score = body.get("score")
    raw_score = float(score) if (status == "ok" and score is not None) else None

    # Confidence: prefer signal_quality (a real published field in the target
    # contract) over the old distance-from-threshold proxy.
    if body.get("signal_quality") is not None:
        confidence = float(body["signal_quality"])
    else:
        err, thr = body.get("reconstruction_error"), body.get("threshold")
        confidence = 0.6
        if err is not None and thr:
            confidence = float(min(max(1.0 - abs(err - thr) / max(thr, 1e-6), 0.3), 0.9))

    seen, required = body.get("baseline_windows_seen"), body.get("baseline_windows_required")
    coverage = min(seen / required, 1.0) if (seen and required) else 1.0

    return ComponentResult(
        raw_score=raw_score, status=status, confidence=confidence, coverage=coverage,
        model_version=body.get("model_version", "c1"), detail=body,
        note=None if status == "ok" else
             f"{status}" + (f" ({body.get('baseline_windows_seen')}/{body.get('baseline_windows_required')} windows)"
                            if status == "warming_up" and body.get("baseline_windows_required") else ""),
        captured_at=_parse_captured_at(body.get("captured_at"), now))


def _call_c1_legacy(client: httpx.Client, user_id: str) -> ComponentResult:
    """GET /predict/{user_id} -> current_risk_index on a 0-100 scale.

    This is what is actually live today per the repo. The 0-100 scale is fine —
    the fusion service percentile-maps it against C1's own reference
    distribution, so absolute scale never enters the composite. DELETE this
    function once C1 ships the target contract above.
    """
    r = client.get(f"{C1_BASE}/predict/{user_id}", headers=_headers(C1_TOKEN), timeout=TIMEOUT_S)
    if r.status_code != 200:
        return ComponentResult(status="error", note=f"HTTP {r.status_code} (legacy endpoint)")
    body = r.json()
    if body.get("status") != "success" or body.get("current_risk_index") is None:
        return ComponentResult(status="error",
                               note=f"no forecast: {str(body.get('message',''))[:80]} (legacy)",
                               detail=body)

    horizon = len(body.get("risk_forecast") or [])
    coverage = min(horizon / 10.0, 1.0) if horizon else 0.5

    err, thr = body.get("current_reconstruction_error"), body.get("reconstruction_error_threshold")
    confidence = 0.6
    if err is not None and thr:
        confidence = float(min(max(1.0 - abs(err - thr) / max(thr, 1e-6), 0.3), 0.9))

    return ComponentResult(
        raw_score=float(body["current_risk_index"]), status="ok",
        confidence=confidence, coverage=coverage,
        model_version=body.get("model_version", "c1-legacy"), detail=body,
        note="legacy endpoint (current_risk_index, 0-100 scale)")


def call_c1(subject_id: str, window: Optional[dict] = None,
            client: Optional[httpx.Client] = None) -> ComponentResult:
    """Target contract first (if `window` supplied), legacy GET as fallback.

    `window` should carry window_start, window_end, sampling_hz, features — the
    same fields the ingestion endpoint receives from the chest strap. Omit it to
    go straight to the legacy path (e.g. in a demo where the app only sends a
    bare tick).
    """
    if not C1_BASE:
        return ComponentResult(status="error", note="C1 not configured")
    own = client is None
    client = client or httpx.Client()
    try:
        if window:
            try:
                result = _call_c1_target_contract(client, subject_id, window)
            except httpx.TimeoutException:
                return ComponentResult(status="error", note=f"timeout {TIMEOUT_S}s (Space waking?)")
            if result is not None:
                return result
            # target contract returned 404 -> component hasn't shipped it yet, fall through

        try:
            return _call_c1_legacy(client, subject_id)
        except httpx.TimeoutException:
            return ComponentResult(status="error", note=f"timeout {TIMEOUT_S}s (Space waking?)")
    except Exception as exc:                                # noqa: BLE001
        return ComponentResult(status="error", note=f"{type(exc).__name__}: {exc}"[:120])
    finally:
        if own:
            client.close()


# ── C2 behavioural ───────────────────────────────────────────────────────────
def call_c2(payload: dict, client: Optional[httpx.Client] = None) -> ComponentResult:
    """Stored for the record, never fused.

    We deliberately do NOT call a remote service here. The component did not
    exceed its permutation null (AUROC 0.5205 vs 0.4991, p = 0.255), so it is
    recorded as `not_validated` and the gate excludes it. Keeping the reading
    visible in the timeline — rather than dropping it silently — is what makes
    the exclusion auditable. When validation lands, this becomes a real client
    with a target/legacy split like C1's, and the backend needs zero other
    changes — the gate already excludes purely on `status`.
    """
    days_seen = payload.get("days_of_history") or payload.get("days")
    status = "not_validated"
    note = "withheld: did not exceed permutation null (AUROC 0.5205 vs 0.4991, p=0.255)"
    if isinstance(days_seen, int) and days_seen < 42:
        status = "insufficient_data"
        note = f"insufficient_data: {days_seen}/42 days of history"

    return ComponentResult(
        raw_score=None, status=status, confidence=0.0, coverage=0.0,
        model_version="c2-withheld", note=note, detail={"observations": payload})


# ── C3 clinical notes ────────────────────────────────────────────────────────
def call_c3(note_text: str, note_type: str = "progress",
            anxiety_support: Optional[list] = None,
            control_support: Optional[list] = None,
            client: Optional[httpx.Client] = None) -> ComponentResult:
    """POST /predict {note_text, note_type, support sets}.

    Score preference order — calibrated_probability > risk_score > score — since
    the target contract's rule is explicit: fusion consumes the CALIBRATED
    probability, never the raw cosine-derived score, because raw scores are not
    comparable across runs.
    """
    if not C3_BASE:
        return ComponentResult(status="error", note="C3 not configured")
    if not note_text or not note_text.strip():
        return ComponentResult(status="error", note="no clinical note supplied")
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.post(f"{C3_BASE}/predict", headers=_headers(C3_TOKEN),
                        json={"note_text": note_text, "note_type": note_type,
                              "anxiety_support": anxiety_support or [],
                              "control_support": control_support or []},
                        timeout=TIMEOUT_S)
        if r.status_code != 200:
            return ComponentResult(status="error", note=f"HTTP {r.status_code}")
        body = r.json()

        score = body.get("calibrated_probability")
        score_source = "calibrated_probability"
        if score is None:
            score = body.get("risk_score")
            score_source = "risk_score (calibrated_probability not sent — ask C3 to add it)"
        if score is None:
            score = body.get("score")
            score_source = "score"
        if score is None:
            return ComponentResult(status="error",
                                   note=f"no score field in {list(body)[:8]}", detail=body)

        support_k = body.get("support_k")
        status = body.get("status")
        if not status:
            status = "no_support_set" if support_k == 0 else "ok"
        elif support_k == 0 and status == "ok":
            # Component said ok but also told us K=0 — trust the more specific signal.
            status = "no_support_set"

        note = None
        if status != "ok":
            note = f"{status}" + (f" (support_k={support_k})" if support_k is not None else "")
        if "risk_score" in score_source:
            note = (note + "; " if note else "") + score_source

        return ComponentResult(
            raw_score=float(score) if status == "ok" else None, status=status,
            confidence=float(body.get("confidence", 0.5)), coverage=1.0,
            model_version=body.get("model_version", "tc-wpn"), detail=body, note=note,
            captured_at=_parse_captured_at(body.get("captured_at") or body.get("note_date"),
                                           dt.datetime.now(dt.timezone.utc)))
    except httpx.TimeoutException:
        return ComponentResult(status="error", note=f"timeout {TIMEOUT_S}s (Space waking?)")
    except Exception as exc:                                # noqa: BLE001
        return ComponentResult(status="error", note=f"{type(exc).__name__}: {exc}"[:120])
    finally:
        if own:
            client.close()


# ── C4 demographic (yours) ───────────────────────────────────────────────────
def call_c4(subject_id: str, demographics: dict,
            client: Optional[httpx.Client] = None) -> ComponentResult:
    """POST /fusion_component -> score + confidence + coverage.

    Called ONCE per patient, at enrolment, when the demographic profile and
    GAD-7 arrive from the patient app. The score never changes afterwards.
    This is our own Space (hf_space/app.py) — it already emits the common
    envelope's status field, so it is trusted verbatim rather than inferred.
    """
    if not C4_BASE:
        return ComponentResult(status="error", note="C4 not configured")
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.post(f"{C4_BASE}/fusion_component", headers=_headers(C4_TOKEN),
                        json={"patient_id": subject_id, **(demographics or {})},
                        timeout=TIMEOUT_S)
        if r.status_code != 200:
            return ComponentResult(status="error", note=f"HTTP {r.status_code}")
        body = r.json()
        block = body.get("c4_demographic", body)
        if block.get("score") is None:
            return ComponentResult(status="error", note="no score field", detail=body)

        status = block.get("status")
        if not status:
            status = "ok" if block.get("available", True) else "poor_signal"

        return ComponentResult(
            raw_score=float(block["score"]) if status == "ok" else None,
            status=status,
            confidence=float(block.get("confidence", 0.5)),
            coverage=float(block.get("coverage", 1.0)),
            model_version=block.get("model_version", "dcar"), detail=body,
            note=None if status == "ok" else "coverage too low to be usable",
            captured_at=_parse_captured_at(block.get("captured_at") or block.get("computed_at"),
                                           dt.datetime.now(dt.timezone.utc)))
    except httpx.TimeoutException:
        return ComponentResult(status="error", note=f"timeout {TIMEOUT_S}s")
    except Exception as exc:                                # noqa: BLE001
        return ComponentResult(status="error", note=f"{type(exc).__name__}: {exc}"[:120])
    finally:
        if own:
            client.close()


CONFIGURED = {
    "c1_physiological": lambda: bool(C1_BASE),
    "c2_behavioral": lambda: True,        # local, always "available"
    "c3_clinical_nlp": lambda: bool(C3_BASE),
    "c4_demographic": lambda: bool(C4_BASE),
}
