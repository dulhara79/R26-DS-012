"""
Modality clients — the Central Backend's adapters to the four component Spaces.

Moved here from the fusion service, following the sequence diagram: the BACKEND
owns ingestion and calls each component's /predict (steps 11, 15, 19, 24); the
Fusion Service only fuses. That separation matters because it means the fusion
layer can be re-run over stored readings at any time without re-calling anyone.

The adapters follow the contracts exposed by the live component services:

    c1_physiological  GET /predict/{user_id} -> current_risk_index [0,100]

    c2_behavioral     excluded — stored, never fused, no Space called

    c3_clinical_nlp   target: POST /predict {note_text,...} -> calibrated_probability
                      legacy: POST /predict {note_text,...} -> risk_score
                      (same endpoint, superset of fields — no fallback branch needed,
                       just a field-preference order)

    c4_demographic    POST /fusion_component -> score [0,1] + confidence + coverage
                      (fully under our control — see hf_space/app.py)

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
import math
import os
from dataclasses import dataclass
from typing import Optional

import httpx

C1_BASE = os.getenv("C1_URL", "").rstrip("/")
C2_BASE = os.getenv("C2_URL", "").rstrip("/")
C3_BASE = os.getenv("C3_URL", "").rstrip("/")
C4_BASE = os.getenv("C4_URL", "").rstrip("/")

C1_TOKEN = os.getenv("C1_TOKEN", "")
C2_TOKEN = os.getenv("C2_TOKEN", "")
C3_TOKEN = os.getenv("C3_TOKEN", "")
C4_TOKEN = os.getenv("C4_TOKEN", "")

# Dulhara's live deployment (unlike the frozen contract) REJECTS an empty
# support_set with HTTP 422 MISSING_SUPPORT_SET instead of falling back to
# status="no_support_set". This generic, fixed default is a stand-in used
# ONLY when the caller supplies none. It is never built from the patient
# being scored (SupportNote's own docstring forbids that — leakage), so it
# is safe to reuse across every request. Flagged to Dulhara/Kaushalya as a
# live-vs-contract drift; not a permanent substitute for real site examples.
_C3_DEFAULT_SUPPORT_SET = [
    {"id": "default-anx-1", "label": "anxiety",
     "text": "Patient exhibits generalized anxiety with sleep disturbance, "
             "excessive worry, and restlessness over the past two weeks.",
     "note_date": "2026-01-15"},
    {"id": "default-ctrl-1", "label": "control",
     "text": "Patient presents for routine follow-up. No psychiatric "
             "complaints reported. Mood and affect stable.",
     "note_date": "2026-01-15"},
]

# ── C3 JWT auto-login ────────────────────────────────────────────────────────
# Dulhara's Space uses POST /auth/login → JWT (expires_in=43200s / 12h).
# This function logs in once, caches the token, and refreshes automatically
# when it's within 5 minutes of expiring. Credentials live in .env, never
# hardcoded. If C3_TOKEN is already set (e.g. a static key), that is used
# as-is and this login path is skipped entirely.
_C3_CLINICIAN_ID = os.getenv("C3_CLINICIAN_ID", "")
_C3_PASSWORD = os.getenv("C3_PASSWORD", "")
_c3_jwt_cache: dict = {"token": "", "expires_at": 0.0}

def _get_c3_token() -> str:
    import time
    # If a static token was set in .env, use it directly (no login needed)
    if C3_TOKEN:
        return C3_TOKEN
    # If no login credentials configured, return empty (will get 401)
    if not _C3_CLINICIAN_ID or not _C3_PASSWORD:
        return ""
    # Return cached JWT if still valid (with 5-minute buffer)
    now = time.time()
    if _c3_jwt_cache["token"] and now < _c3_jwt_cache["expires_at"] - 300:
        return _c3_jwt_cache["token"]
    # Log in fresh. HF Spaces sleep after inactivity — the first call can
    # take 30-60s while the container wakes. We use a 60s timeout and one
    # automatic retry with a 5s gap so a cold start doesn't fail the whole
    # clinical note ingestion during a live demo.
    for _attempt in range(2):
        try:
            with httpx.Client() as lc:
                r = lc.post(f"{C3_BASE}/auth/login",
                            json={"clinician_id": _C3_CLINICIAN_ID,
                                  "password": _C3_PASSWORD},
                            timeout=60.0)
                r.raise_for_status()
                body = r.json()
                _c3_jwt_cache["token"] = body["access_token"]
                _c3_jwt_cache["expires_at"] = now + body.get("expires_in", 43200)
                return _c3_jwt_cache["token"]
        except Exception:
            if _attempt == 0:
                time.sleep(5)
                continue
            _c3_jwt_cache["token"] = ""
            _c3_jwt_cache["expires_at"] = 0.0
            return ""


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


def confidence_from_entropy(entropy: Optional[float], n_classes: int = 2) -> Optional[float]:
    """Convert a predictive entropy into an honest confidence in [0, 1].

    WHY THIS EXISTS. C3 publishes a field literally named `confidence`, but its
    observed value equals the risk score itself (0.671 vs a score of 0.6715 in a
    real response). That is not a confidence — it is the probability restated.
    Feeding it into the reliability term c = 0.5 + 0.5*confidence*coverage would
    make a component's weight rise simply because its score rose, so a high score
    would inflate its own influence on the composite. Circular, and it biases the
    fusion exactly where it matters most (high-risk patients).

    `entropy`, when present, is a real uncertainty measure. Verified against a
    real C3 response: p=0.6715 -> H=0.6331 nats, and ln(2)=0.6931 is the binary
    maximum, so H is reported in NATS. Normalising gives 1 - 0.6331/0.6931 =
    0.087 — that prediction is barely better than a coin flip, which is the truth
    the raw `confidence` field obscures.
    """
    if entropy is None:
        return None
    try:
        h = float(entropy)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(h) or h < 0:
        return None
    h_max = math.log(n_classes)
    return float(min(max(1.0 - h / h_max, 0.0), 1.0))


def verify_subject_echo(sent: str, echoed, modality: str) -> Optional[str]:
    """Return an error string if a service echoed back a DIFFERENT subject id.

    This whole project's core safety property is that one patient's data never
    contaminates another's. If a component echoes a subject_id that isn't the one
    we asked about, we are looking at someone else's reading — a caching bug on
    their side, a race, or a mixed-up id map. Treat it as a hard error rather
    than storing it, because a silently mismatched reading is exactly the failure
    the patient-separation tests exist to prevent.
    """
    if echoed is None or sent is None:
        return None
    if str(echoed).strip() != str(sent).strip():
        return (f"SUBJECT MISMATCH: asked {modality} for '{sent}' but it returned "
                f"'{echoed}' — reading discarded to prevent cross-patient contamination")
    return None


def _parse_captured_at(value, fallback: dt.datetime) -> dt.datetime:
    if not value:
        return fallback
    try:
        v = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return fallback


def _call_c1_legacy(client: httpx.Client, user_id: str) -> ComponentResult:
    """GET /predict/{user_id} — the live C1 contract.

    1. STATUS MAPPING: not_calibrated/buffering -> warming_up, stale -> poor_signal
    2. captured_at IS latest_reading_at, not the time we fetched the response
    3. Coverage is NOT len(risk_forecast)/10 — C1 always returns 10 steps
       regardless of real history (padded with baseline). Record as unknown.
    4. confidence is NOT a C1 field. Record 0.5, not an estimate.
    5. model_version is NOT published by this contract. Record None.
    """
    r = client.get(f"{C1_BASE}/predict/{user_id}",
                   headers=_headers(C1_TOKEN), timeout=TIMEOUT_S)
    if r.status_code != 200:
        return ComponentResult(status="error",
                               note=f"HTTP {r.status_code} (C1 legacy endpoint)")
    body = r.json()

    c1_status = body.get("status", "")
    if c1_status == "success":
        our_status = "ok"
    elif c1_status in ("not_calibrated", "buffering"):
        our_status = "warming_up"
    elif c1_status == "stale":
        our_status = "poor_signal"
    else:
        our_status = "error"

    risk_index = body.get("current_risk_index")
    if our_status == "ok" and risk_index is None:
        our_status = "error"

    raw_score = float(risk_index) if our_status == "ok" else None

    note_parts = [f"C1 status='{c1_status}'"]
    if our_status != "ok":
        note_parts.append(body.get("message", "")[:120])
    note_parts.append("coverage: unknown (C1 pads history; len(forecast) always 10)")

    captured = _parse_captured_at(
        body.get("latest_reading_at"),
        dt.datetime.now(dt.timezone.utc))

    return ComponentResult(
        raw_score=raw_score, status=our_status,
        confidence=0.5,      # not published by C1
        coverage=0.5,        # not reliably computable from this response
        model_version=None,  # not published by C1
        detail=body,
        note="; ".join(p for p in note_parts if p),
        captured_at=captured)


def call_c1(user_id: str, window: Optional[dict] = None,
            client: Optional[httpx.Client] = None) -> ComponentResult:
    """Fetch C1's latest prediction for its participant/device user id.

    ``window`` remains accepted for compatibility with older callers, but C1
    receives its complete sensor features through its own ``/ingest`` endpoint.
    """
    if not C1_BASE:
        return ComponentResult(status="error", note="C1 not configured")
    own = client is None
    client = client or httpx.Client()
    try:
        try:
            return _call_c1_legacy(client, user_id)
        except httpx.TimeoutException:
            return ComponentResult(status="error", note=f"timeout {TIMEOUT_S}s (Space waking?)")
    except Exception as exc:                                # noqa: BLE001
        return ComponentResult(status="error", note=f"{type(exc).__name__}: {exc}"[:120])
    finally:
        if own:
            client.close()


# ── C2 behavioural ───────────────────────────────────────────────────────────
def call_c2(subject_external_id: str, payload: Optional[dict] = None,
            client: Optional[httpx.Client] = None) -> ComponentResult:
    """GET {C2_BASE}/api/score/{subject_external_id} — the real behavioural service.

    THREE INDEPENDENT LOCKS keep this out of the composite. Any one of them alone
    would be sufficient; all three are deliberate, because a single point of
    failure on an exclusion rule is how excluded data quietly gets fused:

      1. THEIR service reports status="not_validated" and fusion_eligible=false.
      2. OUR gate (gate.EXCLUDED_MODALITIES) drops c2_behavioral regardless of
         what any service says.
      3. fusion.py's CLEARS_PERMUTATION_NULL["c2_behavioral"] = False forces its
         base weight to exactly 0.0 even if it somehow reached the maths.

    THE TRAP THIS FUNCTION AVOIDS. A real C2 response carries BOTH
    `score: null` AND `behavioral_vulnerability_score: 0.0254`. The second field
    is an experimental signal their own response describes as "not a calibrated
    clinical anxiety probability". A naive integration greps for anything
    score-shaped, finds 0.0254, and fuses an explicitly uncalibrated number into
    a clinical composite. We read ONLY `score`, and we honour `fusion_eligible`.
    The experimental value is preserved in `detail` so the clinician timeline can
    show it, clearly labelled, without it ever touching the maths.
    """
    if not C2_BASE:
        return ComponentResult(status="error", note="C2 not configured")
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.get(f"{C2_BASE}/api/score/{subject_external_id}",
                       headers=_headers(C2_TOKEN), timeout=TIMEOUT_S)
        if r.status_code == 404:
            return ComponentResult(status="insufficient_data",
                                   note=f"C2 has no record for '{subject_external_id}' yet",
                                   model_version="c2")
        if r.status_code != 200:
            return ComponentResult(status="error", note=f"C2 HTTP {r.status_code}")
        body = r.json()

        mismatch = verify_subject_echo(subject_external_id, body.get("subject_id"), "C2")
        if mismatch:
            return ComponentResult(status="error", note=mismatch, detail=body)

        # Their status is authoritative for THEIR readiness; our exclusion rule is
        # authoritative for whether it is ever fused. Never coerce to "ok".
        status = body.get("status") or "not_validated"
        fusion_eligible = bool(body.get("fusion_eligible", False))
        if status == "ok" and not fusion_eligible:
            # Their service says the reading is fine but explicitly not fusable.
            # Record that faithfully instead of promoting it.
            status = "not_validated"

        coverage_blob = body.get("data_coverage") or {}
        coverage = coverage_blob.get("daily_feature_availability")
        try:
            coverage = float(coverage) if coverage is not None else 0.0
        except (TypeError, ValueError):
            coverage = 0.0

        note = body.get("reason") or body.get("score_semantics")
        if not fusion_eligible:
            note = (f"fusion_eligible=false — excluded from composite. "
                    f"{note or ''}").strip()

        return ComponentResult(
            # Deliberately ONLY body["score"] — never behavioral_vulnerability_score.
            raw_score=body.get("score") if status == "ok" and fusion_eligible else None,
            status=status,
            confidence=0.0,          # no calibrated confidence published by C2
            coverage=coverage,
            model_version=body.get("model_version", "c2"),
            detail=body,             # experimental value preserved here, not fused
            note=note,
            captured_at=_parse_captured_at(
                body.get("window_end") or body.get("computed_at"),
                dt.datetime.now(dt.timezone.utc)))
    except httpx.TimeoutException:
        return ComponentResult(status="error", note=f"C2 timeout {TIMEOUT_S}s")
    except Exception as exc:                                # noqa: BLE001
        return ComponentResult(status="error", note=f"C2 {type(exc).__name__}: {exc}"[:120])
    finally:
        if own:
            client.close()


# ── C3 clinical notes ────────────────────────────────────────────────────────
def call_c3(note_text: str, note_type: str = "progress",
            anxiety_support: Optional[list] = None,
            control_support: Optional[list] = None,
            support_set: Optional[list] = None,
            support_set_version: Optional[str] = None,
            note_date: Optional[str] = None,
            visit_count: Optional[int] = None,
            subject_external_id: Optional[str] = None,
            client: Optional[httpx.Client] = None) -> ComponentResult:
    """POST /predict {note_text, note_type, support sets}.

    Score preference order — calibrated_probability > risk_score > score. The
    paper (Multimodal Digital Biomarker Framework, §III.C) supports treating
    the SCORE this way: TC-WPN applies "a learnable temperature parameter
    [that] scales the prototype distances before the final softmax
    probabilities are obtained" — temperature scaling is a real calibration
    mechanism, so preferring a calibrated field over the raw cosine distance
    is well-founded.

    ⚠️ UNCONFIRMED — the `confidence` field below is NOT covered by that same
    justification. The paper explicitly states its analogous per-support-note
    weight is "referred to as prototype consistency rather than confidence
    because it is neither calibrated nor an estimate of label uncertainty"
    (§III.C). Two things are unclear and need Dulhara to confirm before this
    is trusted further: (1) whether the live API's `confidence` field is even
    the same quantity the paper calls prototype consistency, since the paper
    describes it as a SUPPORT-set weight used to build the class prototype,
    not a per-query inference-time field; (2) if it is, using it as a
    reliability weight in fusion's `c = 0.5 + 0.5*confidence*coverage` may be
    weighting by a quantity the paper's own authors say carries no calibration
    guarantee. NOT changed here pending that answer — see PAPER_ALIGNMENT.md
    item C3-2. Do not "fix" this by guessing a replacement.
    """
    if not C3_BASE:
        return ComponentResult(status="error", note="C3 not configured")
    if not note_text or not note_text.strip():
        return ComponentResult(status="error", note="no clinical note supplied")
    own = client is None
    client = client or httpx.Client()
    try:
        # Kaushalya's frozen contract uses one unified support_set list, not
        # two separate arrays. Build it from the legacy anxiety_support /
        # control_support params when support_set isn't given directly, so
        # existing callers (and the 145-test suite) keep working unchanged.
        if support_set is None:
            support_set = (
                [{"id": f"legacy-anx-{i}", "text": t, "label": "anxiety"}
                 for i, t in enumerate(anxiety_support or [])]
                + [{"id": f"legacy-ctrl-{i}", "text": t, "label": "control"}
                   for i, t in enumerate(control_support or [])]
            )
        request_body = {"note_text": note_text, "note_type": note_type,
                        "support_set": support_set,
                        "return_attention": True,
                        "return_support_contributions": True}
        if note_date:
            request_body["note_date"] = note_date
        if visit_count is not None:
            request_body["visit_count"] = visit_count
        if support_set_version:
            request_body["support_set_version"] = support_set_version
        if subject_external_id:
            request_body["subject_id"] = subject_external_id
        used_default_support = False
        if not request_body["support_set"]:
            request_body["support_set"] = _C3_DEFAULT_SUPPORT_SET
            used_default_support = True
        r = client.post(f"{C3_BASE}/predict", headers=_headers(_get_c3_token()),
                        json=request_body, timeout=TIMEOUT_S)
        if r.status_code != 200:
            return ComponentResult(status="error", note=f"HTTP {r.status_code}")
        body = r.json()

        score = body.get("calibrated_probability")
        score_source = "calibrated_probability"
        if score is None:
            # Dulhara's live deployment names this field "probability" and
            # reports calibration_status="uncalibrated" alongside it — a
            # real field, but not the calibrated one the contract promises.
            score = body.get("probability")
            cal_status = body.get("calibration_status", "unknown")
            score_source = f"probability (calibration_status={cal_status})"
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
        if "risk_score" in score_source or "probability" in score_source:
            note = (note + "; " if note else "") + score_source
        if used_default_support:
            note = (note + "; " if note else "") + "used generic default support set (no site-specific examples supplied)"

        mismatch = verify_subject_echo(subject_external_id, body.get("subject_id"), "C3")
        if mismatch:
            return ComponentResult(status="error", note=mismatch, detail=body)

        # Prefer entropy (a genuine uncertainty measure) over C3's own
        # `confidence` field. In a real observed response C3 reported
        # confidence=0.671 alongside risk_score=0.6715 — the confidence IS the
        # score restated. Using it in c = 0.5 + 0.5*confidence*coverage would let
        # a rising score inflate its own weight, biasing fusion precisely on
        # high-risk patients. Entropy has no such circularity: verified on the
        # same response, p=0.6715 -> H=0.6331 nats, ln(2)=0.6931 max, so the
        # honest confidence is 0.087, not 0.671.
        entropy_conf = confidence_from_entropy(body.get("entropy"))
        if entropy_conf is not None:
            confidence = entropy_conf
            conf_source = f"entropy-derived {entropy_conf:.3f} (H={body.get('entropy')})"
        else:
            confidence = float(body.get("confidence", 0.5))
            conf_source = "C3 confidence field (no entropy published — see docstring)"
        note = (note + "; " if note else "") + f"confidence: {conf_source}"

        return ComponentResult(
            raw_score=float(score) if status == "ok" else None, status=status,
            confidence=confidence, coverage=1.0,
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
    "c2_behavioral": lambda: bool(C2_BASE),
    "c3_clinical_nlp": lambda: bool(C3_BASE),
    "c4_demographic": lambda: bool(C4_BASE),
}
