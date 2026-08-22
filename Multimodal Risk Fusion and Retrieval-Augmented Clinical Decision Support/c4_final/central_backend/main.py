"""
Central Backend — Component 4 · R26-DS-012

Implements the integration sequence diagram end to end.

    ENROLMENT          steps 1-9    identity, pairing, patient separation
    PASSIVE MODALITIES steps 10-21  physiological / behavioural / contextual
    CLINICAL MODALITY  steps 22-26  note enters from the clinician side only
    FUSION             steps 27-31  gate, fuse, persist
    EGRESS             steps 32-35  two views, one source of truth

Run:  uvicorn main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

import conformal
import fusion_client
import gate
import identity
import modality_clients as mc
import rag_client
from db_models import (AuditLog, FusionResult, ModalityReading, PairingCode,
                       Subject, SubjectAlias, Verdict, get_session, init_db, utcnow)

API_TOKEN = os.getenv("BACKEND_API_TOKEN", "")
ALL_MODALITIES = ["c1_physiological", "c2_behavioral", "c3_clinical_nlp", "c4_demographic"]

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Central Backend — R26-DS-012", version="cb-v1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Also create tables at import time. The lifespan handler covers `uvicorn main:app`,
# but a bare TestClient(app) never triggers lifespan, and a missing-table error at
# that point is confusing to debug. create_all is idempotent, so doing both is safe.
init_db()


def _auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "invalid or missing bearer token")


def _audit(db: Session, subject_id: Optional[str], event: str,
           detail: Optional[dict] = None, actor: Optional[str] = None):
    db.add(AuditLog(subject_id=subject_id, event=event, detail=detail, actor=actor))


def _resolve(db: Session, alias_type: str, alias_value: str) -> str:
    row = db.scalar(select(SubjectAlias).where(
        SubjectAlias.alias_type == alias_type,
        SubjectAlias.alias_value == alias_value))
    if not row:
        raise HTTPException(404, f"no subject for that {alias_type}")
    return row.subject_id


# Each component keys patients differently: C2 uses ids like "P_65DC4002E7863773",
# C1 streams under a device id, C3 may use its own. SubjectAlias already supports
# arbitrary alias_type values, so external ids need no schema migration — they are
# just more aliases pointing at the same subject_id. This is the same join problem
# the MRN/app_user_id pairing flow solves, extended to the component services.
EXTERNAL_ID_TYPES = {
    "c1_physiological": "c1_device_id",
    "c2_behavioral": "c2_subject_id",
    "c3_clinical_nlp": "c3_patient_id",
}


def _external_id(db: Session, subject_id: str, modality: str) -> str:
    """The id THIS component knows the patient by, falling back to our own
    subject_id when no mapping is registered (correct for services that simply
    accept whatever id we send)."""
    alias_type = EXTERNAL_ID_TYPES.get(modality)
    if not alias_type:
        return subject_id
    row = db.scalar(select(SubjectAlias).where(
        SubjectAlias.subject_id == subject_id,
        SubjectAlias.alias_type == alias_type))
    return row.alias_value if row else subject_id


class ExternalIdRequest(BaseModel):
    modality: str = Field(..., description="c1_physiological | c2_behavioral | c3_clinical_nlp")
    external_id: str = Field(..., min_length=1)


@app.post("/v1/subjects/{subject_id}/external-ids", tags=["enrolment"])
def register_external_id(subject_id: str, req: ExternalIdRequest,
                         db: Session = Depends(get_session),
                         authorization: Optional[str] = Header(None)):
    """Tell the backend what id a given component knows this patient by.

    Without this, the backend would ask C2 about a UUID that C2 has never heard
    of. Registering is idempotent: re-registering the same modality updates the
    mapping rather than creating a duplicate alias."""
    _auth(authorization)
    _require_subject(db, subject_id)
    alias_type = EXTERNAL_ID_TYPES.get(req.modality)
    if not alias_type:
        raise HTTPException(422, f"modality must be one of {sorted(EXTERNAL_ID_TYPES)}")

    clash = db.scalar(select(SubjectAlias).where(
        SubjectAlias.alias_type == alias_type,
        SubjectAlias.alias_value == req.external_id))
    if clash and clash.subject_id != subject_id:
        # The same component id already points at a different patient. Refusing is
        # the only safe answer — silently repointing would merge two clinical records.
        raise HTTPException(409, f"external_id '{req.external_id}' is already mapped "
                                 f"to a different subject")

    existing = db.scalar(select(SubjectAlias).where(
        SubjectAlias.subject_id == subject_id, SubjectAlias.alias_type == alias_type))
    if existing:
        existing.alias_value = req.external_id
    else:
        db.add(SubjectAlias(subject_id=subject_id, alias_type=alias_type,
                            alias_value=req.external_id))
    _audit(db, subject_id, "external_id.registered",
           {"modality": req.modality, "alias_type": alias_type})
    db.commit()
    return {"subject_id": subject_id, "modality": req.modality,
            "external_id": req.external_id}


def _require_subject(db: Session, subject_id: str) -> Subject:
    s = db.get(Subject, subject_id)
    if not s:
        raise HTTPException(404, f"unknown subject_id {subject_id}")
    if s.status != "active":
        raise HTTPException(409, f"subject is {s.status}")
    return s


# ═════════════════════════════════════════════════════════════════════════════
# ENROLMENT — steps 1-9
# ═════════════════════════════════════════════════════════════════════════════
class EnrolRequest(BaseModel):
    mrn: str = Field(..., description="Raw MRN. Hashed immediately; never stored.")
    enrolled_by: Optional[str] = None


class EnrolResponse(BaseModel):
    subject_id: str
    pairing_code: str
    expires_at: dt.datetime


@app.post("/v1/subjects", response_model=EnrolResponse, tags=["enrolment"])
def enrol_subject(req: EnrolRequest, db: Session = Depends(get_session),
                  authorization: Optional[str] = Header(None)):
    """Steps 2-4. Clinician enrols a patient by MRN.

    The MRN is HMAC-hashed on arrival and the raw value is never persisted.
    Re-enrolling the same MRN returns the existing subject with a fresh pairing
    code, rather than creating a duplicate patient.
    """
    _auth(authorization)
    try:
        mrn_hash = identity.hash_mrn(req.mrn)
    except identity.PepperNotConfigured as exc:
        raise HTTPException(500, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    existing = db.scalar(select(SubjectAlias).where(
        SubjectAlias.alias_type == "mrn_hash", SubjectAlias.alias_value == mrn_hash))

    if existing:
        subject_id = existing.subject_id
        _audit(db, subject_id, "enrol.repeat", {"note": "existing MRN, new pairing code"},
               req.enrolled_by)
    else:
        subject_id = identity.new_subject_id()
        db.add(Subject(subject_id=subject_id, enrolled_by=req.enrolled_by))
        db.add(SubjectAlias(subject_id=subject_id, alias_type="mrn_hash",
                            alias_value=mrn_hash))
        _audit(db, subject_id, "enrol.created", {"alias": "mrn_hash"}, req.enrolled_by)

    code = identity.new_pairing_code()
    expires = identity.pairing_expiry()
    db.add(PairingCode(code=code, subject_id=subject_id, expires_at=expires))
    db.commit()

    return EnrolResponse(subject_id=subject_id, pairing_code=code, expires_at=expires)


class PairRequest(BaseModel):
    pairing_code: str
    app_user_id: str


class PairResponse(BaseModel):
    subject_id: str


@app.post("/v1/subjects/pair", response_model=PairResponse, tags=["enrolment"])
def pair_subject(req: PairRequest, db: Session = Depends(get_session)):
    """Steps 7-9. Patient app redeems the code, attaching app_user_id to the
    SAME subject_id. This is the join that makes every downstream reading land
    on the right patient."""
    code = db.get(PairingCode, req.pairing_code.strip().upper())
    if not code:
        raise HTTPException(404, "unknown pairing code")
    if code.used_at is not None:
        raise HTTPException(409, "pairing code already used")
    if identity.is_expired(code.expires_at):
        raise HTTPException(410, "pairing code expired — ask the clinician for a new one")

    clash = db.scalar(select(SubjectAlias).where(
        SubjectAlias.alias_type == "app_user_id",
        SubjectAlias.alias_value == req.app_user_id))
    if clash and clash.subject_id != code.subject_id:
        # This phone is already attached to a different patient. Refusing is the
        # only safe answer — silently re-pointing it would cross two records.
        raise HTTPException(409, "this app_user_id is already paired to another subject")

    if not clash:
        db.add(SubjectAlias(subject_id=code.subject_id, alias_type="app_user_id",
                            alias_value=req.app_user_id))
    code.used_at = utcnow()
    _audit(db, code.subject_id, "enrol.paired", {"app_user_id": req.app_user_id})
    db.commit()
    return PairResponse(subject_id=code.subject_id)


@app.get("/v1/subjects/resolve", tags=["enrolment"])
def resolve_subject(app_user_id: Optional[str] = None, mrn: Optional[str] = None,
                    db: Session = Depends(get_session),
                    authorization: Optional[str] = Header(None)):
    """Look up a subject_id from either alias. The clinician app uses the MRN
    form; the patient app uses app_user_id."""
    _auth(authorization)
    if app_user_id:
        return {"subject_id": _resolve(db, "app_user_id", app_user_id)}
    if mrn:
        return {"subject_id": _resolve(db, "mrn_hash", identity.hash_mrn(mrn))}
    raise HTTPException(422, "supply app_user_id or mrn")


# ═════════════════════════════════════════════════════════════════════════════
# INGESTION — steps 10-26
# ═════════════════════════════════════════════════════════════════════════════

# ── auto-trigger (the diagram's "FUSION — triggered by the event") ───────────
# Contextual and clinical-note ingests fuse IMMEDIATELY — they are rare events
# (once at enrolment; daily-to-monthly respectively). The physiological stream
# arrives every 60 seconds, and fusing on every tick would write 1,440 rows per
# patient per day, so physio-triggered fusion is DEBOUNCED: it only fires if at
# least AUTO_FUSION_DEBOUNCE_MIN minutes have passed since this subject's last
# fusion of any kind. Behavioural ingests never trigger — the stream is excluded
# from the composite by pre-registered rule, so a fusion after it could not
# change the answer and would only add noise rows to the trend.
AUTO_FUSION_DEBOUNCE_MIN = float(os.getenv("AUTO_FUSION_DEBOUNCE_MIN", "5"))


def _minutes_since_last_fusion(db: Session, subject_id: str) -> Optional[float]:
    last = db.scalar(select(FusionResult.computed_at)
                     .where(FusionResult.subject_id == subject_id)
                     .order_by(FusionResult.computed_at.desc(), FusionResult.id.desc())
                     .limit(1))
    if last is None:
        return None
    if last.tzinfo is None:                      # SQLite drops tzinfo on round-trip
        last = last.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - last).total_seconds() / 60.0


def _auto_fuse(db: Session, subject_id: str, trigger: str,
               debounce: bool = False) -> dict:
    """Run fusion after an ingest. Never lets a fusion failure fail the ingest —
    the reading is already committed, and losing it because the composite step
    hiccuped would violate the append-only design."""
    if debounce:
        since = _minutes_since_last_fusion(db, subject_id)
        if since is not None and since < AUTO_FUSION_DEBOUNCE_MIN:
            return {"fusion_triggered": False,
                    "fusion_skipped_reason": (f"debounced: last fusion {since:.1f} min ago, "
                                              f"threshold {AUTO_FUSION_DEBOUNCE_MIN:g} min")}
    try:
        row = run_fusion(db, subject_id, trigger)
        return {"fusion_triggered": True,
                "fusion": {"composite": row.composite, "tier": row.tier,
                           "band": row.band, "reason": row.reason}}
    except Exception as exc:                     # noqa: BLE001
        return {"fusion_triggered": False,
                "fusion_error": f"{type(exc).__name__}: {exc}"[:160]}
def _store(db: Session, subject_id: str, modality: str,
           result: mc.ComponentResult) -> ModalityReading:
    row = ModalityReading(
        subject_id=subject_id, modality=modality, raw_score=result.raw_score,
        status=result.status, confidence=result.confidence, coverage=result.coverage,
        captured_at=result.captured_at, model_version=result.model_version,
        detail={"note": result.note, "response": result.detail})
    db.add(row)
    _audit(db, subject_id, f"ingest.{modality}",
           {"status": result.status, "score": result.raw_score, "note": result.note})
    return row


class PhysiologicalWindow(BaseModel):
    app_user_id: Optional[str] = None
    subject_id: Optional[str] = None
    device_user_id: Optional[str] = Field(
        None, description="id the chest strap streams under; defaults to subject_id")
    # Target-contract fields (R26-DS-012_service_contracts.md §2). All optional:
    # omit them and the client falls back to the legacy GET endpoint that's
    # actually live today. Once C1 ships the target contract, the patient app
    # should start sending these on every 60s window.
    window_start: Optional[dt.datetime] = None
    window_end: Optional[dt.datetime] = None
    sampling_hz: Optional[int] = None
    features: Optional[Dict[str, float]] = Field(
        None, description="the 10 features exactly as C1 was trained on: mean HR, "
                          "mean RR interval, SDNN, RMSSD, mean breathing rate, "
                          "breathing-rate variability, mean temperature, temperature "
                          "variability, mean acceleration magnitude, acceleration "
                          "variability (paper §III.A — was documented as 11 before "
                          "the paper's exact feature list was available)")


@app.post("/v1/ingest/physiological", tags=["ingestion"])
def ingest_physiological(req: PhysiologicalWindow, db: Session = Depends(get_session),
                         authorization: Optional[str] = Header(None)):
    """Steps 10-13. Chest-strap window, every 60 seconds."""
    _auth(authorization)
    subject_id = req.subject_id or _resolve(db, "app_user_id", req.app_user_id or "")
    _require_subject(db, subject_id)

    window = None
    if req.features:
        window = {
            "window_start": req.window_start.isoformat() if req.window_start else None,
            "window_end": req.window_end.isoformat() if req.window_end else None,
            "sampling_hz": req.sampling_hz,
            "features": req.features,
        }

    result = mc.call_c1(req.device_user_id or _external_id(db, subject_id, "c1_physiological"),
                        window=window)
    row = _store(db, subject_id, "c1_physiological", result)
    db.commit()
    fusion_info = _auto_fuse(db, subject_id, "physio-ingest", debounce=True)
    return {"subject_id": subject_id, "reading_id": row.id,
            "status": result.status, "score": result.raw_score, "note": result.note,
            **fusion_info}


class BehaviouralAggregate(BaseModel):
    app_user_id: Optional[str] = None
    subject_id: Optional[str] = None
    observations: Dict[str, Any] = Field(default_factory=dict)


@app.post("/v1/ingest/behavioural", tags=["ingestion"])
def ingest_behavioural(req: BehaviouralAggregate, db: Session = Depends(get_session),
                       authorization: Optional[str] = Header(None)):
    """Steps 14-17. Stored with status `not_validated` and excluded from the
    composite by pre-registered rule. Kept visible so the exclusion is auditable."""
    _auth(authorization)
    subject_id = req.subject_id or _resolve(db, "app_user_id", req.app_user_id or "")
    _require_subject(db, subject_id)

    result = mc.call_c2(_external_id(db, subject_id, "c2_behavioral"), req.observations)
    row = _store(db, subject_id, "c2_behavioral", result)
    db.commit()
    return {"subject_id": subject_id, "reading_id": row.id,
            "status": result.status,
            # Explicitly null rather than omitted: a caller inspecting this
            # response should see that there is no fusable score, not have to
            # infer it from a missing key. The experimental
            # behavioral_vulnerability_score lives in the stored detail blob and
            # is deliberately NOT surfaced here as a score.
            "score": result.raw_score,
            "excluded_from_composite": True, "note": result.note}


class ContextualIntake(BaseModel):
    app_user_id: Optional[str] = None
    subject_id: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[float] = None
    edu: Optional[str] = None
    smoke: Optional[str] = None
    drink: Optional[str] = None
    gad7_items: Optional[List[int]] = Field(
        None, description="the SEVEN item responses, 0-3 each. The total is "
                          "recomputed server-side; a client-sent total is display only.")


@app.post("/v1/ingest/contextual", tags=["ingestion"])
def ingest_contextual(req: ContextualIntake, db: Session = Depends(get_session),
                      authorization: Optional[str] = Header(None)):
    """Steps 18-21. Demographics + GAD-7 from the patient app, scored once by
    your DCAR model."""
    _auth(authorization)
    subject_id = req.subject_id or _resolve(db, "app_user_id", req.app_user_id or "")
    _require_subject(db, subject_id)

    gad7_total = None
    if req.gad7_items is not None:
        if len(req.gad7_items) != 7 or not all(0 <= v <= 3 for v in req.gad7_items):
            raise HTTPException(422, "gad7_items must be exactly 7 values, each 0-3")
        gad7_total = sum(req.gad7_items)

    demographics = {k: v for k, v in
                    {"gender": req.gender, "age": req.age, "edu": req.edu,
                     "smoke": req.smoke, "drink": req.drink}.items() if v is not None}

    result = mc.call_c4(subject_id, demographics)
    if result.detail is None:
        result.detail = {}
    result.detail["gad7_total"] = gad7_total
    result.detail["gad7_items"] = req.gad7_items

    row = _store(db, subject_id, "c4_demographic", result)
    db.commit()
    fusion_info = _auto_fuse(db, subject_id, "contextual-ingest")
    return {"subject_id": subject_id, "reading_id": row.id, "status": result.status,
            "score": result.raw_score, "gad7_total": gad7_total, "note": result.note,
            **fusion_info}


class ClinicalNote(BaseModel):
    subject_id: Optional[str] = None
    mrn: Optional[str] = None
    note_text: str
    note_type: str = "progress"
    anxiety_support: List[str] = Field(default_factory=list)
    control_support: List[str] = Field(default_factory=list)
    author: Optional[str] = None


@app.post("/v1/clinical-notes", tags=["ingestion"])
def ingest_clinical_note(req: ClinicalNote, db: Session = Depends(get_session),
                         authorization: Optional[str] = Header(None)):
    """Steps 22-26. Note enters from the clinician side only.

    The raw note text is stored ONLY in this component's detail blob and is never
    returned to the patient app — see the egress endpoints, which expose the
    score but never the text.
    """
    _auth(authorization)
    subject_id = req.subject_id or _resolve(db, "mrn_hash", identity.hash_mrn(req.mrn or ""))
    _require_subject(db, subject_id)

    result = mc.call_c3(req.note_text, req.note_type,
                        req.anxiety_support, req.control_support,
                        subject_external_id=_external_id(db, subject_id, "c3_clinical_nlp"))
    row = _store(db, subject_id, "c3_clinical_nlp", result)
    db.commit()
    fusion_info = _auto_fuse(db, subject_id, "note-ingest")
    return {"subject_id": subject_id, "reading_id": row.id,
            "status": result.status, "score": result.raw_score, "note": result.note,
            **fusion_info}


# ═════════════════════════════════════════════════════════════════════════════
# FUSION — steps 27-31
# ═════════════════════════════════════════════════════════════════════════════
def _latest_readings(db: Session, subject_id: str) -> Dict[str, dict]:
    """Step 27. Latest reading per modality, for THIS subject only.

    The subject_id filter is the patient-separation guarantee. Every query in
    this file is scoped by it.
    """
    out: Dict[str, dict] = {}
    for modality in ALL_MODALITIES:
        row = db.scalar(
            select(ModalityReading)
            .where(ModalityReading.subject_id == subject_id,
                   ModalityReading.modality == modality)
            .order_by(ModalityReading.captured_at.desc(), ModalityReading.id.desc())
            .limit(1))
        if row:
            out[modality] = {
                "raw_score": row.raw_score, "status": row.status,
                "confidence": row.confidence, "coverage": row.coverage,
                "captured_at": row.captured_at, "model_version": row.model_version,
                "reading_id": row.id,
            }
    return out


def _calibration_pairs(db: Session):
    """(composite_at_verdict_time, clinician_tier) across ALL subjects.

    Global, not per-patient: the band edges are population-level constructs and
    a per-patient calibration would never accumulate enough labels to certify
    anything. The composite is taken from the fusion row the clinician actually
    judged, not the latest one — judging one row and calibrating on another
    would silently misalign score and label."""
    rows = db.execute(
        select(FusionResult.composite, Verdict.tier_label)
        .join(Verdict, Verdict.fusion_result_id == FusionResult.id)
        .where(FusionResult.composite.is_not(None))).all()
    return [(c, t) for c, t in rows]


def run_fusion(db: Session, subject_id: str, trigger: str = "manual") -> FusionResult:
    """Steps 27-31. Gate, fuse, persist. Always re-derived server-side."""
    _require_subject(db, subject_id)
    readings = _latest_readings(db, subject_id)

    decision = gate.evaluate(readings)

    if not decision.passed:
        row = FusionResult(
            subject_id=subject_id, composite=None, tier=None, band="GREY",
            confidence=0.0, modalities_used=len(decision.usable), renormalised=True,
            weights={}, contributions={},
            harmonisation={"gate": decision.summary()},
            reason=decision.reason, trigger=trigger, model_version="gate-blocked")
        db.add(row)
        _audit(db, subject_id, "fusion.blocked", decision.summary())
        db.commit()
        return row

    result = fusion_client.fuse(subject_id, decision.usable)
    harmonisation = result.get("harmonisation", {})
    harmonisation["gate"] = decision.summary()
    conf = conformal.predict_set(result.get("composite_score"), _calibration_pairs(db))
    harmonisation["conformal"] = conf.to_wire()

    row = FusionResult(
        subject_id=subject_id,
        composite=result.get("composite_score"),
        tier=result.get("tier"),
        band=result.get("band"),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        modalities_used=int(result.get("modalities_available", 0) or 0),
        renormalised=bool(result.get("renormalised", False)),
        weights=result.get("weights"), contributions=result.get("contributions"),
        harmonisation=harmonisation, reason=result.get("reason"),
        trigger=trigger, model_version=result.get("model_version"))
    db.add(row)
    _audit(db, subject_id, "fusion.computed",
           {"composite": row.composite, "tier": row.tier, "trigger": trigger,
            "modalities": sorted(decision.usable)})
    db.commit()
    return row


class FuseRequest(BaseModel):
    subject_id: Optional[str] = None
    mrn: Optional[str] = None
    trigger: str = "manual"


@app.post("/v1/fusion/run", tags=["fusion"])
def fusion_run(req: FuseRequest, db: Session = Depends(get_session),
               authorization: Optional[str] = Header(None)):
    _auth(authorization)
    subject_id = req.subject_id or _resolve(db, "mrn_hash", identity.hash_mrn(req.mrn or ""))
    row = run_fusion(db, subject_id, req.trigger)
    return {
        "subject_id": subject_id, "composite": row.composite, "tier": row.tier,
        "band": row.band, "confidence": round(row.confidence, 4),
        "modalities_used": row.modalities_used, "weights": row.weights,
        "contributions": row.contributions, "reason": row.reason,
        "gate": (row.harmonisation or {}).get("gate"),
        **((row.harmonisation or {}).get("conformal") or {}),
        "computed_at": row.computed_at,
    }


class VerdictRequest(BaseModel):
    fusion_result_id: int
    tier_label: str = Field(..., description="Low | Medium | High — the clinician's judgement")
    author: Optional[str] = None
    note: Optional[str] = None


@app.post("/v1/verdict", tags=["fusion"])
def record_verdict(req: VerdictRequest, db: Session = Depends(get_session),
                   authorization: Optional[str] = Header(None)):
    """The clinician's HITL tier judgement — the label source for conformal
    calibration and the safety record of every model/clinician disagreement.

    Assign the verdict BEFORE looking at the conformal set (the UI must order
    the controls that way), or the label is contaminated by the prediction it
    exists to calibrate."""
    _auth(authorization)
    if req.tier_label not in conformal.TIERS:
        raise HTTPException(422, f"tier_label must be one of {conformal.TIERS}")
    fr = db.get(FusionResult, req.fusion_result_id)
    if not fr:
        raise HTTPException(404, f"no fusion result {req.fusion_result_id}")

    v = Verdict(subject_id=fr.subject_id, fusion_result_id=fr.id,
                tier_label=req.tier_label,
                agrees_with_model=(fr.tier == req.tier_label) if fr.tier else None,
                author=req.author, note=req.note)
    db.add(v)
    _audit(db, fr.subject_id, "verdict.recorded",
           {"fusion_result_id": fr.id, "clinician_tier": req.tier_label,
            "model_tier": fr.tier, "agrees": v.agrees_with_model}, req.author)
    db.commit()

    n = len(_calibration_pairs(db))
    return {"verdict_id": v.id, "subject_id": fr.subject_id,
            "agrees_with_model": v.agrees_with_model,
            "calibration_labels_total": n,
            "conformal_calibrated": n >= conformal.MIN_CALIBRATION_N}


# ═════════════════════════════════════════════════════════════════════════════
# EGRESS — steps 32-35: different views, same source of truth
# ═════════════════════════════════════════════════════════════════════════════
def _latest_fusion(db: Session, subject_id: str) -> Optional[FusionResult]:
    return db.scalar(select(FusionResult)
                     .where(FusionResult.subject_id == subject_id)
                     .order_by(FusionResult.computed_at.desc(), FusionResult.id.desc())
                     .limit(1))


@app.get("/v1/patients/{subject_id}/risk", tags=["egress"])
def patient_risk(subject_id: str, db: Session = Depends(get_session)):
    """Steps 32-33. PATIENT view: composite, band, updated_at. Nothing else.

    Deliberately withholds per-modality scores, weights and any clinical note
    content. A patient seeing "your clinical notes score is 0.81" without a
    clinician present is a harm, not transparency.
    """
    _require_subject(db, subject_id)
    row = _latest_fusion(db, subject_id)
    if not row:
        return {"subject_id": subject_id, "composite": None, "band": "GREY",
                "message": "no assessment yet", "updated_at": None}
    _audit(db, subject_id, "egress.patient", None)
    db.commit()
    return {"subject_id": subject_id,
            "composite": row.composite, "band": row.band,
            "message": row.reason or "assessment available",
            "updated_at": row.computed_at}


@app.get("/v1/doctor/patients/{subject_id}/timeline", tags=["egress"])
def doctor_timeline(subject_id: str, limit: int = 20,
                    db: Session = Depends(get_session),
                    authorization: Optional[str] = Header(None)):
    """Steps 34-35. CLINICIAN view: composite + per-modality scores + freshness
    + status flags + the gate decision + trend history."""
    _auth(authorization)
    _require_subject(db, subject_id)

    latest = _latest_fusion(db, subject_id)
    readings = _latest_readings(db, subject_id)
    now = dt.datetime.now(dt.timezone.utc)

    modality_view = {}
    for modality in ALL_MODALITIES:
        r = readings.get(modality)
        if not r:
            modality_view[modality] = {"status": "absent", "score": None}
            continue
        captured = r["captured_at"]
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=dt.timezone.utc)
        age_min = (now - captured).total_seconds() / 60.0
        max_age = gate.MAX_AGE_MINUTES.get(modality)
        modality_view[modality] = {
            "status": r["status"],
            "score": r["raw_score"],
            "confidence": round(r["confidence"], 3),
            "coverage": round(r["coverage"], 3),
            "captured_at": captured,
            "age_minutes": round(age_min, 1),
            "fresh": (max_age is None) or (age_min <= max_age),
            "model_version": r["model_version"],
            "excluded": modality in gate.EXCLUDED_MODALITIES,
        }

    history = db.scalars(select(FusionResult)
                         .where(FusionResult.subject_id == subject_id)
                         .order_by(FusionResult.computed_at.desc())
                         .limit(limit)).all()

    _audit(db, subject_id, "egress.clinician", None)
    db.commit()

    return {
        "subject_id": subject_id,
        "composite": latest.composite if latest else None,
        "tier": latest.tier if latest else None,
        "band": latest.band if latest else "GREY",
        "confidence": round(latest.confidence, 4) if latest else 0.0,
        "reason": latest.reason if latest else "no assessment yet",
        "weights": latest.weights if latest else {},
        "contributions": latest.contributions if latest else {},
        "gate": (latest.harmonisation or {}).get("gate") if latest else None,
        "conformal": (latest.harmonisation or {}).get("conformal") if latest else None,
        "harmonisation": {k: v for k, v in (latest.harmonisation or {}).items()
                          if k != "gate"} if latest else {},
        "modalities": modality_view,
        "updated_at": latest.computed_at if latest else None,
        "trend": [{"composite": h.composite, "tier": h.tier, "band": h.band,
                   "computed_at": h.computed_at, "trigger": h.trigger}
                  for h in reversed(history)],
    }


class EvidenceRequest(BaseModel):
    question: str = Field(..., min_length=1, description="the clinician's question, "
                          "e.g. 'What does stepped care recommend at this severity?'")


@app.post("/v1/doctor/patients/{subject_id}/evidence", tags=["egress"])
def doctor_evidence(subject_id: str, req: EvidenceRequest,
                    db: Session = Depends(get_session),
                    authorization: Optional[str] = Header(None)):
    """Clinician decision support via CARE-AnxRAG, a separate HTTP service —
    NOT imported into this process (see rag_client.py for why). subject_id is
    used for auth/audit only; per the current integration contract, no patient
    data is forwarded into the RAG call. CARE-AnxRAG does its own retrieval,
    evidence scoring, and abstention — this endpoint's job is to call it
    honestly and never fabricate an answer if it can't be reached.
    """
    _auth(authorization)
    _require_subject(db, subject_id)

    result = rag_client.call_rag(req.question)

    # Audit the query, not its content — a clinician's free-text question can
    # contain patient-specific detail even though it isn't clinical note text,
    # so only its metadata is logged, same principle as raw notes never
    # leaving the clinician side.
    _audit(db, subject_id, "rag.evidence",
           {"available": result.available, "abstained": result.abstained,
            "safety_level": result.safety_level,
            "local_crisis_bypass": result.local_crisis_bypass,
            "error": result.error})
    db.commit()

    return {"subject_id": subject_id, **result.to_wire()}


@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "version": app.version,
        "fusion_mode": fusion_client.FUSION_MODE,
        "components_configured": {m: fn() for m, fn in mc.CONFIGURED.items()},
        "rag": rag_client.check_rag_health(),
        "mrn_pepper_set": bool(identity.MRN_PEPPER),
        "gate": {"min_usable_modalities": gate.MIN_USABLE_MODALITIES,
                 "excluded": sorted(gate.EXCLUDED_MODALITIES),
                 "max_age_minutes": gate.MAX_AGE_MINUTES},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
