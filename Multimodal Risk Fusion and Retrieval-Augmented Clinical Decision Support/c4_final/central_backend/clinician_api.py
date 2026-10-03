from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
import secrets
import uuid
from typing import Literal, Optional

import gate
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db_models import (AttentionEvent, AuditLog, Clinician, ClinicianAssignmentInvite, ClinicianSubjectAssignment,
                       ForecastResult, FusionResult, ModalityReading, Subject,
                       get_session, utcnow)
from patient_auth import PatientPrincipal, require_patient

router = APIRouter()
ALL_MODALITIES = ["c1_physiological", "c2_behavioral", "c3_clinical_nlp", "c4_demographic"]
MAX_AGE_MINUTES = gate.MAX_AGE_MINUTES


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210_000)
    return "pbkdf2_sha256$210000$%s$%s" % (
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256": return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(rounds))
        return hmac.compare_digest(actual, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


def _jwt_settings():
    secret = os.getenv("CLINICIAN_JWT_SECRET")
    if not secret: raise HTTPException(503, "clinician authentication is not configured")
    return secret, os.getenv("CLINICIAN_JWT_ISSUER", "r26-central-backend"), os.getenv("CLINICIAN_JWT_AUDIENCE", "clinanx")


class Principal(BaseModel):
    clinician_id: str
    display_name: str
    role: str
    token_id: str
    expires_at: dt.datetime


def require_clinician(authorization: Optional[str] = Header(None), db: Session = Depends(get_session)) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    secret, issuer, audience = _jwt_settings()
    try:
        claims = jwt.decode(
            authorization[7:],
            secret,
            algorithms=["HS256"],
            issuer=issuer,
            audience=audience,
            options={"require": ["sub", "clinician_id", "iat", "exp", "jti"]},
        )
        for name in ("sub", "clinician_id", "jti"):
            if not isinstance(claims[name], str) or not claims[name].strip():
                raise jwt.InvalidTokenError(f"{name} must be a non-empty string")
        for name in ("iat", "exp"):
            if isinstance(claims[name], bool) or not isinstance(
                claims[name], (int, float)
            ):
                raise jwt.InvalidTokenError(f"{name} must be a numeric date")
        expires_at = dt.datetime.fromtimestamp(claims["exp"], dt.timezone.utc)
    except (jwt.PyJWTError, OSError, OverflowError, TypeError, ValueError):
        raise HTTPException(401, "invalid or expired bearer token")
    clinician = db.get(Clinician, claims["clinician_id"])
    if not clinician or not clinician.active or claims.get("sub") != clinician.clinician_id:
        raise HTTPException(401, "invalid clinician principal")
    if clinician.role.lower() not in {"clinician", "doctor"}:
        raise HTTPException(403, "principal does not have a clinician role")
    return Principal(clinician_id=clinician.clinician_id, display_name=clinician.display_name,
                     role=clinician.role, token_id=claims["jti"],
                     expires_at=expires_at)


def service_or_clinician(authorization: Optional[str] = Header(None), db: Session = Depends(get_session)) -> Optional[Principal]:
    service_token = os.getenv("BACKEND_API_TOKEN", "")
    if service_token and authorization and hmac.compare_digest(authorization, f"Bearer {service_token}"):
        return None
    return require_clinician(authorization, db)


def require_assignment(db: Session, principal: Principal, subject_id: str) -> None:
    assigned = db.scalar(select(ClinicianSubjectAssignment.id).where(
        ClinicianSubjectAssignment.clinician_id == principal.clinician_id,
        ClinicianSubjectAssignment.subject_id == subject_id,
        ClinicianSubjectAssignment.active.is_(True)))
    if not assigned:
        db.add(AuditLog(subject_id=subject_id, event="access.denied",
                        actor=principal.clinician_id, detail={"reason": "not_assigned"}))
        db.commit()
        raise HTTPException(403, "patient is not assigned to this clinician")


def audit_clinician(db: Session, subject_id: Optional[str], event: str, principal: Principal,
                    detail: Optional[dict] = None) -> None:
    db.add(AuditLog(subject_id=subject_id, event=event,
                    actor=principal.clinician_id, detail=detail))
    db.commit()


class LoginRequest(BaseModel):
    clinician_id: str
    password: str

class ClinicianIdentity(BaseModel):
    clinician_id: str; display_name: str; role: Optional[str] = None
class LoginResponse(BaseModel):
    clinician: ClinicianIdentity; access_token: str; token_type: str; expires_at: dt.datetime
class CurrentAssessment(BaseModel):
    score: Optional[float]; tier: Optional[str]; band: Optional[str]
class ForecastWire(BaseModel):
    forecast_result_id: str; scope: str; horizon_minutes: int; score: Optional[float]
    tier: Optional[str]; escalation_probability: Optional[float]; escalation_predicted: bool
    generated_at: dt.datetime; valid_until: dt.datetime
class ModalityWire(BaseModel):
    component_id: str; score: Optional[float]; available: bool; included_in_fusion: bool
    status: str; confidence: Optional[float]; coverage: Optional[float]
    captured_at: Optional[dt.datetime]; contribution: Optional[float]
    freshness_age_minutes: Optional[float] = None
    max_age_minutes: Optional[int] = None
    exclusion_reason: Optional[str] = None
class AssessmentWire(BaseModel):
    subject_id: str; fusion_result_id: Optional[int]; current_assessment: Optional[CurrentAssessment]
    forecast: Optional[ForecastWire]; confidence: Optional[float]; assessment_status: str
    modalities: list[ModalityWire]; computed_at: Optional[dt.datetime]; model_version: Optional[str]
class EventWire(BaseModel):
    id: str; subject_id: str; fusion_result_id: Optional[int]; forecast_result_id: Optional[str]
    event_type: str; severity: str; reason: str; forecast_horizon: int; status: str
    created_at: dt.datetime; acknowledged_at: Optional[dt.datetime]; acknowledged_by: Optional[str]
    resolved_at: Optional[dt.datetime]; resolved_by: Optional[str]
    resolution_note: Optional[str]; policy_version: str
class EventResponse(BaseModel): event: EventWire
class EventsResponse(BaseModel): events: list[EventWire]


@router.post("/auth/login", tags=["clinician-auth"], response_model=LoginResponse)
def login(req: LoginRequest, db: Session = Depends(get_session)):
    clinician = db.get(Clinician, req.clinician_id)
    if not clinician or not clinician.active or not verify_password(req.password, clinician.password_hash):
        raise HTTPException(401, "invalid credentials")
    if clinician.role.lower() not in {"clinician", "doctor"}:
        raise HTTPException(403, "account does not have a clinician role")
    secret, issuer, audience = _jwt_settings()
    now = utcnow(); expiry = now + dt.timedelta(seconds=int(os.getenv("CLINICIAN_JWT_TTL_SECONDS", "28800")))
    claims = {"sub": clinician.clinician_id, "clinician_id": clinician.clinician_id,
              "role": clinician.role, "iss": issuer, "aud": audience,
              "iat": int(now.timestamp()), "exp": int(expiry.timestamp()), "jti": uuid.uuid4().hex}
    return {"clinician": {"clinician_id": clinician.clinician_id,
                           "display_name": clinician.display_name, "role": clinician.role},
            "access_token": jwt.encode(claims, secret, algorithm="HS256"),
            "token_type": "bearer", "expires_at": expiry}


@router.get("/v1/me", tags=["clinician-auth"], response_model=Principal)
def me(principal: Principal = Depends(require_clinician)):
    return principal.model_dump()


def _assessment_status(row: Optional[FusionResult]) -> str:
    if not row: return "unavailable"
    status = ((row.harmonisation or {}).get("assessment") or {}).get("status")
    return {"complete": "complete", "provisional": "partial", "insufficient": "unavailable"}.get(status, "partial" if row.composite is not None else "unavailable")


def _latest_forecast(db: Session, subject_id: str, as_of: Optional[dt.datetime] = None):
    now = as_of or utcnow()
    return db.scalar(select(ForecastResult).where(ForecastResult.subject_id == subject_id,
        ForecastResult.scope == "physiological", ForecastResult.generated_at <= now,
        ForecastResult.valid_until >= now)
        .order_by(ForecastResult.generated_at.desc()).limit(1))


def _forecast_for_assessment(db: Session, row: FusionResult):
    stmt = select(ForecastResult).where(
        ForecastResult.subject_id == row.subject_id,
        ForecastResult.source_fusion_result_id == row.id)
    snapshot = (row.harmonisation or {}).get("source_reading_ids")
    if snapshot is not None:
        reading_id = snapshot.get("c1_physiological")
        if reading_id is None:
            return None
        stmt = stmt.where(ForecastResult.source_reading_id == reading_id)
    # Old rows cannot recover their source reading; pin the earliest linked
    # forecast rather than replacing the displayed history on every poll.
    return db.scalar(stmt.order_by(ForecastResult.generated_at.asc(),
                                   ForecastResult.forecast_result_id.asc()).limit(1))


def _forecast_wire(f):
    if not f: return None
    return {"forecast_result_id": f.forecast_result_id, "scope": f.scope,
            "horizon_minutes": f.horizon_minutes, "score": f.score, "tier": f.tier,
            "escalation_probability": f.escalation_probability,
            "escalation_predicted": f.escalation_predicted, "generated_at": f.generated_at,
            "valid_until": f.valid_until}


def _assessment_wire(db: Session, subject_id: str, row: Optional[FusionResult],
                     as_of: Optional[dt.datetime] = None,
                     historical: bool = False):
    as_of = as_of or utcnow()
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=dt.timezone.utc)
    snapshot = (row.harmonisation or {}).get("source_reading_ids") if row else None
    if snapshot is not None:
        readings = db.scalars(select(ModalityReading).where(
            ModalityReading.subject_id == subject_id,
            ModalityReading.id.in_(list(snapshot.values())))).all() if snapshot else []
        latest = {r.modality: r for r in readings
                  if snapshot.get(r.modality) == r.id}
    else:
        # Historical pre-snapshot rows retain the legacy best-effort view.
        readings = db.scalars(select(ModalityReading).where(
            ModalityReading.subject_id == subject_id, ModalityReading.captured_at <= as_of)
            .order_by(ModalityReading.captured_at.desc(), ModalityReading.id.desc())).all()
        latest = {}
        for r in readings: latest.setdefault(r.modality, r)
    now = as_of
    modalities = []
    for name in ALL_MODALITIES:
        r = latest.get(name)
        if not r:
            modalities.append({"component_id": name, "score": None, "available": False,
                "included_in_fusion": False, "status": "absent", "confidence": None,
                "coverage": None, "captured_at": None, "contribution": None,
                "freshness_age_minutes": None, "max_age_minutes": MAX_AGE_MINUTES[name],
                "exclusion_reason": "research-only" if name == "c2_behavioral" else "absent"})
            continue
        captured = r.captured_at.replace(tzinfo=dt.timezone.utc) if r.captured_at.tzinfo is None else r.captured_at
        max_age = MAX_AGE_MINUTES[name]
        age = max(0.0, (now-captured).total_seconds() / 60)
        fresh = max_age is None or age <= max_age
        included = bool(row and name != "c2_behavioral" and name in (row.weights or {})
                        and (snapshot is None or snapshot.get(name) == r.id))
        modalities.append({"component_id": name, "score": r.raw_score, "available": r.status == "ok" and fresh,
            "included_in_fusion": included, "status": r.status if fresh else "stale",
            "confidence": r.confidence, "coverage": r.coverage, "captured_at": r.captured_at,
            "contribution": (row.contributions or {}).get(name) if row else None,
            "freshness_age_minutes": round(age, 2), "max_age_minutes": max_age,
            "exclusion_reason": ("research-only" if name == "c2_behavioral" else
                                 "stale" if not fresh else
                                 r.status if r.status != "ok" else None)})
    return {"subject_id": subject_id, "fusion_result_id": row.id if row else None,
            "current_assessment": ({"score": row.composite, "tier": row.tier, "band": row.band} if row else None),
            "forecast": _forecast_wire(_forecast_for_assessment(db, row) if historical and row else
                                       _latest_forecast(db, subject_id, as_of)),
            "confidence": row.confidence if row else None, "assessment_status": _assessment_status(row),
            "modalities": modalities, "computed_at": row.computed_at if row else None,
            "model_version": row.model_version if row else None}


def _assigned_ids(db, principal):
    return list(db.scalars(select(ClinicianSubjectAssignment.subject_id).where(
        ClinicianSubjectAssignment.clinician_id == principal.clinician_id,
        ClinicianSubjectAssignment.active.is_(True))).all())


@router.get("/v1/patients/{subject_id}/assessment/latest", tags=["clinician"], response_model=AssessmentWire)
def latest_assessment(subject_id: str, db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    require_assignment(db, principal, subject_id)
    row = db.scalar(select(FusionResult).where(FusionResult.subject_id == subject_id).order_by(FusionResult.computed_at.desc(), FusionResult.id.desc()).limit(1))
    audit_clinician(db, subject_id, "assessment.read", principal,
                    {"fusion_result_id": row.id if row else None})
    return _assessment_wire(db, subject_id, row)


@router.get("/v1/patients/{subject_id}/assessments", tags=["clinician"])
def assessments(subject_id: str, db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    require_assignment(db, principal, subject_id)
    rows = db.scalars(select(FusionResult).where(FusionResult.subject_id == subject_id).order_by(FusionResult.computed_at.desc())).all()
    audit_clinician(db, subject_id, "assessment.history_read", principal,
                    {"count": len(rows)})
    events = db.scalars(select(AttentionEvent).where(AttentionEvent.subject_id == subject_id)
                        .order_by(AttentionEvent.created_at.desc())).all()
    return {"assessments": [_assessment_wire(db, subject_id, row, row.computed_at, historical=True) for row in rows],
            "events": [_event_wire(event) for event in events]}


@router.get("/v1/patients/{subject_id}/data-quality", tags=["clinician"])
def data_quality(subject_id: str, db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    require_assignment(db, principal, subject_id)
    audit_clinician(db, subject_id, "assessment.data_quality_read", principal)
    return {"subject_id": subject_id, "modalities": _assessment_wire(db, subject_id, None)["modalities"]}


def _patient_summary(db, sid):
    row = db.scalar(select(FusionResult).where(FusionResult.subject_id == sid).order_by(FusionResult.computed_at.desc(), FusionResult.id.desc()).limit(1))
    f = _latest_forecast(db, sid)
    count = len(db.scalars(select(AttentionEvent.id).where(AttentionEvent.subject_id == sid, AttentionEvent.status == "OPEN")).all())
    display_id = "Patient " + hashlib.sha256(f"display-v1:{sid}".encode()).hexdigest()[:8].upper()
    return {"subject_id": sid, "display_id": display_id, "fusion_result_id": row.id if row else None,
            "current": {"score": row.composite if row else None, "tier": row.tier if row else None},
            "forecast": ({"score": f.score, "tier": f.tier, "horizon_minutes": f.horizon_minutes,
                          "predicted": f.escalation_predicted, "escalation_predicted": f.escalation_predicted} if f else None),
            "assessment_status": _assessment_status(row), "last_updated": row.computed_at if row else None,
            "open_event_count": count}


@router.get("/v1/clinicians/me/patients", tags=["clinician"])
def roster(db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    audit_clinician(db, None, "roster.read", principal)
    return {"patients": [_patient_summary(db, sid) for sid in _assigned_ids(db, principal)]}


def _event_wire(e):
    wire = {key: getattr(e, key) for key in ("id", "subject_id", "fusion_result_id", "forecast_result_id",
        "event_type", "severity", "reason", "forecast_horizon", "status", "created_at",
        "acknowledged_at", "acknowledged_by", "resolved_at", "resolved_by",
        "resolution_note", "policy_version")}
    # SQLite drops timezone information on reload; PostgreSQL retains it.
    # Both reads and transitions must expose the same UTC event timestamps.
    for key in ("created_at", "acknowledged_at", "resolved_at"):
        value = wire[key]
        if value is not None:
            wire[key] = (value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None
                         else value.astimezone(dt.timezone.utc))
    return wire


@router.get("/v1/clinicians/me/dashboard", tags=["clinician"])
def dashboard(db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    ids = _assigned_ids(db, principal)
    audit_clinician(db, None, "dashboard.read", principal, {"assigned_count": len(ids)})
    events = db.scalars(select(AttentionEvent).where(AttentionEvent.subject_id.in_(ids), AttentionEvent.status == "OPEN").order_by(AttentionEvent.created_at.desc())).all() if ids else []
    return {"clinician": {"clinician_id": principal.clinician_id, "display_name": principal.display_name},
            "assigned_count": len(ids), "open_attention_events": [_event_wire(e) for e in events],
            "patients": [_patient_summary(db, sid) for sid in ids]}


@router.get("/v1/attention-events", tags=["attention-events"], response_model=EventsResponse)
def list_events(status: Optional[Literal["OPEN", "ACKNOWLEDGED", "RESOLVED"]] = Query(None), subject_id: Optional[str] = None,
                db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    ids = _assigned_ids(db, principal)
    if subject_id:
        require_assignment(db, principal, subject_id); ids = [subject_id]
    audit_clinician(db, subject_id, "attention.list", principal, {"status": status})
    if not ids: return {"events": []}
    stmt = select(AttentionEvent).where(AttentionEvent.subject_id.in_(ids))
    if status: stmt = stmt.where(AttentionEvent.status == status)
    return {"events": [_event_wire(e) for e in db.scalars(stmt.order_by(AttentionEvent.created_at.desc())).all()]}


def _assigned_event(db, principal, event_id):
    e = db.get(AttentionEvent, event_id)
    if not e: raise HTTPException(404, "attention event not found")
    require_assignment(db, principal, e.subject_id)
    return e


@router.get("/v1/attention-events/{event_id}", tags=["attention-events"], response_model=EventResponse)
def event_detail(event_id: str, db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    event = _assigned_event(db, principal, event_id)
    audit_clinician(db, event.subject_id, "attention.read", principal, {"event_id": event_id})
    return {"event": _event_wire(event)}


class EmptyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssignmentInviteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invite_code: str = Field(..., min_length=1, max_length=128)


@router.post("/v1/patients/me/assignment-invites", tags=["patient-auth"])
def create_assignment_invite(body: EmptyBody,
                             principal: PatientPrincipal = Depends(require_patient),
                             db: Session = Depends(get_session)):
    code = secrets.token_urlsafe(32)
    expires = utcnow() + dt.timedelta(minutes=10)
    db.add(ClinicianAssignmentInvite(code_hash=hashlib.sha256(code.encode()).hexdigest(),
                                      subject_id=principal.subject_id, expires_at=expires))
    db.add(AuditLog(subject_id=principal.subject_id, event="assignment.invite_created",
                    actor=f"patient:{principal.subject_id}", detail={"expires_at": expires.isoformat()}))
    db.commit()
    return {"invite_code": code, "expires_at": expires}


@router.post("/v1/clinicians/me/assignments", tags=["clinician"])
def accept_assignment_invite(body: AssignmentInviteRequest,
                             principal: Principal = Depends(require_clinician),
                             db: Session = Depends(get_session)):
    digest = hashlib.sha256(body.invite_code.encode()).hexdigest()
    invite = db.scalar(select(ClinicianAssignmentInvite).where(
        ClinicianAssignmentInvite.code_hash == digest).with_for_update())
    if invite is None:
        raise HTTPException(404, "assignment invite not found")
    if invite.redeemed_at is not None:
        raise HTTPException(409, "assignment invite already used")
    expires = invite.expires_at.replace(tzinfo=dt.timezone.utc) if invite.expires_at.tzinfo is None else invite.expires_at
    if expires <= utcnow():
        raise HTTPException(410, "assignment invite expired")
    subject = db.scalar(select(Subject).where(
        Subject.subject_id == invite.subject_id).with_for_update())
    if subject is None or subject.status != "active":
        raise HTTPException(409, "subject is not active")
    now = utcnow()
    updated = db.execute(update(ClinicianAssignmentInvite).where(
        ClinicianAssignmentInvite.code_hash == digest,
        ClinicianAssignmentInvite.redeemed_at.is_(None)).values(
            redeemed_at=now, redeemed_by=principal.clinician_id))
    if updated.rowcount != 1:
        db.rollback()
        raise HTTPException(409, "assignment invite already used")
    assignment = db.scalar(select(ClinicianSubjectAssignment).where(
        ClinicianSubjectAssignment.clinician_id == principal.clinician_id,
        ClinicianSubjectAssignment.subject_id == invite.subject_id))
    if assignment is None:
        try:
            with db.begin_nested():
                db.add(ClinicianSubjectAssignment(clinician_id=principal.clinician_id,
                                                  subject_id=invite.subject_id))
                db.flush()
        except IntegrityError:
            # A different valid invite for the same clinician/subject may have
            # won the unique-key race. Keep redemption atomic and reactivate it.
            assignment = db.scalar(select(ClinicianSubjectAssignment).where(
                ClinicianSubjectAssignment.clinician_id == principal.clinician_id,
                ClinicianSubjectAssignment.subject_id == invite.subject_id))
            if assignment is None:
                raise
    if assignment is not None:
        assignment.active = True
        assignment.assigned_at = now
        assignment.ended_at = None
    audit_clinician(db, invite.subject_id, "assignment.accepted", principal)
    return {"clinician_id": principal.clinician_id, "subject_id": invite.subject_id, "active": True}


class ResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: Optional[str] = Field(None, max_length=255)

    @field_validator("note", mode="before")
    @classmethod
    def normalize_note(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip() or None


@router.post("/v1/attention-events/{event_id}/acknowledge", tags=["attention-events"], response_model=EventResponse)
def acknowledge(event_id: str, body: EmptyBody, db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    event = _assigned_event(db, principal, event_id); now = utcnow()
    result = db.execute(update(AttentionEvent).where(AttentionEvent.id == event_id, AttentionEvent.status == "OPEN")
        .values(status="ACKNOWLEDGED", acknowledged_at=now, acknowledged_by=principal.clinician_id))
    if result.rowcount != 1: db.rollback(); raise HTTPException(409, "event state has changed")
    db.add(AuditLog(subject_id=event.subject_id, event="attention.acknowledged",
                    actor=principal.clinician_id, detail={"event_id": event_id}))
    db.commit(); return {"event": _event_wire(db.get(AttentionEvent, event_id))}


@router.post("/v1/attention-events/{event_id}/resolve", tags=["attention-events"], response_model=EventResponse)
def resolve(event_id: str, body: ResolveBody, db: Session = Depends(get_session), principal: Principal = Depends(require_clinician)):
    event = _assigned_event(db, principal, event_id); now = utcnow()
    result = db.execute(update(AttentionEvent).where(AttentionEvent.id == event_id, AttentionEvent.status == "ACKNOWLEDGED")
        .values(status="RESOLVED", resolved_at=now, resolved_by=principal.clinician_id,
                resolution_note=body.note))
    if result.rowcount != 1: db.rollback(); raise HTTPException(409, "event state has changed")
    db.add(AuditLog(subject_id=event.subject_id, event="attention.resolved",
                    actor=principal.clinician_id, detail={"event_id": event_id}))
    db.commit(); return {"event": _event_wire(db.get(AttentionEvent, event_id))}
