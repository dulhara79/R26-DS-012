"""Backend-owned minimal attention notifications (handbook section 12).

FCM device tokens are encrypted at rest; this module never sends clinical
content in notifications. Delivery is queued in the same transaction as the
authoritative attention event and handled by a separate retryable worker.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import uuid
from typing import Literal, Optional

import jwt
from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from clinician_api import require_clinician
from db_models import (
    AuditLog, ClinicianSubjectAssignment, DeviceToken,
    NotificationDelivery, get_session, utcnow,
)
from patient_auth import require_patient

router = APIRouter(tags=["device-notifications"])


def _fernet() -> Fernet:
    key = os.getenv("DEVICE_TOKEN_ENCRYPTION_KEY", "")
    if not key:
        raise HTTPException(503, "device-token encryption not configured")
    try:
        return Fernet(key.encode())
    except (TypeError, ValueError) as exc:
        raise HTTPException(503, "invalid device-token encryption configuration") from exc


def _owner(authorization: Optional[str], db: Session):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    # Inspect untrusted audience solely to choose the real JWT validator.
    # No access is granted based on this decode.
    try:
        claims = jwt.decode(authorization[7:], options={"verify_signature": False})
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "invalid bearer token") from exc
    if claims.get("aud") == os.getenv("CLINICIAN_JWT_AUDIENCE", "clinanx"):
        clinician = require_clinician(authorization=authorization, db=db)
        return "clinician", clinician.clinician_id
    if claims.get("aud") == os.getenv("PATIENT_JWT_AUDIENCE", "aura"):
        patient = require_patient(authorization=authorization, db=db)
        return "patient", patient.subject_id
    raise HTTPException(401, "invalid bearer token audience")


class DeviceTokenRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=20, max_length=4096)
    platform: Literal["android", "ios"]


def _summary(device: DeviceToken) -> dict:
    return {"device_id": device.id, "platform": device.platform,
            "active": device.active, "registered_at": device.registered_at}


@router.post("/v1/device-tokens", status_code=200)
def register_device(body: DeviceTokenRegistration, authorization: Optional[str] = Header(None),
                    db: Session = Depends(get_session)):
    kind, owner_id = _owner(authorization, db)
    encrypted = _fernet().encrypt(body.token.encode()).decode()
    digest = hashlib.sha256(body.token.encode()).hexdigest()
    device = db.scalar(select(DeviceToken).where(DeviceToken.token_hash == digest))
    if device is None:
        device = DeviceToken(id=f"dev_{uuid.uuid4().hex}", token_hash=digest,
                             token_ciphertext=encrypted, owner_kind=kind,
                             owner_id=owner_id, platform=body.platform)
        db.add(device)
    else:
        # Device logout/account switch must revoke the old owner's access.
        device.owner_kind, device.owner_id = kind, owner_id
        device.platform, device.token_ciphertext = body.platform, encrypted
        device.active, device.registered_at = True, utcnow()
    db.add(AuditLog(subject_id=owner_id if kind == "patient" else None,
                    event="device.registered", actor=f"{kind}:{owner_id}",
                    detail={"device_id": device.id, "platform": device.platform}))
    db.commit()
    return _summary(device)


@router.get("/v1/device-tokens")
def list_devices(authorization: Optional[str] = Header(None),
                 db: Session = Depends(get_session)):
    kind, owner_id = _owner(authorization, db)
    devices = db.scalars(select(DeviceToken).where(
        DeviceToken.owner_kind == kind, DeviceToken.owner_id == owner_id,
        DeviceToken.active.is_(True))).all()
    return {"devices": [_summary(d) for d in devices]}


@router.delete("/v1/device-tokens/{device_id}", status_code=204)
def revoke_device(device_id: str, authorization: Optional[str] = Header(None),
                  db: Session = Depends(get_session)):
    kind, owner_id = _owner(authorization, db)
    device = db.get(DeviceToken, device_id)
    if device is None or device.owner_kind != kind or device.owner_id != owner_id:
        raise HTTPException(404, "device token not found")
    device.active = False
    db.add(AuditLog(subject_id=owner_id if kind == "patient" else None,
                    event="device.revoked", actor=f"{kind}:{owner_id}",
                    detail={"device_id": device.id}))
    db.commit()


def queue_event_delivery(db: Session, event) -> int:
    """Queue delivery for currently assigned clinicians and registered patient.

    Caller commits with the AttentionEvent; there is no notification if the
    transaction rolls back. Duplicates are prevented per event/device by DB.
    """
    clinicians = set(db.scalars(select(ClinicianSubjectAssignment.clinician_id).where(
        ClinicianSubjectAssignment.subject_id == event.subject_id,
        ClinicianSubjectAssignment.active.is_(True))).all())
    devices = db.scalars(select(DeviceToken).where(DeviceToken.active.is_(True))).all()
    selected = [d for d in devices if (d.owner_kind == "patient" and
                d.owner_id == event.subject_id) or
                (d.owner_kind == "clinician" and d.owner_id in clinicians)]
    for device in selected:
        db.add(NotificationDelivery(id=f"delivery_{uuid.uuid4().hex}",
                  event_id=event.id, device_id=device.id))
    db.add(AuditLog(subject_id=event.subject_id, event="notification.queued",
                    actor="backend", detail={"event_id": event.id,
                                             "recipient_devices": len(selected)}))
    return len(selected)
