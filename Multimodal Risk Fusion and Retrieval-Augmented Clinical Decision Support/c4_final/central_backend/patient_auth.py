from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
import secrets
import uuid
from typing import Optional

import jwt
from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db_models import Subject, get_session, utcnow


class PatientPrincipal(BaseModel):
    subject_id: str
    role: str
    token_id: str
    expires_at: dt.datetime


def hash_installation_secret(
    installation_secret: str, salt: Optional[bytes] = None
) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", installation_secret.encode(), salt, 210_000
    )
    return "pbkdf2_sha256$210000$%s$%s" % (
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_installation_secret(installation_secret: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            installation_secret.encode(),
            base64.b64decode(salt),
            int(rounds),
        )
        return hmac.compare_digest(actual, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


def _jwt_settings() -> tuple[str, str, str]:
    secret = os.getenv("PATIENT_JWT_SECRET")
    if not secret:
        raise HTTPException(503, "patient authentication is not configured")
    return (
        secret,
        os.getenv("PATIENT_JWT_ISSUER", "r26-central-backend"),
        os.getenv("PATIENT_JWT_AUDIENCE", "aura"),
    )


def issue_patient_token(subject_id: str) -> tuple[str, dt.datetime]:
    secret, issuer, audience = _jwt_settings()
    now = utcnow()
    expires_at = now + dt.timedelta(
        seconds=int(os.getenv("PATIENT_JWT_TTL_SECONDS", "28800"))
    )
    claims = {
        "sub": subject_id,
        "subject_id": subject_id,
        "role": "patient",
        "iss": issuer,
        "aud": audience,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, secret, algorithm="HS256"), expires_at


def require_patient(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_session),
) -> PatientPrincipal:
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
            options={
                "require": [
                    "sub",
                    "subject_id",
                    "role",
                    "iat",
                    "exp",
                    "jti",
                ]
            },
        )
        for name in ("sub", "subject_id", "role", "jti"):
            if not isinstance(claims[name], str) or not claims[name].strip():
                raise jwt.InvalidTokenError(f"{name} must be a non-empty string")
        for name in ("iat", "exp"):
            if isinstance(claims[name], bool) or not isinstance(
                claims[name], (int, float)
            ):
                raise jwt.InvalidTokenError(f"{name} must be a numeric date")
        if claims["role"] != "patient" or claims["sub"] != claims["subject_id"]:
            raise jwt.InvalidTokenError("invalid patient principal claims")
        expires_at = dt.datetime.fromtimestamp(claims["exp"], dt.timezone.utc)
    except (jwt.PyJWTError, OSError, OverflowError, TypeError, ValueError):
        raise HTTPException(401, "invalid or expired bearer token")

    subject = db.get(Subject, claims["subject_id"])
    if not subject or subject.status != "active":
        raise HTTPException(401, "invalid patient principal")

    return PatientPrincipal(
        subject_id=subject.subject_id,
        role="patient",
        token_id=claims["jti"],
        expires_at=expires_at,
    )


def patient_or_service(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_session),
) -> Optional[PatientPrincipal]:
    """Accept a subject-bound patient JWT or the configured internal token."""
    service_token = os.getenv("BACKEND_API_TOKEN", "")
    if service_token and authorization == f"Bearer {service_token}":
        return None
    return require_patient(authorization, db)
