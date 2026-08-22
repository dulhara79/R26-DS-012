"""
Database schema — Central Backend, Component 4 · R26-DS-012

Implements the persistence layer of the integration sequence diagram.

Design rules baked into this schema, each for a reason you can defend:

1. THE RAW MRN IS NEVER STORED. Only sha256(MRN + pepper) lands in the database,
   in `subject_aliases`. If the database is ever exfiltrated, the attacker cannot
   recover patient identifiers without the pepper, which lives outside the DB.

2. ONE subject_id, MANY aliases. The clinician knows the patient by MRN hash; the
   patient app knows them by app_user_id. Both resolve to the same subject_id.
   This is what makes the pairing flow work and what keeps every modality writing
   against one key.

3. READINGS ARE APPEND-ONLY. A modality reading is never updated or deleted, so
   the clinical record is auditable. "Latest" is a query, not a mutation.

4. STATUS IS EXPLICIT. A reading carries `ok` / `not_validated` / `error`, so a
   component that failed its permutation null (C2) is stored and visible but
   excluded from the composite by a stated rule rather than by silent omission.

SQLite for development, PostgreSQL for deployment — same ORM either way. Set
DATABASE_URL to switch.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Optional

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Index, Integer,
                        String, UniqueConstraint, create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./central_backend.db")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


# ── identity ─────────────────────────────────────────────────────────────────
class Subject(Base):
    """One enrolled patient. Carries no identifying information whatsoever."""
    __tablename__ = "subjects"

    subject_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    enrolled_by: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | withdrawn

    aliases: Mapped[list["SubjectAlias"]] = relationship(back_populates="subject")


class SubjectAlias(Base):
    """How each system refers to this subject.

    alias_type = 'mrn_hash'    -> sha256(MRN + pepper), written at enrolment
    alias_type = 'app_user_id' -> the patient app's own id, written at pairing
    """
    __tablename__ = "subject_aliases"
    __table_args__ = (
        UniqueConstraint("alias_type", "alias_value", name="uq_alias"),
        Index("ix_alias_lookup", "alias_type", "alias_value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), index=True)
    alias_type: Mapped[str] = mapped_column(String(24))
    alias_value: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    subject: Mapped[Subject] = relationship(back_populates="aliases")


class PairingCode(Base):
    """Short-lived code the clinician reads aloud to the patient.

    Single-use and time-limited: a code that stayed valid forever would let anyone
    who overheard it attach their phone to someone else's clinical record.
    """
    __tablename__ = "pairing_codes"

    code: Mapped[str] = mapped_column(String(16), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))


# ── modality readings ────────────────────────────────────────────────────────
class ModalityReading(Base):
    """One score from one component at one time. Append-only."""
    __tablename__ = "modality_readings"
    __table_args__ = (
        Index("ix_reading_lookup", "subject_id", "modality", "captured_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), index=True)
    modality: Mapped[str] = mapped_column(String(32))          # c1_physiological, ...
    raw_score: Mapped[Optional[float]] = mapped_column(Float)  # component's own scale
    status: Mapped[str] = mapped_column(String(16), default="ok")   # ok|not_validated|error
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    coverage: Mapped[float] = mapped_column(Float, default=1.0)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_version: Mapped[Optional[str]] = mapped_column(String(64))
    detail: Mapped[Optional[dict]] = mapped_column(JSON)        # component's full response
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ── fusion output ────────────────────────────────────────────────────────────
class FusionResult(Base):
    """One composite. Also append-only — the history IS the trend the clinician sees."""
    __tablename__ = "fusion_results"
    __table_args__ = (Index("ix_fusion_lookup", "subject_id", "computed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), index=True)
    composite: Mapped[Optional[float]] = mapped_column(Float)
    tier: Mapped[Optional[str]] = mapped_column(String(16))     # Low|Medium|High|None
    band: Mapped[Optional[str]] = mapped_column(String(16))     # GREEN|AMBER|RED|GREY
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    modalities_used: Mapped[int] = mapped_column(Integer, default=0)
    renormalised: Mapped[bool] = mapped_column(default=False)
    weights: Mapped[Optional[dict]] = mapped_column(JSON)
    contributions: Mapped[Optional[dict]] = mapped_column(JSON)
    harmonisation: Mapped[Optional[dict]] = mapped_column(JSON)
    reason: Mapped[Optional[str]] = mapped_column(String(255))  # why no tier, if none
    trigger: Mapped[Optional[str]] = mapped_column(String(32))  # which event caused this
    model_version: Mapped[Optional[str]] = mapped_column(String(32))
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Verdict(Base):
    """The clinician's HITL tier judgement for one fusion result.

    Serves two purposes at once: (1) the label source for conformal calibration
    and any future learned fusion weights, and (2) the safety record — every
    time a clinician disagrees with the composite, that disagreement is stored.
    Assign the verdict BEFORE looking at the conformal set, or the label is
    contaminated by the prediction it is meant to calibrate.
    """
    __tablename__ = "verdicts"
    __table_args__ = (Index("ix_verdict_lookup", "subject_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"), index=True)
    fusion_result_id: Mapped[int] = mapped_column(ForeignKey("fusion_results.id"), index=True)
    tier_label: Mapped[str] = mapped_column(String(16))     # Low | Medium | High
    agrees_with_model: Mapped[Optional[bool]] = mapped_column()
    author: Mapped[Optional[str]] = mapped_column(String(64))
    note: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    """Every enrolment, pairing, ingestion, fusion and egress. Append-only.

    An ethics committee will ask who saw what and when. This is the answer.
    """
    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_lookup", "subject_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    event: Mapped[str] = mapped_column(String(48))
    actor: Mapped[Optional[str]] = mapped_column(String(64))
    detail: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ── engine / session ─────────────────────────────────────────────────────────
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
