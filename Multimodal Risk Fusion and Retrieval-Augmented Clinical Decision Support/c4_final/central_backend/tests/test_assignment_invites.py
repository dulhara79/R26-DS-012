"""Patient consent must be required before a clinician joins an existing subject."""

import os

os.environ.setdefault("MRN_PEPPER", "assignment-invite-test-pepper")
os.environ.setdefault("CLINICIAN_JWT_SECRET", "assignment-invite-clinician-secret")
os.environ.setdefault("PATIENT_JWT_SECRET", "assignment-invite-patient-secret")

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from clinician_api import hash_password
from db_models import (
    AuditLog, AttentionEvent, Clinician, ClinicianAssignmentInvite,
    ClinicianSubjectAssignment, EscalationEpisode, ForecastResult, FusionResult,
    ModalityReading, PairingCode, PatientCredential, SessionLocal, Subject,
    SubjectAlias, Verdict, init_db,
)
from main import app
from patient_auth import issue_patient_token


def setup_function():
    init_db()
    with SessionLocal() as db:
        for model in (AuditLog, AttentionEvent, EscalationEpisode,
                      ForecastResult, Verdict, ClinicianSubjectAssignment,
                      ClinicianAssignmentInvite, FusionResult, ModalityReading,
                      PairingCode, PatientCredential, SubjectAlias, Clinician, Subject):
            db.execute(delete(model))
        db.add(Subject(subject_id="subject-a"))
        db.add(Subject(subject_id="subject-b"))
        db.add(Clinician(clinician_id="DR001", display_name="Dr X",
                         password_hash=hash_password("secret")))
        db.commit()


def auth(client):
    login = client.post("/auth/login", json={"clinician_id": "DR001", "password": "secret"})
    return {"Authorization": "Bearer " + login.json()["access_token"]}


def test_patient_issued_invite_assigns_only_consented_subject_once():
    client = TestClient(app)
    clinician = auth(client)
    patient_token, _ = issue_patient_token("subject-a")
    issued = client.post("/v1/patients/me/assignment-invites",
                         headers={"Authorization": "Bearer " + patient_token}, json={})
    assert issued.status_code == 200
    code = issued.json()["invite_code"]
    assert len(code) >= 32
    assert client.post("/v1/clinicians/me/assignments", headers=clinician,
                       json={"invite_code": "wrong"}).status_code == 404
    accepted = client.post("/v1/clinicians/me/assignments", headers=clinician,
                           json={"invite_code": code})
    assert accepted.status_code == 200
    assert accepted.json() == {"clinician_id": "DR001", "subject_id": "subject-a", "active": True}
    assert client.post("/v1/clinicians/me/assignments", headers=clinician,
                       json={"invite_code": code}).status_code == 409
    assert client.get("/v1/patients/subject-a/assessment/latest", headers=clinician).status_code == 200
    assert client.get("/v1/patients/subject-b/assessment/latest", headers=clinician).status_code == 403
    with SessionLocal() as db:
        assert db.scalar(select(AuditLog).where(AuditLog.event == "assignment.accepted")).actor == "DR001"


def test_anonymous_patient_cannot_create_invite_and_clinician_cannot_self_assign():
    client = TestClient(app)
    assert client.post("/v1/patients/me/assignment-invites", json={}).status_code == 401
    assert client.post("/v1/clinicians/me/assignments", headers=auth(client),
                       json={"subject_id": "subject-b"}).status_code == 422


def test_separate_valid_invite_reactivates_single_assignment():
    client = TestClient(app)
    clinician = auth(client)
    token, _ = issue_patient_token("subject-a")
    patient = {"Authorization": "Bearer " + token}
    def redeem():
        invite = client.post("/v1/patients/me/assignment-invites", headers=patient,
                             json={}).json()["invite_code"]
        return client.post("/v1/clinicians/me/assignments", headers=clinician,
                           json={"invite_code": invite})
    assert redeem().status_code == 200
    with SessionLocal() as db:
        assignment = db.scalar(select(ClinicianSubjectAssignment))
        assignment.active = False
        db.commit()
    assert redeem().status_code == 200
    with SessionLocal() as db:
        assignments = db.scalars(select(ClinicianSubjectAssignment)).all()
        assert len(assignments) == 1
        assert assignments[0].active


def test_clinician_unassign_requires_auth_and_preserves_patient_record():
    client = TestClient(app)
    clinician = auth(client)
    token, _ = issue_patient_token("subject-a")
    issued = client.post("/v1/patients/me/assignment-invites",
                         headers={"Authorization": "Bearer " + token}, json={})
    invitation = issued.json()["invite_code"]
    accepted = client.post("/v1/clinicians/me/assignments", headers=clinician,
                           json={"invite_code": invitation})
    assert accepted.status_code == 200

    endpoint = "/v1/clinicians/me/assignments/subject-a"
    assert client.delete(endpoint).status_code == 401
    assert client.delete("/v1/clinicians/me/assignments/subject-b",
                         headers=clinician).status_code == 404
    response = client.delete(endpoint, headers=clinician)
    assert response.status_code == 200
    assert response.json() == {
        "clinician_id": "DR001", "subject_id": "subject-a", "active": False
    }
    assert client.delete(endpoint, headers=clinician).status_code == 404
    assert client.get("/v1/clinicians/me/patients", headers=clinician).json()["patients"] == []
    assert client.get("/v1/patients/subject-a/assessment/latest",
                      headers=clinician).status_code == 403

    with SessionLocal() as db:
        assignment = db.scalar(select(ClinicianSubjectAssignment))
        assert not assignment.active
        assert assignment.ended_at is not None
        assert db.get(Subject, "subject-a") is not None
        audit = db.scalar(select(AuditLog).where(AuditLog.event == "assignment.ended"))
        assert audit.actor == "DR001" and audit.subject_id == "subject-a"


def test_unassign_requires_attention_events_to_be_resolved():
    client = TestClient(app)
    clinician = auth(client)
    with SessionLocal() as db:
        db.add(ClinicianSubjectAssignment(
            clinician_id="DR001", subject_id="subject-a"))
        db.add(AttentionEvent(
            id="pending-event", subject_id="subject-a",
            event_type="acute_escalation_forecast", severity="high",
            reason="test", forecast_horizon=10, status="OPEN",
            policy_version="test-policy"
        ))
        db.commit()
    endpoint = "/v1/clinicians/me/assignments/subject-a"
    assert client.delete(endpoint, headers=clinician).status_code == 409
    with SessionLocal() as db:
        assert db.scalar(select(ClinicianSubjectAssignment)).active
        event = db.get(AttentionEvent, "pending-event")
        event.status = "RESOLVED"
        db.commit()
    assert client.delete(endpoint, headers=clinician).status_code == 200
