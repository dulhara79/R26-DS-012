"""The old clinician routes must honor the same JWT and assignment as P0."""

import os

os.environ.setdefault("MRN_PEPPER", "legacy-route-contract-pepper")
os.environ.setdefault("CLINICIAN_JWT_SECRET", "legacy-route-contract-secret-long-enough")

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

import identity
from clinician_api import hash_password
from db_models import (
    AuditLog, AttentionEvent, Clinician, ClinicianSubjectAssignment,
    EscalationEpisode, ForecastResult, FusionResult, ModalityReading,
    PairingCode, PatientCredential, SessionLocal, Subject, SubjectAlias, Verdict,
    init_db,
)
from main import app
import modality_clients as mc


def setup_function():
    init_db()
    with SessionLocal() as db:
        for model in (AuditLog, AttentionEvent, EscalationEpisode,
                      ForecastResult, Verdict, ClinicianSubjectAssignment,
                      FusionResult, ModalityReading, PairingCode,
                      PatientCredential, SubjectAlias, Clinician, Subject):
            db.execute(delete(model))
        db.add_all([
            Clinician(clinician_id="DR001", display_name="Dr One",
                      password_hash=hash_password("secret")),
            Clinician(clinician_id="DR002", display_name="Dr Two",
                      password_hash=hash_password("secret")),
            Subject(subject_id="owned-subject"),
            Subject(subject_id="other-subject"),
            ClinicianSubjectAssignment(clinician_id="DR001", subject_id="owned-subject"),
            SubjectAlias(subject_id="other-subject", alias_type="app_user_id",
                         alias_value="P_0123456789ABCDEF"),
        ])
        db.flush()
        db.add(FusionResult(subject_id="other-subject", composite=.58,
                            tier="Medium", band="AMBER", confidence=.71))
        db.commit()


def clinician_auth(client, clinician_id="DR001"):
    login = client.post("/auth/login", json={"clinician_id": clinician_id, "password": "secret"})
    assert login.status_code == 200
    return {"Authorization": "Bearer " + login.json()["access_token"]}


def test_legacy_subject_routes_reject_unassigned_clinician():
    client = TestClient(app, raise_server_exceptions=False)
    headers = clinician_auth(client)
    with SessionLocal() as db:
        fusion_id = db.scalar(select(FusionResult.id).where(FusionResult.subject_id == "other-subject"))
    requests = [
        client.get("/v1/subjects/resolve?app_user_id=P_0123456789ABCDEF", headers=headers),
        client.post("/v1/subjects/other-subject/external-ids", headers=headers,
                    json={"modality": "c1_physiological", "external_id": "device-x"}),
        client.post("/v1/fusion/run", headers=headers,
                    json={"subject_id": "other-subject", "trigger": "manual"}),
        client.post("/v1/verdict", headers=headers,
                    json={"fusion_result_id": fusion_id, "tier_label": "High", "author": "DR002"}),
        client.get("/v1/doctor/patients/other-subject/timeline", headers=headers),
        client.get("/v1/doctor/patients/other-subject/explanation", headers=headers),
        client.post("/v1/doctor/patients/other-subject/evidence", headers=headers,
                    json={"question": "What evidence?"}),
    ]
    assert [response.status_code for response in requests] == [403] * len(requests)
    with SessionLocal() as db:
        assert not db.scalars(select(Verdict)).all()
        assert not db.scalars(select(SubjectAlias).where(SubjectAlias.alias_value == "device-x")).all()


def test_clinician_enrolment_assigns_only_new_subject_and_records_jwt_actor():
    client = TestClient(app)
    headers = clinician_auth(client)
    created = client.post("/v1/subjects", headers=headers,
                          json={"mrn": "new-test-mrn", "enrolled_by": "DR002"})
    assert created.status_code == 200
    subject_id = created.json()["subject_id"]
    assert client.get("/v1/subjects/resolve?mrn=new-test-mrn", headers=headers).json() == {"subject_id": subject_id}
    with SessionLocal() as db:
        assert db.get(Subject, subject_id).enrolled_by == "DR001"
        assert db.scalar(select(ClinicianSubjectAssignment).where(
            ClinicianSubjectAssignment.subject_id == subject_id,
            ClinicianSubjectAssignment.clinician_id == "DR001")).active
        assert db.scalar(select(AuditLog).where(AuditLog.subject_id == subject_id,
                                                AuditLog.event == "enrol.created")).actor == "DR001"
    assert client.post("/v1/subjects", headers=clinician_auth(client, "DR002"),
                       json={"mrn": "new-test-mrn"}).status_code == 403


def test_assigned_clinician_verdict_ignores_spoofed_author():
    client = TestClient(app)
    headers = clinician_auth(client)
    with SessionLocal() as db:
        db.add(FusionResult(subject_id="owned-subject", composite=.58,
                            tier="Medium", band="AMBER", confidence=.7))
        db.commit()
        fusion_id = db.scalar(select(FusionResult.id).where(FusionResult.subject_id == "owned-subject"))
    response = client.post("/v1/verdict", headers=headers,
                           json={"fusion_result_id": fusion_id, "tier_label": "High", "author": "DR002"})
    assert response.status_code == 200
    with SessionLocal() as db:
        assert db.scalar(select(Verdict).where(Verdict.fusion_result_id == fusion_id)).author == "DR001"


def test_unconfigured_service_token_never_permits_anonymous_clinical_note(monkeypatch):
    monkeypatch.delenv("BACKEND_API_TOKEN", raising=False)
    response = TestClient(app).post("/v1/clinical-notes",
                                    json={"subject_id": "owned-subject", "note_text": "test"})
    assert response.status_code == 401


def test_clinician_jwt_preserves_clinical_note_to_c3_to_fusion(monkeypatch):
    client = TestClient(app)
    headers = clinician_auth(client)
    calls = []

    def c3_stub(*args, **kwargs):
        calls.append(kwargs["subject_external_id"])
        return mc.ComponentResult(raw_score=.65, status="ok", confidence=.6,
                                  coverage=1.0, detail={"status": "ok"})

    monkeypatch.setattr(mc, "call_c3", c3_stub)
    note = {"subject_id": "owned-subject", "note_text": "Synthetic progress note",
            "author": "DR002", "support_set": [{"id": "s1", "text": "reference", "label": "anxiety"}]}
    response = client.post("/v1/clinical-notes", headers=headers, json=note)
    assert response.status_code == 200
    assert response.json()["fusion_triggered"] is True
    assert response.json()["score"] == .65
    assert calls == ["owned-subject"]
    assert client.post("/v1/clinical-notes", headers=headers,
                       json={**note, "subject_id": "other-subject"}).status_code == 403
