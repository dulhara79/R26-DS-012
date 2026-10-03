"""One synthetic Aura -> Central Backend -> ClinAnx episode through public routes.

Component responses are deterministic; this is an API contract rehearsal, not
evidence that external C1/C3/C4 services or mobile devices are deployed.
"""

import datetime as dt
import os

os.environ.setdefault("MRN_PEPPER", "synthetic-episode-pepper")
os.environ.setdefault("BACKEND_API_TOKEN", "synthetic-episode-service-token")
os.environ.setdefault("CLINICIAN_JWT_SECRET", "synthetic-episode-clinician-secret")
os.environ.setdefault("PATIENT_JWT_SECRET", "synthetic-episode-patient-secret")
os.environ.setdefault("FUSION_MODE", "inprocess")

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

import forecast
import modality_clients as mc
from clinician_api import hash_password
from db_models import (
    AttentionEvent, AuditLog, Clinician, ClinicianAssignmentInvite,
    ClinicianSubjectAssignment, EscalationEpisode, ForecastResult, FusionResult,
    ModalityReading, PairingCode, PatientCredential, SessionLocal, Subject,
    SubjectAlias, Verdict, init_db, utcnow,
)
from main import app


def setup_function():
    init_db()
    with SessionLocal() as db:
        for model in (
            AuditLog, AttentionEvent, EscalationEpisode, ForecastResult, Verdict,
            ClinicianSubjectAssignment, ClinicianAssignmentInvite, FusionResult,
            ModalityReading, PairingCode, PatientCredential, SubjectAlias,
            Clinician, Subject,
        ):
            db.execute(delete(model))
        db.add_all([
            Clinician(clinician_id=clinician_id, display_name=clinician_id,
                      password_hash=hash_password("synthetic-password"))
            for clinician_id in ("DR001", "DR002")
        ])
        db.commit()


def _login(client, clinician_id):
    response = client.post("/auth/login", json={
        "clinician_id": clinician_id, "password": "synthetic-password",
    })
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _get(client, path, headers):
    response = client.get(path, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _post(client, path, headers, body):
    response = client.post(path, headers=headers, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_synthetic_episode_from_patient_invite_to_persisted_resolution(monkeypatch):
    client = TestClient(app)
    patient_id = "P_0123456789ABCDEF"
    enrol = _post(client, "/v1/subjects/self", {}, {
        "app_user_id": patient_id,
        "installation_secret": "synthetic-installation-proof-longer-than-32-chars",
    })
    subject_id = enrol["subject_id"]
    patient = {"Authorization": f"Bearer {enrol['access_token']}"}
    clinician = _login(client, "DR001")
    unassigned = _login(client, "DR002")

    latest = f"/v1/patients/{subject_id}/assessment/latest"
    assert client.get(latest, headers=clinician).status_code == 403
    invite = _post(client, "/v1/patients/me/assignment-invites", patient, {})
    accepted = _post(client, "/v1/clinicians/me/assignments", clinician,
                     {"invite_code": invite["invite_code"]})
    assert accepted == {"clinician_id": "DR001", "subject_id": subject_id,
                        "active": True}
    assert client.post("/v1/clinicians/me/assignments", headers=unassigned,
                       json={"invite_code": invite["invite_code"]}).status_code == 409
    assert _get(client, "/v1/clinicians/me/patients", clinician)["patients"][0]["subject_id"] == subject_id

    # The clock moves 30 seconds between independent source-stamped C1 windows.
    # No sleep or production clock/policy changes are needed for this rehearsal.
    now = utcnow()
    windows = [now - dt.timedelta(seconds=30), now, now + dt.timedelta(seconds=1)]
    monkeypatch.setattr(forecast, "utcnow", lambda: windows.pop(0))
    c1_windows = [now - dt.timedelta(seconds=30), now, now]

    def c1(_user_id):
        captured_at = c1_windows.pop(0)
        return mc.ComponentResult(
            raw_score=.50, status="ok", confidence=.8, coverage=1.0,
            model_version="c1-synthetic", captured_at=captured_at,
            detail={"risk_forecast": [.50] * 9 + [.84],
                    "latest_reading_at": captured_at.isoformat()},
        )

    monkeypatch.setattr(mc, "call_c1", c1)
    monkeypatch.setattr(mc, "call_c4", lambda *_args, **_kwargs:
                        mc.ComponentResult(raw_score=.55, status="ok",
                                           confidence=.7, coverage=1.0))
    monkeypatch.setattr(mc, "call_c3", lambda *_args, **_kwargs:
                        mc.ComponentResult(raw_score=.68, status="ok",
                                           confidence=.7, coverage=1.0))

    first = _post(client, "/v1/ingest/physiological", patient,
                  {"app_user_id": patient_id})
    assert first["forecast_result_id"] and "attention_event_id" not in first
    _post(client, "/v1/ingest/contextual", patient,
          {"subject_id": subject_id, "gender": "female", "age": 30})
    note = _post(client, "/v1/clinical-notes", clinician, {
        "subject_id": subject_id,
        "note_text": "Synthetic patient reports persistent worry and restlessness.",
        "support_set": [{"id": "synthetic-anx", "label": "anxiety",
                         "text": "Persistent worry", "note_date": "2026-09-01"}],
    })
    assert note["fusion_triggered"] is True

    second = _post(client, "/v1/ingest/physiological", patient,
                   {"app_user_id": patient_id})
    event_id = second["attention_event_id"]
    assert second["forecast_result_id"] != first["forecast_result_id"]
    assessment = _get(client, latest, clinician)
    risk = _get(client, f"/v1/patients/{subject_id}/risk", patient)
    dashboard = _get(client, "/v1/clinicians/me/dashboard", clinician)
    events = _get(client, "/v1/attention-events?status=OPEN", clinician)["events"]
    patient_events = _get(client, "/v1/patients/me/attention-events?status=OPEN", patient)["events"]
    assert assessment["assessment_status"] == "complete"
    assert assessment["current_assessment"]["score"] == risk["composite"]
    assert assessment["forecast"]["scope"] == risk["forecast"]["scope"] == "physiological"
    assert assessment["fusion_result_id"] == risk["fusion_result_id"] == dashboard["patients"][0]["fusion_result_id"]
    assert dashboard["assigned_count"] == 1
    assert dashboard["patients"][0]["open_event_count"] == 1
    assert [item["id"] for item in events] == [event_id]
    assert [item["id"] for item in patient_events] == [event_id]
    assert events[0]["fusion_result_id"] == assessment["fusion_result_id"]
    assert events[0]["forecast_result_id"] == second["forecast_result_id"]
    assert "subject_id" not in patient_events[0] and "reason" not in patient_events[0]
    assert client.get(f"/v1/attention-events/{event_id}", headers=unassigned).status_code == 403
    assert client.get(latest, headers=unassigned).status_code == 403

    # A repeated source window cannot create another episode or event.
    duplicate = _post(client, "/v1/ingest/physiological", patient,
                      {"app_user_id": patient_id})
    assert "attention_event_id" not in duplicate
    assert [item["id"] for item in _get(client, "/v1/attention-events", clinician)["events"]] == [event_id]
    # Read/ACK/RESOLVE are all served from persistent backend state.
    assert client.post(f"/v1/attention-events/{event_id}/resolve", headers=clinician,
                       json={}).status_code == 409
    acknowledged = _post(client, f"/v1/attention-events/{event_id}/acknowledge",
                         clinician, {})["event"]
    assert acknowledged["status"] == "ACKNOWLEDGED"
    assert acknowledged["acknowledged_by"] == "DR001" and acknowledged["acknowledged_at"]
    assert client.post(f"/v1/attention-events/{event_id}/acknowledge",
                       headers=clinician, json={}).status_code == 409
    resolved = _post(client, f"/v1/attention-events/{event_id}/resolve",
                     clinician, {})["event"]
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolved_by"] == "DR001" and resolved["resolved_at"]
    assert _get(client, f"/v1/attention-events/{event_id}", clinician)["event"] == resolved
    assert _get(client, "/v1/patients/me/attention-events", patient)["events"][0]["status"] == "RESOLVED"
    assert _get(client, "/v1/clinicians/me/dashboard", clinician)["open_attention_events"] == []
    with SessionLocal() as db:
        rows = db.scalars(select(AttentionEvent).where(AttentionEvent.subject_id == subject_id)).all()
        assert len(rows) == 1 and rows[0].id == event_id and rows[0].status == "RESOLVED"
