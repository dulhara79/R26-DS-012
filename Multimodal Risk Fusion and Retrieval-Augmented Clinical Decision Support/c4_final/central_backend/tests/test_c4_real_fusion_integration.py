"""Backend-authoritative integration checks with the REAL local C4 model.

C1 and C3 are deterministic stubs to avoid network dependency; patient/clinician
auth, C4 inference, DB persistence, harmonisation, gating, fusion and projections
run through the production FastAPI routes. All subjects are synthetic.
"""
import math
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

os.environ.setdefault("MRN_PEPPER", "synthetic-real-c4-integration-pepper")
os.environ.setdefault("BACKEND_API_TOKEN", "synthetic-real-c4-service-token")
os.environ.setdefault("CLINICIAN_JWT_SECRET", "synthetic-real-c4-clinician-secret")
os.environ.setdefault("PATIENT_JWT_SECRET", "synthetic-real-c4-patient-secret")
os.environ.setdefault("FUSION_MODE", "inprocess")

import modality_clients as mc
from clinician_api import hash_password
from components.c4.inference import load_model
from db_models import (Clinician, ClinicianSubjectAssignment, FusionResult,
                       ModalityReading, SessionLocal, init_db)
from main import app

PROFILE = {
    "age": 21, "gender": "female", "edu": "bachelor's degree",
    "smoke": "never smokes", "drink": "never drinks",
}
GOLDEN_PROBABILITY = 0.0182847334986927


def _post(client, path, payload, headers=None):
    response = client.post(path, json=payload, headers=headers or {})
    assert response.status_code == 200, (path, response.status_code, response.text)
    return response.json()


def _new_patient(client):
    participant_id = "P_" + uuid.uuid4().hex[:16].upper()
    enrolment = _post(client, "/v1/subjects/self", {
        "app_user_id": participant_id,
        "installation_secret": "synthetic-installation-secret-40-characters-minimum",
    })
    return enrolment["subject_id"], participant_id, {
        "Authorization": "Bearer " + enrolment["access_token"]
    }


def _new_doctor(client, subject_id):
    doctor_id = "DR_C4_" + uuid.uuid4().hex[:10].upper()
    with SessionLocal() as db:
        db.add(Clinician(clinician_id=doctor_id, display_name="Synthetic C4 clinician",
                         password_hash=hash_password("synthetic-only-password")))
        db.add(ClinicianSubjectAssignment(clinician_id=doctor_id,
                                          subject_id=subject_id))
        db.commit()
    login = _post(client, "/auth/login", {
        "clinician_id": doctor_id, "password": "synthetic-only-password"
    })
    return {"Authorization": "Bearer " + login["access_token"]}


def _stub_dynamic_models(monkeypatch):
    monkeypatch.setattr(
        mc, "call_c1",
        lambda *args, **kwargs: mc.ComponentResult(
            raw_score=0.51, status="ok", confidence=0.8, coverage=1,
            model_version="c1-synthetic-stub"))
    monkeypatch.setattr(
        mc, "call_c3",
        lambda *args, **kwargs: mc.ComponentResult(
            raw_score=0.68, status="ok", confidence=0.75, coverage=1,
            model_version="c3-synthetic-stub"))


def _add_dynamic_modalities(client, subject_id, patient_headers, doctor_headers):
    _post(client, "/v1/ingest/physiological",
          {"subject_id": subject_id}, patient_headers)
    return _post(client, "/v1/clinical-notes", {
        "subject_id": subject_id,
        "note_text": "Synthetic note describes ongoing worry.",
        "support_set": [
            {"id": "synthetic-anx", "label": "anxiety",
             "text": "Persistent worry.", "note_date": "2026-10-01"},
        ],
    }, doctor_headers)


def test_real_local_c4_flows_from_authenticated_patient_to_complete_fusion(monkeypatch):
    init_db()
    client = TestClient(app)
    assert mc.CONFIGURED["c4_demographic"]() is True
    subject_id, _, patient_headers = _new_patient(client)
    doctor_headers = _new_doctor(client, subject_id)
    _stub_dynamic_models(monkeypatch)

    # Unlike the earlier episode test, DO NOT monkeypatch mc.call_c4.
    intake = _post(client, "/v1/ingest/contextual", {
        "subject_id": subject_id, **PROFILE, "gad7_items": [0, 1, 0, 0, 0, 0, 0]
    }, patient_headers)
    assert intake["status"] == "ok"
    assert math.isclose(intake["score"], GOLDEN_PROBABILITY, abs_tol=1e-13)
    assert intake["gad7_total"] == 1
    assert intake["fusion_triggered"] is True
    assert intake["fusion"]["composite"] is None
    assert intake["fusion"]["assessment_status"] == "insufficient"

    note = _add_dynamic_modalities(client, subject_id, patient_headers, doctor_headers)
    assert note["fusion_triggered"] is True
    assert note["fusion"]["assessment_status"] == "complete"
    with SessionLocal() as db:
        c4 = db.scalar(
            select(ModalityReading).where(
                ModalityReading.subject_id == subject_id,
                ModalityReading.modality == "c4_demographic",
            ).order_by(ModalityReading.id.desc()))
        assert c4 is not None and c4.status == "ok"
        assert math.isclose(c4.raw_score, GOLDEN_PROBABILITY, abs_tol=1e-13)
        assert c4.model_version == load_model()["model_version"]
        assert c4.detail["response"]["gad7_total"] == 1
        assert c4.detail["response"]["score_semantics"] == "calibrated_probability_gad7_ge10"

        fusion = db.scalar(
            select(FusionResult).where(FusionResult.subject_id == subject_id)
            .order_by(FusionResult.id.desc()))
        assert fusion is not None and fusion.composite is not None
        assert fusion.modalities_used == 3 and fusion.tier in ("Low", "Medium", "High")
        assert fusion.harmonisation["gate"]["passed"] is True
        assert fusion.harmonisation["gate"]["usable_modalities"] == [
            "c1_physiological", "c3_clinical_nlp", "c4_demographic"
        ]
        assert fusion.harmonisation["assessment"]["status"] == "complete"
        assert 0 < fusion.weights["c4_demographic"] <= 0.35001
        assert abs(sum(fusion.weights.values()) - 1.0) < .001
        h = fusion.harmonisation["c4_demographic"]
        assert math.isclose(h["raw"], GOLDEN_PROBABILITY, abs_tol=0.0001)
        assert 0 <= h["harmonised"] <= 1

    clinician_assessment = client.get(
        f"/v1/patients/{subject_id}/assessment/latest",
        headers=doctor_headers)
    assert clinician_assessment.status_code == 200, clinician_assessment.text
    body = clinician_assessment.json()
    assert body["assessment_status"] == "complete"
    c4_wire = next(m for m in body["modalities"]
                   if m["component_id"] == "c4_demographic")
    assert c4_wire["status"] == "ok"
    assert c4_wire["included_in_fusion"] is True
    assert math.isclose(c4_wire["score"], GOLDEN_PROBABILITY, abs_tol=1e-13)
    patient_risk = client.get(
        f"/v1/patients/{subject_id}/risk", headers=patient_headers)
    dashboard = client.get("/v1/clinicians/me/dashboard", headers=doctor_headers)
    assert patient_risk.status_code == 200 and dashboard.status_code == 200
    assert body["fusion_result_id"] == patient_risk.json()["fusion_result_id"]
    assert body["fusion_result_id"] == dashboard.json()["patients"][0]["fusion_result_id"]


def test_actual_aura_contextual_shape_marks_c4_unavailable_but_keeps_c1_c3(monkeypatch):
    """Current mobile client omits smoking/drinking and uses non-trained education labels."""
    init_db()
    client = TestClient(app)
    subject_id, _, patient_headers = _new_patient(client)
    doctor_headers = _new_doctor(client, subject_id)
    _stub_dynamic_models(monkeypatch)
    mobile_shape = {
        "subject_id": subject_id, "gender": "female", "age": 21,
        "edu": "tertiary", "gad7_items": [0] * 7,
    }
    intake = _post(client, "/v1/ingest/contextual", mobile_shape, patient_headers)
    assert intake["status"] == "poor_signal"
    assert intake["score"] is None
    assert intake["fusion"]["assessment_status"] == "insufficient"

    note = _add_dynamic_modalities(client, subject_id, patient_headers, doctor_headers)
    assert note["fusion_triggered"] is True
    assert note["fusion"]["assessment_status"] == "provisional"
    with SessionLocal() as db:
        fusion = db.scalar(select(FusionResult).where(
            FusionResult.subject_id == subject_id).order_by(FusionResult.id.desc()))
        assert fusion.composite is not None
        assert fusion.modalities_used == 2
        assert "status=poor_signal" in (
            fusion.harmonisation["gate"]["rejected"]["c4_demographic"])
        assert fusion.weights["c4_demographic"] == 0
    result = client.get(
        f"/v1/patients/{subject_id}/assessment/latest",
        headers=doctor_headers)
    assert result.status_code == 200
    body = result.json()
    assert body["assessment_status"] == "partial"
    c4_wire = next(m for m in body["modalities"]
                   if m["component_id"] == "c4_demographic")
    assert not c4_wire["available"]
    assert not c4_wire["included_in_fusion"]
