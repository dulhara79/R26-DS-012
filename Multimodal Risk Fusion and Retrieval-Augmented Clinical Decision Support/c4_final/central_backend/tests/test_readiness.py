"""Operational readiness must not be inferred from a running HTTP process."""

import os

os.environ.setdefault("MRN_PEPPER", "readiness-test-pepper")

from fastapi.testclient import TestClient

from db_models import engine
from main import app
from migrate_p0 import run_migrations


def test_readiness_checks_schema_database_and_patient_and_clinician_secrets(monkeypatch):
    run_migrations(engine)
    monkeypatch.setenv("CLINICIAN_JWT_SECRET", "test-clinician-secret")
    monkeypatch.setenv("PATIENT_JWT_SECRET", "test-patient-secret")
    monkeypatch.setenv("BACKEND_API_TOKEN", "test-internal-service-token")
    client = TestClient(app)
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["database"] == "ready"
    assert ready.json()["schema_revision"] == "0002_assignment_invites_and_forecast_link"
    assert ready.json()["auth"] == {"clinician": True, "patient": True}
    monkeypatch.delenv("PATIENT_JWT_SECRET")
    assert client.get("/ready").status_code == 503
