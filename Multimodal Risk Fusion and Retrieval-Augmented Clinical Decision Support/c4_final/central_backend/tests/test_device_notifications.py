"""Backend device registry and event outbox authorization contracts."""
import os

os.environ.setdefault("MRN_PEPPER", "device-notification-test-pepper")
os.environ.setdefault("CLINICIAN_JWT_SECRET", "device-test-clinician-secret")
os.environ.setdefault("PATIENT_JWT_SECRET", "device-test-patient-secret")

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import select

from clinician_api import hash_password
from db_models import (
    AttentionEvent, Clinician, ClinicianSubjectAssignment, DeviceToken,
    NotificationDelivery, SessionLocal, Subject, init_db,
)
from main import app
from notifications import queue_event_delivery
from patient_auth import issue_patient_token


def test_device_registration_and_scoped_delivery(monkeypatch):
    init_db()
    monkeypatch.setenv("DEVICE_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    client = TestClient(app)
    with SessionLocal() as db:
        db.add_all([
            Subject(subject_id="patient-notify-a"),
            Subject(subject_id="patient-notify-b"),
            Clinician(clinician_id="DRNOTIFY", display_name="Dr Notification",
                      role="clinician", password_hash=hash_password("sample-password")),
        ])
        db.flush()
        db.add(ClinicianSubjectAssignment(clinician_id="DRNOTIFY",
                    subject_id="patient-notify-a"))
        db.commit()
    patient_token, _ = issue_patient_token("patient-notify-a")
    patient = {"Authorization": f"Bearer {patient_token}"}
    clinician_login = client.post("/auth/login", json={
        "clinician_id": "DRNOTIFY", "password": "sample-password"})
    assert clinician_login.status_code == 200
    clinician = {"Authorization": f"Bearer {clinician_login.json()['access_token']}"}

    patient_reg = client.post("/v1/device-tokens", headers=patient, json={
        "token": "synthetic-patient-device-token-123", "platform": "android"})
    clinician_reg = client.post("/v1/device-tokens", headers=clinician, json={
        "token": "synthetic-clinician-device-token-123", "platform": "android"})
    assert patient_reg.status_code == clinician_reg.status_code == 200
    patient_device_id = patient_reg.json()["device_id"]
    clinician_device_id = clinician_reg.json()["device_id"]
    assert client.get("/v1/device-tokens", headers=clinician).json()["devices"][0]["device_id"] == clinician_device_id
    assert client.delete(f"/v1/device-tokens/{clinician_device_id}", headers=patient).status_code == 404
    assert client.post("/v1/device-tokens", json={
        "token": "synthetic-anonymous-device-token-123", "platform": "android"}).status_code == 401

    with SessionLocal() as db:
        device = db.get(DeviceToken, patient_device_id)
        assert "synthetic-patient-device-token" not in device.token_ciphertext
        e = AttentionEvent(id="evt_notify_test", subject_id="patient-notify-a",
                           severity="high", reason="synthetic forecast",
                           forecast_horizon=10)
        db.add(e)
        db.flush()
        assert queue_event_delivery(db, e) == 2
        db.commit()
        deliveries = db.scalars(select(NotificationDelivery).where(
            NotificationDelivery.event_id == e.id)).all()
        assert len(deliveries) == 2

    assert client.delete(f"/v1/device-tokens/{patient_device_id}", headers=patient).status_code == 204
    assert client.get("/v1/device-tokens", headers=patient).json()["devices"] == []

    from notification_worker import dispatch_once

    monkeypatch.setenv("ENABLE_PUSH_NOTIFICATIONS", "1")
    monkeypatch.setenv("FCM_PROJECT_ID", "synthetic-fcm-project")
    sent = []

    def synthetic_sender(token, event):
        sent.append((token, event.id))
        return 200

    result = dispatch_once(sender=synthetic_sender)
    assert result["sent"] == 1
    assert result["skipped"] == 1  # the revoked patient device is fail-closed
    assert sent == [("synthetic-clinician-device-token-123", "evt_notify_test")]
    with SessionLocal() as db:
        statuses = sorted(db.scalars(select(NotificationDelivery.status).where(
            NotificationDelivery.event_id == "evt_notify_test")).all())
        assert statuses == ["sent", "skipped"]
