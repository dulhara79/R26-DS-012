"""A failed clinician login must not expose credentials or account existence."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from clinician_api import hash_password
from db_models import AuditLog, Base, Clinician, get_session
from main import app


def test_login_audits_without_disclosing_credentials(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def test_session():
        with Session() as db:
            yield db

    app.dependency_overrides[get_session] = test_session
    monkeypatch.setenv("CLINICIAN_JWT_SECRET", "synthetic-test-secret-at-least-32-bytes")
    try:
        with Session() as db:
            db.add(Clinician(clinician_id="DRTEST", display_name="Dr Test",
                             role="clinician", active=True,
                             password_hash=hash_password("safe-synthetic-password")))
            db.commit()
        with TestClient(app) as client:
            for clinician_id in ("DRTEST", "NOT_REGISTERED"):
                rejected = client.post("/auth/login", json={
                    "clinician_id": clinician_id, "password": "definitely-wrong"})
                assert rejected.status_code == 401
                assert rejected.json()["detail"] == "invalid credentials"
            accepted = client.post("/auth/login", json={
                "clinician_id": "DRTEST", "password": "safe-synthetic-password"})
            assert accepted.status_code == 200 and accepted.json()["access_token"]

        with Session() as db:
            audit = db.scalars(select(AuditLog).where(
                AuditLog.event.like("auth.login.%")).order_by(AuditLog.id)).all()
            assert [row.event for row in audit] == [
                "auth.login.failure", "auth.login.failure", "auth.login.success"]
            assert [row.actor for row in audit] == [
                "unauthenticated", "unauthenticated", "DRTEST"]
            assert all("safe-synthetic-password" not in str(row.detail) for row in audit)
            assert all("definitely-wrong" not in str(row.detail) for row in audit)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
