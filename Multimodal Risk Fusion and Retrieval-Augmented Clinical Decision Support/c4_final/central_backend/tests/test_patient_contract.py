import datetime as dt
import os

os.environ.setdefault("MRN_PEPPER", "patient-contract-pepper")
os.environ.setdefault(
    "PATIENT_JWT_SECRET",
    "patient-contract-secret-that-is-long-enough-for-tests",
)

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from db_models import (
    AttentionEvent,
    AuditLog,
    ForecastResult,
    FusionResult,
    PatientCredential,
    SessionLocal,
    Subject,
    SubjectAlias,
    PairingCode,
    init_db,
)
from main import app


PARTICIPANT_ID = "P_0123456789ABCDEF"
INSTALLATION_SECRET = "installation-proof-with-more-than-thirty-two-characters"


def setup_function():
    init_db()
    with SessionLocal() as db:
        db.execute(delete(AttentionEvent))
        db.execute(delete(ForecastResult))
        db.execute(delete(FusionResult))
        db.execute(delete(AuditLog))
        db.execute(delete(PatientCredential))
        db.execute(delete(PairingCode))
        db.execute(delete(SubjectAlias))
        db.execute(delete(Subject))
        db.commit()


def _self_enrol(client: TestClient, secret: str = INSTALLATION_SECRET, pairing_code=None):
    return client.post(
        "/v1/subjects/self",
        json={"app_user_id": PARTICIPANT_ID, "installation_secret": secret,
              **({"pairing_code": pairing_code} if pairing_code else {})},
    )


def _patient_session(client: TestClient):
    body = _self_enrol(client).json()
    return body["subject_id"], {
        "Authorization": f"Bearer {body['access_token']}"
    }


def test_self_enrolment_issues_subject_bound_patient_session():
    response = _self_enrol(TestClient(app))

    assert response.status_code == 200
    body = response.json()
    assert body["subject_id"]
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert dt.datetime.fromisoformat(body["expires_at"]) > dt.datetime.now(
        dt.timezone.utc
    )

    claims = jwt.decode(
        body["access_token"],
        os.environ["PATIENT_JWT_SECRET"],
        algorithms=["HS256"],
        issuer="r26-central-backend",
        audience="aura",
    )
    assert claims["sub"] == body["subject_id"]
    assert claims["subject_id"] == body["subject_id"]
    assert claims["role"] == "patient"
    assert isinstance(claims["jti"], str) and claims["jti"]


def test_self_enrolment_requires_the_original_installation_proof():
    client = TestClient(app)
    first = _self_enrol(client)
    repeated = _self_enrol(client)
    wrong_proof = _self_enrol(
        client,
        "different-installation-proof-with-more-than-thirty-two-characters",
    )

    assert first.status_code == 200
    assert repeated.status_code == 200
    assert repeated.json()["subject_id"] == first.json()["subject_id"]
    assert wrong_proof.status_code == 403


def test_existing_subject_cannot_be_claimed_by_identifier_without_pairing_proof():
    from identity import hash_mrn
    from db_models import utcnow
    client = TestClient(app)
    with SessionLocal() as db:
        db.add(Subject(subject_id="preexisting"))
        db.add(SubjectAlias(subject_id="preexisting", alias_type="mrn_hash",
                            alias_value=hash_mrn(PARTICIPANT_ID)))
        db.add(PairingCode(code="PROOF123", subject_id="preexisting",
                           expires_at=utcnow() + dt.timedelta(minutes=5)))
        db.commit()
    assert _self_enrol(client).status_code == 403
    claim = _self_enrol(client, pairing_code="PROOF123")
    assert claim.status_code == 200
    assert claim.json()["subject_id"] == "preexisting"
    assert _self_enrol(client).status_code == 200
    with SessionLocal() as db:
        assert db.get(PairingCode, "PROOF123").used_at is not None
        db.get(PatientCredential, "preexisting").secret_hash = "invalid"
        db.commit()
    assert _self_enrol(client, pairing_code="PROOF123").status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer definitely-not-a-jwt"},
    ],
)
def test_patient_principal_rejects_missing_or_invalid_bearer(headers):
    response = TestClient(app, raise_server_exceptions=False).get(
        "/v1/patients/me",
        headers=headers,
    )
    assert response.status_code == 401


def test_patient_principal_rejects_expired_token():
    subject_id = _self_enrol(TestClient(app)).json()["subject_id"]
    now = dt.datetime.now(dt.timezone.utc)
    token = jwt.encode(
        {
            "sub": subject_id,
            "subject_id": subject_id,
            "role": "patient",
            "iss": "r26-central-backend",
            "aud": "aura",
            "iat": int((now - dt.timedelta(minutes=10)).timestamp()),
            "exp": int((now - dt.timedelta(minutes=5)).timestamp()),
            "jti": "expired-patient-session",
        },
        os.environ["PATIENT_JWT_SECRET"],
        algorithm="HS256",
    )

    response = TestClient(app, raise_server_exceptions=False).get(
        "/v1/patients/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("claim_name", "claim_value"),
    [
        ("sub", 123),
        ("subject_id", ["subject"]),
        ("iat", "not-a-numeric-date"),
        ("exp", "4102444800"),
        ("jti", 123),
    ],
)
def test_patient_principal_rejects_malformed_claim_types(claim_name, claim_value):
    subject_id = _self_enrol(TestClient(app)).json()["subject_id"]
    now = dt.datetime.now(dt.timezone.utc)
    claims = {
        "sub": subject_id,
        "subject_id": subject_id,
        "role": "patient",
        "iss": "r26-central-backend",
        "aud": "aura",
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=5)).timestamp()),
        "jti": "patient-session",
    }
    claims[claim_name] = claim_value
    token = jwt.encode(
        claims,
        os.environ["PATIENT_JWT_SECRET"],
        algorithm="HS256",
    )

    response = TestClient(app, raise_server_exceptions=False).get(
        "/v1/patients/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_patient_risk_requires_authentication_and_is_self_scoped():
    client = TestClient(app)
    subject_id, headers = _patient_session(client)
    with SessionLocal() as db:
        other = Subject(subject_id="other-patient")
        db.add(other)
        db.add(
            FusionResult(
                subject_id=subject_id,
                composite=0.58,
                tier="Medium",
                band="AMBER",
                confidence=0.71,
                modalities_used=3,
                model_version="ragf-v0.4",
                harmonisation={"assessment": {"status": "complete"}},
            )
        )
        db.commit()

    assert client.get(f"/v1/patients/{subject_id}/risk").status_code == 401
    own = client.get(f"/v1/patients/{subject_id}/risk", headers=headers)
    cross = client.get("/v1/patients/other-patient/risk", headers=headers)

    assert own.status_code == 200
    assert own.json()["subject_id"] == subject_id
    assert own.json()["fusion_result_id"] is not None
    assert cross.status_code == 403


def test_patient_risk_includes_server_tier_and_separate_physiological_forecast():
    client = TestClient(app)
    subject_id, headers = _patient_session(client)
    with SessionLocal() as db:
        db.add(FusionResult(subject_id=subject_id, composite=.58, tier="Medium",
                            band="AMBER", confidence=.71))
        db.add(ForecastResult(forecast_result_id="fcst_patient", subject_id=subject_id,
                              scope="physiological", horizon_minutes=10, score=.84,
                              tier="High", escalation_predicted=True,
                              generated_at=dt.datetime.now(dt.timezone.utc),
                              valid_until=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)))
        db.commit()
    body = client.get(f"/v1/patients/{subject_id}/risk", headers=headers).json()
    assert body["tier"] == "Medium"
    assert body["band"] == "AMBER"
    assert body["forecast"]["scope"] == "physiological"
    assert body["forecast"]["score"] == .84
    assert body["forecast"]["tier"] == "High"
    assert body["forecast"]["horizon_minutes"] == 10
    assert body["forecast"]["valid_until"]


def test_patient_attention_events_use_server_records_and_privacy_projection():
    client = TestClient(app)
    subject_id, headers = _patient_session(client)
    with SessionLocal() as db:
        fusion = FusionResult(
            subject_id=subject_id,
            composite=0.82,
            tier="High",
            band="RED",
            confidence=0.8,
            modalities_used=1,
            model_version="ragf-v0.4",
            harmonisation={"assessment": {"status": "partial"}},
        )
        db.add(fusion)
        db.flush()
        forecast = ForecastResult(
            forecast_result_id="fcst_patient_test",
            subject_id=subject_id,
            scope="physiological",
            horizon_minutes=10,
            score=0.84,
            tier="High",
            escalation_predicted=True,
            generated_at=dt.datetime.now(dt.timezone.utc),
            valid_until=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10),
        )
        db.add(forecast)
        db.flush()
        db.add(
            AttentionEvent(
                id="evt_shared_patient_test",
                subject_id=subject_id,
                fusion_result_id=fusion.id,
                forecast_result_id=forecast.forecast_result_id,
                event_type="acute_escalation_forecast",
                severity="high",
                reason="clinician-only policy explanation",
                forecast_horizon=10,
                status="OPEN",
                policy_version="escalation-v1",
            )
        )
        db.commit()

    response = client.get(
        "/v1/patients/me/attention-events?status=OPEN",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "events": [
            {
                "id": "evt_shared_patient_test",
                "event_type": "acute_escalation_forecast",
                "severity": "high",
                "forecast_horizon": 10,
                "status": "OPEN",
                "created_at": response.json()["events"][0]["created_at"],
                "policy_version": "escalation-v1",
            }
        ]
    }
    assert "reason" not in response.json()["events"][0]
    assert "subject_id" not in response.json()["events"][0]


def test_patient_ingest_rejects_cross_subject_identity_before_component_call():
    client = TestClient(app)
    _, headers = _patient_session(client)
    with SessionLocal() as db:
        db.add(Subject(subject_id="other-patient"))
        db.commit()

    contextual = client.post(
        "/v1/ingest/contextual",
        headers=headers,
        json={"subject_id": "other-patient", "gad7_items": [0] * 7},
    )
    physiological = client.post(
        "/v1/ingest/physiological",
        headers=headers,
        json={"subject_id": "other-patient"},
    )

    assert contextual.status_code == 403
    assert physiological.status_code == 403
