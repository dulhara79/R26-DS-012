import datetime as dt
import os

os.environ.setdefault("CLINICIAN_JWT_SECRET", "test-secret-that-is-long-enough-for-tests")
os.environ.setdefault(
    "PATIENT_JWT_SECRET",
    "patient-contract-secret-that-is-long-enough-for-tests",
)

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from clinician_api import hash_password
from db_models import (AttentionEvent, AuditLog, Clinician, ClinicianAssignmentInvite, ClinicianSubjectAssignment,
                       EscalationEpisode, ForecastResult, FusionResult,
                       ModalityReading, SessionLocal, Subject, init_db, utcnow)
from forecast import persist_c1_forecast_and_event
from main import app
from patient_auth import issue_patient_token


def setup_function():
    init_db()
    with SessionLocal() as db:
        for model in (AuditLog, AttentionEvent, EscalationEpisode, ForecastResult,
                      ClinicianAssignmentInvite,
                      ClinicianSubjectAssignment, FusionResult, ModalityReading,
                      Clinician, Subject):
            db.execute(delete(model))
        db.commit()


def seed():
    with SessionLocal() as db:
        db.add(Clinician(clinician_id="DR001", display_name="Dr X", role="clinician",
                         password_hash=hash_password("secret")))
        db.add_all([Subject(subject_id="patient-a"), Subject(subject_id="patient-b")])
        db.flush()
        db.add(ClinicianSubjectAssignment(clinician_id="DR001", subject_id="patient-a"))
        fusion = FusionResult(subject_id="patient-a", composite=.58, tier="Medium", band="AMBER",
            confidence=.71, modalities_used=3, model_version="ragf-v0.4",
            harmonisation={"assessment": {"status": "complete"}})
        db.add(fusion); db.flush()
        forecast = ForecastResult(forecast_result_id="fcst_test", subject_id="patient-a",
            scope="physiological", horizon_minutes=10, score=.84, tier="High",
            escalation_predicted=True, generated_at=utcnow(), valid_until=utcnow()+dt.timedelta(minutes=10))
        db.add(forecast); db.flush()
        db.add(AttentionEvent(id="evt_test", subject_id="patient-a", fusion_result_id=fusion.id,
            forecast_result_id=forecast.forecast_result_id, severity="high", reason="policy",
            forecast_horizon=10))
        db.commit()


def auth(client, clinician_id="DR001"):
    response = client.post(
        "/auth/login",
        json={"clinician_id": clinician_id, "password": "secret"},
    )
    assert response.status_code == 200
    assert response.json()["clinician"]["clinician_id"] == clinician_id
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def patient_auth(subject_id):
    token, _ = issue_patient_token(subject_id)
    return {"Authorization": f"Bearer {token}"}


def test_auth_dashboard_assessment_and_assignment_scope():
    seed(); client = TestClient(app); headers = auth(client)
    assert client.get("/v1/me", headers=headers).status_code == 200
    dashboard = client.get("/v1/clinicians/me/dashboard", headers=headers).json()
    assert dashboard["assigned_count"] == 1
    assert dashboard["patients"][0]["fusion_result_id"] == dashboard["open_attention_events"][0]["fusion_result_id"]
    assessment = client.get("/v1/patients/patient-a/assessment/latest", headers=headers).json()
    assert assessment["current_assessment"] == {"score": .58, "tier": "Medium", "band": "AMBER"}
    assert assessment["forecast"]["scope"] == "physiological"
    assert client.get("/v1/patients/patient-b/assessment/latest", headers=headers).status_code == 403


def test_patient_summary_uses_stable_pseudonymous_display_id():
    seed(); client = TestClient(app); headers = auth(client)
    first = client.get("/v1/clinicians/me/dashboard", headers=headers).json()["patients"][0]
    second = client.get("/v1/clinicians/me/patients", headers=headers).json()["patients"][0]
    assert first["display_id"] == second["display_id"]
    assert first["display_id"].startswith("Patient ")
    assert first["display_id"] != first["subject_id"]


def test_attention_lifecycle_is_atomic_and_server_attributed():
    seed(); client = TestClient(app); headers = auth(client)
    assert client.get("/v1/attention-events/evt_test", headers=headers).status_code == 200
    ack = client.post("/v1/attention-events/evt_test/acknowledge", headers=headers, json={})
    assert ack.status_code == 200 and ack.json()["event"]["acknowledged_by"] == "DR001"
    assert client.post("/v1/attention-events/evt_test/acknowledge", headers=headers, json={}).status_code == 409
    resolved = client.post("/v1/attention-events/evt_test/resolve", headers=headers, json={})
    assert resolved.status_code == 200 and resolved.json()["event"]["resolved_by"] == "DR001"
    assert client.post("/v1/attention-events/evt_test/resolve", headers=headers, json={}).status_code == 409


def test_attention_lifecycle_writes_actor_attributed_audit_without_note_text():
    seed(); client = TestClient(app); headers = auth(client)
    assert client.get("/v1/attention-events/evt_test", headers=headers).status_code == 200
    assert client.post("/v1/attention-events/evt_test/acknowledge", headers=headers, json={}).status_code == 200
    assert client.post("/v1/attention-events/evt_test/resolve", headers=headers,
                       json={"note": "private clinical note"}).status_code == 200
    with SessionLocal() as db:
        records = db.scalars(select(AuditLog).where(
            AuditLog.subject_id == "patient-a",
            AuditLog.event.in_(("attention.read", "attention.acknowledged", "attention.resolved"))
        ).order_by(AuditLog.id)).all()
    assert [record.event for record in records] == ["attention.read", "attention.acknowledged", "attention.resolved"]
    assert all(record.actor == "DR001" and record.created_at for record in records)
    assert all(record.detail["event_id"] == "evt_test" for record in records)
    assert all("private clinical note" not in str(record.detail) for record in records)


def test_assessment_reads_and_unassigned_denials_are_audited():
    seed(); client = TestClient(app); headers = auth(client)
    assert client.get("/v1/patients/patient-a/assessment/latest", headers=headers).status_code == 200
    assert client.get("/v1/patients/patient-b/assessment/latest", headers=headers).status_code == 403
    with SessionLocal() as db:
        allowed = db.scalar(select(AuditLog).where(AuditLog.subject_id == "patient-a",
                                                   AuditLog.event == "assessment.read"))
        denied = db.scalar(select(AuditLog).where(AuditLog.subject_id == "patient-b",
                                                  AuditLog.event == "access.denied"))
    assert allowed.actor == "DR001"
    assert denied.actor == "DR001"


def test_data_quality_uses_fusion_freshness_and_explicit_exclusion():
    seed()
    with SessionLocal() as db:
        row = db.scalar(select(FusionResult).where(FusionResult.subject_id == "patient-a"))
        row.weights = {"c1_physiological": .3}
        db.add(ModalityReading(subject_id="patient-a", modality="c1_physiological",
                               raw_score=.58, status="ok", confidence=.8, coverage=.9,
                               captured_at=utcnow() - dt.timedelta(minutes=3)))
        db.commit()
    client = TestClient(app)
    response = client.get("/v1/patients/patient-a/data-quality", headers=auth(client))
    assert response.status_code == 200
    quality = {modality["component_id"]: modality for modality in response.json()["modalities"]}
    assert quality["c1_physiological"]["available"] is True
    assert quality["c1_physiological"]["max_age_minutes"] == 15
    assert 2.5 < quality["c1_physiological"]["freshness_age_minutes"] < 4
    assert quality["c2_behavioral"]["exclusion_reason"] == "research-only"


def test_resolution_note_is_optional_normalized_and_server_persisted():
    seed()
    client = TestClient(app)
    headers = auth(client)
    assert client.post(
        "/v1/attention-events/evt_test/acknowledge",
        headers=headers,
        json={},
    ).status_code == 200

    resolved = client.post(
        "/v1/attention-events/evt_test/resolve",
        headers=headers,
        json={"note": "  Follow-up arranged with patient.  "},
    )
    detail = client.get("/v1/attention-events/evt_test", headers=headers)

    assert resolved.status_code == 200
    assert resolved.json()["event"]["resolution_note"] == (
        "Follow-up arranged with patient."
    )
    assert detail.json()["event"]["resolution_note"] == (
        "Follow-up arranged with patient."
    )


def test_resolution_note_rejects_unknown_or_overlong_fields():
    seed()
    client = TestClient(app)
    headers = auth(client)
    assert client.post(
        "/v1/attention-events/evt_test/acknowledge",
        headers=headers,
        json={},
    ).status_code == 200

    assert client.post(
        "/v1/attention-events/evt_test/resolve",
        headers=headers,
        json={"reason": "not the frozen field"},
    ).status_code == 422
    assert client.post(
        "/v1/attention-events/evt_test/resolve",
        headers=headers,
        json={"note": "x" * 256},
    ).status_code == 422


def test_openapi_contains_frozen_paths_and_strict_empty_body():
    schema = TestClient(app).get("/openapi.json").json()
    for path in ("/auth/login", "/v1/me", "/v1/clinicians/me/dashboard",
                 "/v1/clinicians/me/patients", "/v1/patients/{subject_id}/assessment/latest",
                 "/v1/patients/{subject_id}/assessments", "/v1/patients/{subject_id}/data-quality",
                 "/v1/attention-events", "/v1/attention-events/{event_id}",
                 "/v1/attention-events/{event_id}/acknowledge", "/v1/attention-events/{event_id}/resolve"):
        assert path in schema["paths"]
    empty = schema["components"]["schemas"]["EmptyBody"]
    assert empty["additionalProperties"] is False


def test_non_clinician_role_cannot_use_clinician_api():
    with SessionLocal() as db:
        db.add(Clinician(clinician_id="ADMIN1", display_name="Admin", role="admin",
                         password_hash=hash_password("secret")))
        db.commit()
    client = TestClient(app)
    login = client.post("/auth/login", json={"clinician_id": "ADMIN1", "password": "secret"})
    assert login.status_code == 403


def test_signed_token_missing_required_claim_is_401():
    seed()
    now = dt.datetime.now(dt.timezone.utc)
    token = jwt.encode(
        {
            "sub": "DR001",
            "clinician_id": "DR001",
            "role": "clinician",
            "iss": "r26-central-backend",
            "aud": "clinanx",
            "iat": int(now.timestamp()),
            "exp": int((now + dt.timedelta(minutes=5)).timestamp()),
            # A token without jti is not a valid clinician session.
        },
        os.environ["CLINICIAN_JWT_SECRET"],
        algorithm="HS256",
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/v1/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("claim_name", "claim_value"),
    [
        ("clinician_id", ["DR001"]),
        ("sub", 123),
        ("iat", "not-a-numeric-date"),
        ("exp", "4102444800"),
        ("jti", 123),
    ],
)
def test_signed_token_with_malformed_claim_type_is_401(claim_name, claim_value):
    seed()
    now = dt.datetime.now(dt.timezone.utc)
    claims = {
        "sub": "DR001",
        "clinician_id": "DR001",
        "role": "clinician",
        "iss": "r26-central-backend",
        "aud": "clinanx",
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=5)).timestamp()),
        "jti": "session-1",
    }
    claims[claim_name] = claim_value
    token = jwt.encode(
        claims,
        os.environ["CLINICIAN_JWT_SECRET"],
        algorithm="HS256",
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/v1/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer definitely-not-a-jwt"},
    ],
)
def test_missing_or_invalid_bearer_token_is_401(headers):
    seed()
    response = TestClient(app, raise_server_exceptions=False).get(
        "/v1/me",
        headers=headers,
    )
    assert response.status_code == 401


def test_expired_clinician_token_is_401():
    seed()
    now = dt.datetime.now(dt.timezone.utc)
    token = jwt.encode(
        {
            "sub": "DR001",
            "clinician_id": "DR001",
            "role": "clinician",
            "iss": "r26-central-backend",
            "aud": "clinanx",
            "iat": int((now - dt.timedelta(minutes=10)).timestamp()),
            "exp": int((now - dt.timedelta(minutes=5)).timestamp()),
            "jti": "expired-session",
        },
        os.environ["CLINICIAN_JWT_SECRET"],
        algorithm="HS256",
    )
    response = TestClient(app, raise_server_exceptions=False).get(
        "/v1/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("secret", "issuer", "audience"),
    [
        ("different-test-secret-that-is-long-enough", "r26-central-backend", "clinanx"),
        (os.environ["CLINICIAN_JWT_SECRET"], "wrong-issuer", "clinanx"),
        (os.environ["CLINICIAN_JWT_SECRET"], "r26-central-backend", "wrong-audience"),
    ],
)
def test_wrong_signature_issuer_or_audience_is_401(secret, issuer, audience):
    seed()
    now = dt.datetime.now(dt.timezone.utc)
    token = jwt.encode(
        {
            "sub": "DR001",
            "clinician_id": "DR001",
            "role": "clinician",
            "iss": issuer,
            "aud": audience,
            "iat": int(now.timestamp()),
            "exp": int((now + dt.timedelta(minutes=5)).timestamp()),
            "jti": "invalid-session",
        },
        secret,
        algorithm="HS256",
    )
    response = TestClient(app, raise_server_exceptions=False).get(
        "/v1/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_attention_event_access_is_assignment_scoped():
    seed()
    with SessionLocal() as db:
        db.add(
            Clinician(
                clinician_id="DR002",
                display_name="Dr Y",
                role="clinician",
                password_hash=hash_password("secret"),
            )
        )
        db.commit()

    client = TestClient(app)
    headers = auth(client, "DR002")

    assert client.get("/v1/attention-events", headers=headers).json() == {"events": []}
    assert client.get(
        "/v1/attention-events?subject_id=patient-a",
        headers=headers,
    ).status_code == 403
    assert client.get(
        "/v1/attention-events/evt_test",
        headers=headers,
    ).status_code == 403
    assert client.post(
        "/v1/attention-events/evt_test/acknowledge",
        headers=headers,
        json={},
    ).status_code == 403
    assert client.post(
        "/v1/attention-events/evt_test/resolve",
        headers=headers,
        json={},
    ).status_code == 403

    with SessionLocal() as db:
        assert db.get(AttentionEvent, "evt_test").status == "OPEN"


def test_attention_lifecycle_persists_and_conflicting_clients_lose():
    seed()
    with SessionLocal() as db:
        db.add(
            Clinician(
                clinician_id="DR002",
                display_name="Dr Y",
                role="clinician",
                password_hash=hash_password("secret"),
            )
        )
        db.add(
            ClinicianSubjectAssignment(
                clinician_id="DR002",
                subject_id="patient-a",
            )
        )
        db.commit()

    first_client = TestClient(app)
    second_client = TestClient(app)
    first_headers = auth(first_client, "DR001")
    second_headers = auth(second_client, "DR002")

    acknowledged = first_client.post(
        "/v1/attention-events/evt_test/acknowledge",
        headers=first_headers,
        json={},
    )
    conflict = second_client.post(
        "/v1/attention-events/evt_test/acknowledge",
        headers=second_headers,
        json={},
    )

    assert acknowledged.status_code == 200
    assert acknowledged.json()["event"]["acknowledged_by"] == "DR001"
    assert conflict.status_code == 409

    with SessionLocal() as fresh_db_session:
        persisted = fresh_db_session.get(AttentionEvent, "evt_test")
        assert persisted.status == "ACKNOWLEDGED"
        assert persisted.acknowledged_by == "DR001"
        assert persisted.acknowledged_at is not None

    resolved = second_client.post(
        "/v1/attention-events/evt_test/resolve",
        headers=second_headers,
        json={},
    )
    stale_resolve = first_client.post(
        "/v1/attention-events/evt_test/resolve",
        headers=first_headers,
        json={},
    )

    assert resolved.status_code == 200
    assert resolved.json()["event"]["resolved_by"] == "DR002"
    assert stale_resolve.status_code == 409

    with SessionLocal() as fresh_db_session:
        persisted = fresh_db_session.get(AttentionEvent, "evt_test")
        assert persisted.status == "RESOLVED"
        assert persisted.resolved_by == "DR002"
        assert persisted.resolved_at is not None


def test_confirmed_forecast_episode_creates_one_attention_event():
    seed()
    now = utcnow()

    with SessionLocal() as db:
        db.add(FusionResult(subject_id="patient-b", composite=.52, tier="Medium", band="AMBER"))
        db.commit()

    def reading(db, captured=None):
        captured = captured or utcnow()
        row = ModalityReading(
            subject_id="patient-b",
            modality="c1_physiological",
            raw_score=.50,
            status="ok",
            confidence=.80,
            coverage=1.0,
            captured_at=captured,
            model_version="c1-test",
            detail={"response": {"risk_forecast": [.50] * 9 + [.80],
                                 "latest_reading_at": captured.isoformat()}},
        )
        db.add(row)
        db.flush()
        return row

    with SessionLocal() as db:
        first_forecast, first_event = persist_c1_forecast_and_event(
            db,
            "patient-b",
            reading(db, now - dt.timedelta(seconds=30)),
        )
        first_forecast.generated_at = now - dt.timedelta(seconds=30)
        db.commit()
        assert first_event is None

        _, confirmed_event = persist_c1_forecast_and_event(
            db,
            "patient-b",
            reading(db),
        )
        _, duplicate_event = persist_c1_forecast_and_event(
            db,
            "patient-b",
            reading(db),
        )

        episodes = db.scalars(
            select(EscalationEpisode).where(
                EscalationEpisode.subject_id == "patient-b"
            )
        ).all()
        events = db.scalars(
            select(AttentionEvent).where(AttentionEvent.subject_id == "patient-b")
        ).all()
    assert confirmed_event is not None
    assert duplicate_event is None
    assert len(episodes) == 1
    assert len(events) == 1
    assert events[0].episode_id == episodes[0].episode_id
    assert events[0].fusion_result_id is not None


def test_replayed_c1_window_cannot_confirm_an_episode():
    seed()
    now = utcnow()
    with SessionLocal() as db:
        db.add(FusionResult(subject_id="patient-b", composite=.52, tier="Medium", band="AMBER"))
        db.commit()
        captured = now - dt.timedelta(seconds=31)
        def send():
            row = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                raw_score=.50, status="ok", captured_at=captured,
                detail={"response": {"risk_forecast": [.50] * 9 + [.80],
                                     "latest_reading_at": captured.isoformat()}})
            db.add(row); db.flush()
            return persist_c1_forecast_and_event(db, "patient-b", row)
        first, _ = send()
        first.generated_at = now - dt.timedelta(seconds=30)
        db.commit()
        _, event = send()
        assert event is None
        assert db.scalars(select(AttentionEvent).where(
            AttentionEvent.subject_id == "patient-b")).all() == []


def test_unparseable_c1_source_timestamps_never_confirm_escalation():
    seed()
    now = utcnow()
    with SessionLocal() as db:
        db.add(FusionResult(subject_id="patient-b", composite=.52, tier="Medium", band="AMBER"))
        db.commit()
        for captured in (now-dt.timedelta(seconds=30), now):
            row = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                raw_score=.50, status="ok", captured_at=captured,
                detail={"response": {"risk_forecast": [.50] * 9 + [.80],
                                     "latest_reading_at": "not-a-timestamp"}})
            db.add(row); db.flush()
            forecast, event = persist_c1_forecast_and_event(db, "patient-b", row)
            if captured < now:
                forecast.generated_at = now-dt.timedelta(seconds=30)
                db.commit()
            assert event is None


@pytest.mark.parametrize("older_seconds,newer_seconds", [(220, 0), (30, -30)])
def test_source_windows_outside_confirmation_interval_never_alert(older_seconds, newer_seconds):
    seed()
    now = utcnow()
    with SessionLocal() as db:
        db.add(FusionResult(subject_id="patient-b", composite=.52, tier="Medium", band="AMBER"))
        db.commit()
        for captured, is_first in ((now-dt.timedelta(seconds=older_seconds), True),
                                   (now-dt.timedelta(seconds=newer_seconds), False)):
            row = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                raw_score=.50, status="ok", captured_at=captured,
                detail={"response": {"risk_forecast": [.50] * 9 + [.80],
                                     "latest_reading_at": captured.isoformat()}})
            db.add(row); db.flush()
            if is_first:
                db.add(ForecastResult(forecast_result_id="fcst_prior", subject_id="patient-b",
                    source_reading_id=row.id, scope="physiological", horizon_minutes=10,
                    score=.80, tier="High", escalation_predicted=True,
                    generated_at=now-dt.timedelta(seconds=30),
                    valid_until=now+dt.timedelta(minutes=9)))
                db.commit()
            else:
                _, event = persist_c1_forecast_and_event(db, "patient-b", row)
                assert event is None


def test_recovery_closes_episode_even_if_new_forecast_unavailable():
    seed()
    with SessionLocal() as db:
        db.add(EscalationEpisode(episode_id="ep_recover", subject_id="patient-b", status="active"))
        db.commit()
        row = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                              raw_score=.20, status="ok", captured_at=utcnow(), detail={"response": {}})
        db.add(row); db.flush()
        forecast, event = persist_c1_forecast_and_event(db, "patient-b", row)
        assert forecast is None and event is None
        assert db.get(EscalationEpisode, "ep_recover").status == "closed"


def test_forecast_requires_ten_valid_c1_points_and_never_makes_a_pseudo_forecast():
    seed()
    with SessionLocal() as db:
        for points in (None, [], [.80], [.50] * 9 + [float("nan")], [.50] * 9 + [150]):
            reading = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                raw_score=.50, status="ok", detail={"response": {"risk_forecast": points}})
            db.add(reading); db.flush()
            forecast, event = persist_c1_forecast_and_event(db, "patient-b", reading)
            assert forecast is None and event is None
        assert db.scalars(select(ForecastResult).where(ForecastResult.subject_id == "patient-b")).all() == []


def test_escalation_without_fusion_result_waits_for_authoritative_assessment():
    seed()
    with SessionLocal() as db:
        for number in range(2):
            reading = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                raw_score=.50, status="ok", detail={"response": {"risk_forecast": [.50] * 9 + [.80]}})
            db.add(reading); db.flush()
            forecast, event = persist_c1_forecast_and_event(db, "patient-b", reading)
            assert forecast is not None and event is None
            if number == 0:
                forecast.generated_at = utcnow() - dt.timedelta(seconds=30)
                db.commit()
        assert db.scalars(select(AttentionEvent).where(AttentionEvent.subject_id == "patient-b")).all() == []


def test_recovery_closes_episode_and_cooldown_prevents_immediate_repeat_event():
    seed()
    with SessionLocal() as db:
        db.add(FusionResult(subject_id="patient-b", composite=.52, tier="Medium", band="AMBER"))
        db.commit()

        def send(current, future, captured=None):
            captured = captured or utcnow()
            reading = ModalityReading(subject_id="patient-b", modality="c1_physiological",
                raw_score=current, status="ok", captured_at=captured,
                detail={"response": {"risk_forecast": [current] * 9 + [future],
                                     "latest_reading_at": captured.isoformat()}})
            db.add(reading); db.flush()
            return persist_c1_forecast_and_event(db, "patient-b", reading)

        first, _ = send(.50, .80, utcnow()-dt.timedelta(seconds=30))
        first.generated_at = utcnow() - dt.timedelta(seconds=30)
        db.commit()
        _, event = send(.50, .80)
        assert event is not None
        send(.20, .20)
        assert db.scalar(select(EscalationEpisode).where(
            EscalationEpisode.subject_id == "patient-b")).status == "closed"
        _, repeated = send(.50, .80)
        assert repeated is None
        assert len(db.scalars(select(AttentionEvent).where(
            AttentionEvent.subject_id == "patient-b")).all()) == 1


def test_assessment_history_uses_assessment_time_for_modality_freshness():
    seed()
    assessed_at = utcnow() - dt.timedelta(days=1)
    with SessionLocal() as db:
        db.add(
            ClinicianSubjectAssignment(
                clinician_id="DR001",
                subject_id="patient-b",
            )
        )
        db.add(
            ModalityReading(
                subject_id="patient-b",
                modality="c1_physiological",
                raw_score=.62,
                status="ok",
                confidence=.80,
                coverage=1.0,
                captured_at=assessed_at - dt.timedelta(minutes=1),
                model_version="c1-history",
            )
        )
        historical = FusionResult(
            subject_id="patient-b",
            composite=.62,
            tier="Medium",
            band="AMBER",
            confidence=.80,
            modalities_used=1,
            weights={"c1_physiological": 1.0},
            contributions={"c1_physiological": .62},
            harmonisation={"assessment": {"status": "provisional"}},
            model_version="ragf-history",
            computed_at=assessed_at,
        )
        db.add(historical)
        db.commit()
        historical_id = historical.id

    client = TestClient(app)
    response = client.get(
        "/v1/patients/patient-b/assessments",
        headers=auth(client),
    )

    assert response.status_code == 200
    history = response.json()["assessments"]
    assessment = next(
        item for item in history if item["fusion_result_id"] == historical_id
    )
    c1 = next(
        modality
        for modality in assessment["modalities"]
        if modality["component_id"] == "c1_physiological"
    )
    assert c1["status"] == "ok"
    assert c1["available"] is True
    assert c1["included_in_fusion"] is True


def test_assessment_history_does_not_attach_newer_forecast_to_older_fusion():
    seed()
    with SessionLocal() as db:
        older = db.scalar(select(FusionResult).where(FusionResult.subject_id == "patient-a"))
        now = utcnow()
        older.computed_at = now - dt.timedelta(seconds=10)
        db.add(FusionResult(subject_id="patient-a", composite=.67, tier="Medium", band="AMBER",
                            confidence=.81, computed_at=now - dt.timedelta(seconds=5)))
        db.commit()
        newer = db.scalar(select(FusionResult).where(FusionResult.subject_id == "patient-a")
                          .order_by(FusionResult.id.desc()))
        forecast = db.get(ForecastResult, "fcst_test")
        forecast.generated_at = now - dt.timedelta(seconds=12)
        forecast.source_fusion_result_id = newer.id
        db.commit()
        newer_id, older_id = newer.id, older.id
    client = TestClient(app)
    response = client.get("/v1/patients/patient-a/assessments", headers=auth(client))
    assert response.status_code == 200
    history = {row["fusion_result_id"]: row for row in response.json()["assessments"]}
    assert history[older_id]["forecast"] is None
    assert history[newer_id]["forecast"]["forecast_result_id"] == "fcst_test"


def test_history_keeps_original_reading_and_forecast_when_later_poll_arrives():
    seed()
    with SessionLocal() as db:
        original = ModalityReading(subject_id="patient-a", modality="c1_physiological",
                                   raw_score=.53, status="ok", captured_at=utcnow()-dt.timedelta(seconds=1))
        db.add(original); db.flush()
        fusion = db.scalar(select(FusionResult).where(FusionResult.subject_id == "patient-a"))
        fusion.weights = {"c1_physiological": 1.0}
        fusion.harmonisation = {"assessment": {"status": "complete"},
                                "source_reading_ids": {"c1_physiological": original.id}}
        first = db.get(ForecastResult, "fcst_test")
        first.source_fusion_result_id = fusion.id
        first.source_reading_id = original.id
        first.generated_at = utcnow() - dt.timedelta(seconds=20)
        db.commit()
        backdated = ModalityReading(subject_id="patient-a", modality="c1_physiological",
                                    raw_score=.99, status="ok", captured_at=original.captured_at)
        db.add(backdated); db.flush()
        db.add(ForecastResult(forecast_result_id="fcst_later", subject_id="patient-a",
            source_fusion_result_id=fusion.id, source_reading_id=backdated.id,
            scope="physiological", horizon_minutes=10, score=.99, tier="High",
            escalation_predicted=True, generated_at=utcnow(), valid_until=utcnow()+dt.timedelta(minutes=10)))
        db.commit()
        fusion_id = fusion.id
    client = TestClient(app)
    result = client.get("/v1/patients/patient-a/assessments", headers=auth(client))
    assert result.status_code == 200
    row = next(item for item in result.json()["assessments"] if item["fusion_result_id"] == fusion_id)
    c1 = next(item for item in row["modalities"] if item["component_id"] == "c1_physiological")
    assert c1["score"] == .53
    assert c1["included_in_fusion"] is True
    assert row["forecast"]["forecast_result_id"] == "fcst_test"


def test_assessment_history_exposes_assignment_scoped_attention_markers():
    seed(); client = TestClient(app); headers = auth(client)
    response = client.get("/v1/patients/patient-a/assessments", headers=headers)
    assert response.status_code == 200
    assert response.json()["events"][0]["id"] == "evt_test"
    assert response.json()["events"][0]["fusion_result_id"] == response.json()["assessments"][0]["fusion_result_id"]
    assert client.get("/v1/patients/patient-b/assessments", headers=headers).status_code == 403


def test_patient_and_clinician_views_share_authoritative_fusion_identity():
    seed()
    client = TestClient(app)
    headers = auth(client)

    patient = client.get(
        "/v1/patients/patient-a/risk",
        headers=patient_auth("patient-a"),
    )
    clinician = client.get(
        "/v1/patients/patient-a/assessment/latest",
        headers=headers,
    )

    assert patient.status_code == 200
    assert clinician.status_code == 200
    patient_body = patient.json()
    clinician_body = clinician.json()

    assert patient_body["fusion_result_id"] == clinician_body["fusion_result_id"]
    assert patient_body["composite"] == clinician_body["current_assessment"]["score"]
    assert patient_body["fusion_result_id"] is not None


def test_patient_and_clinician_views_share_unavailable_state():
    seed()
    with SessionLocal() as db:
        db.add(
            ClinicianSubjectAssignment(
                clinician_id="DR001",
                subject_id="patient-b",
            )
        )
        db.commit()

    client = TestClient(app)
    headers = auth(client)

    patient = client.get(
        "/v1/patients/patient-b/risk",
        headers=patient_auth("patient-b"),
    )
    clinician = client.get(
        "/v1/patients/patient-b/assessment/latest",
        headers=headers,
    )

    assert patient.status_code == 200
    assert clinician.status_code == 200
    patient_body = patient.json()
    clinician_body = clinician.json()

    assert patient_body["fusion_result_id"] is None
    assert clinician_body["fusion_result_id"] is None
    assert patient_body["composite"] is None
    assert clinician_body["current_assessment"] is None
    assert patient_body["band"] == "GREY"
    assert clinician_body["assessment_status"] == "unavailable"
