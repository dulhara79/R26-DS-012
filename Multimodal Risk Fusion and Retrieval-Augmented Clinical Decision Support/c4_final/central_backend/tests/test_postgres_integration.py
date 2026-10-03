"""Deployment-engine gates. Run only against a disposable PostgreSQL test DB.

CI supplies POSTGRES_TEST_URL; local SQLite runs skip this file. Every test
owns and drops a random schema, leaving other schemas and databases untouched.
"""

import datetime as dt
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker

from clinician_api import hash_password
from db_models import (
    AttentionEvent, Base, Clinician, ClinicianSubjectAssignment,
    EscalationEpisode, ForecastResult, FusionResult, ModalityReading,
    Subject, get_session, utcnow,
)
from forecast import persist_c1_forecast_and_event
from main import app
from migrate_p0 import run_migrations


@pytest.fixture
def pg_database():
    value = os.getenv("POSTGRES_TEST_URL")
    if not value:
        pytest.skip("POSTGRES_TEST_URL is set only by the PostgreSQL CI job")
    url = make_url(value)
    if not url.drivername.startswith("postgresql") or url.database != "r26_test":
        pytest.fail("POSTGRES_TEST_URL must target the disposable r26_test database")
    schema = f"r26_check_{uuid.uuid4().hex}"
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        yield engine, schema, url
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _client_for(engine):
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def session_override():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_session] = session_override
    return TestClient(app), sessions


def _clinician(client, clinician_id):
    response = client.post("/auth/login", json={
        "clinician_id": clinician_id, "password": "synthetic-password"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _seed_event(sessions):
    now = utcnow()
    with sessions() as db:
        db.add(Subject(subject_id="subject-postgres"))
        for clinician_id in ("DR001", "DR002"):
            db.add(Clinician(clinician_id=clinician_id, display_name=clinician_id,
                             password_hash=hash_password("synthetic-password")))
        db.flush()
        for clinician_id in ("DR001", "DR002"):
            db.add(ClinicianSubjectAssignment(clinician_id=clinician_id,
                                              subject_id="subject-postgres"))
        db.flush()
        fusion = FusionResult(subject_id="subject-postgres", composite=.58,
                              tier="Medium", band="AMBER")
        db.add(fusion)
        db.flush()
        db.add(ForecastResult(forecast_result_id="fcst_postgres", subject_id="subject-postgres",
                              scope="physiological", horizon_minutes=10, score=.84,
                              tier="High", escalation_predicted=True,
                              generated_at=now, valid_until=now + dt.timedelta(minutes=10)))
        db.add(EscalationEpisode(episode_id="ep_postgres", subject_id="subject-postgres"))
        db.flush()
        db.add(AttentionEvent(id="evt_postgres", subject_id="subject-postgres",
                              fusion_result_id=fusion.id, forecast_result_id="fcst_postgres",
                              episode_id="ep_postgres", severity="high",
                              reason="Synthetic policy confirmation", forecast_horizon=10))
        db.commit()


def test_postgres_v1_upgrade_preserves_rows_and_adds_real_fk(pg_database):
    engine, _, _ = pg_database
    from migrate_p0 import CORE_TABLES, P0_TABLES

    Base.metadata.create_all(engine, tables=[Base.metadata.tables[name] for name in CORE_TABLES])
    # v1 persisted forecasts without the v2 source assessment column.
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE forecast_results (
            forecast_result_id varchar(48) PRIMARY KEY,
            subject_id varchar(36) REFERENCES subjects(subject_id),
            source_reading_id integer REFERENCES modality_readings(id),
            scope varchar(32) NOT NULL, horizon_minutes integer NOT NULL,
            score double precision, tier varchar(16), escalation_probability double precision,
            escalation_predicted boolean NOT NULL, generated_at timestamptz NOT NULL,
            valid_until timestamptz NOT NULL, model_version varchar(64)
        )"""))
        conn.execute(text("CREATE TABLE schema_migrations (revision varchar(96) PRIMARY KEY)"))
        conn.execute(text("INSERT INTO schema_migrations VALUES ('0001_p0_tables')"))
        conn.execute(text("INSERT INTO subjects (subject_id, created_at, status) "
                          "VALUES ('legacy-subject', now(), 'active')"))
        conn.execute(text("INSERT INTO fusion_results (subject_id, confidence, modalities_used, renormalised, computed_at) "
                          "VALUES ('legacy-subject', 0, 0, false, now())"))
        conn.execute(text("""INSERT INTO forecast_results
            (forecast_result_id, subject_id, scope, horizon_minutes, escalation_predicted,
             generated_at, valid_until) VALUES
            ('fcst_legacy', 'legacy-subject', 'physiological', 10, false, now(), now())"""))
    Base.metadata.create_all(engine, tables=[Base.metadata.tables[name] for name in P0_TABLES
                                             if name != "forecast_results"])

    run_migrations(engine)
    run_migrations(engine)
    inspector = inspect(engine)
    assert "clinician_assignment_invites" in inspector.get_table_names()
    assert "source_fusion_result_id" in {c["name"] for c in inspector.get_columns("forecast_results")}
    assert any(fk["referred_table"] == "fusion_results" and
               "source_fusion_result_id" in fk["constrained_columns"]
               for fk in inspector.get_foreign_keys("forecast_results"))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT forecast_result_id FROM forecast_results")).scalar_one() == "fcst_legacy"
        assert conn.execute(text("SELECT count(*) FROM schema_migrations")).scalar_one() == 2


def test_postgres_concurrent_lifecycle_and_episode_deduplication(pg_database):
    engine, _, _ = pg_database
    run_migrations(engine)
    client, sessions = _client_for(engine)
    try:
        _seed_event(sessions)
        clinicians = [_clinician(client, clinician_id) for clinician_id in ("DR001", "DR002")]

        def race(path):
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(lambda headers: client.post(path, headers=headers, json={}),
                                     clinicians))

        ack = race("/v1/attention-events/evt_postgres/acknowledge")
        assert sorted(response.status_code for response in ack) == [200, 409]
        winner = next(response.json()["event"] for response in ack if response.status_code == 200)
        assert winner["acknowledged_by"] in {"DR001", "DR002"}
        resolve = race("/v1/attention-events/evt_postgres/resolve")
        assert sorted(response.status_code for response in resolve) == [200, 409]
        resolved = next(response.json()["event"] for response in resolve if response.status_code == 200)
        assert resolved["resolved_by"] in {"DR001", "DR002"}
        assert resolved["resolved_at"].endswith("Z")
        detail = client.get("/v1/attention-events/evt_postgres", headers=clinicians[0])
        assert detail.status_code == 200 and detail.json()["event"] == resolved

        now = utcnow()
        with sessions() as db:
            prior = ModalityReading(subject_id="subject-postgres", modality="c1_physiological",
                raw_score=.50, status="ok", captured_at=now-dt.timedelta(seconds=30),
                detail={"response": {"risk_forecast": [.50] * 9 + [.84],
                                     "latest_reading_at": (now-dt.timedelta(seconds=30)).isoformat()}})
            db.add(prior)
            db.flush()
            db.add(ForecastResult(forecast_result_id="fcst_prior", subject_id="subject-postgres",
                source_reading_id=prior.id, scope="physiological", horizon_minutes=10,
                score=.84, tier="High", escalation_predicted=True,
                generated_at=now-dt.timedelta(seconds=30),
                valid_until=now+dt.timedelta(minutes=9)))
            # Close the separate seeded episode before racing policy creation.
            episode = db.get(EscalationEpisode, "ep_postgres")
            episode.status = "closed"
            episode.closed_at = now-dt.timedelta(minutes=11)
            db.commit()

        def forecast_worker(_):
            with sessions() as db:
                captured = utcnow()
                reading = ModalityReading(subject_id="subject-postgres", modality="c1_physiological",
                    raw_score=.50, status="ok", captured_at=captured,
                    detail={"response": {"risk_forecast": [.50] * 9 + [.84],
                                         "latest_reading_at": captured.isoformat()}})
                db.add(reading)
                db.flush()
                _, event = persist_c1_forecast_and_event(db, "subject-postgres", reading)
                return event.id if event else None

        with ThreadPoolExecutor(max_workers=2) as pool:
            created = list(pool.map(forecast_worker, range(2)))
        assert sum(item is not None for item in created) == 1
        with sessions() as db:
            assert len(db.scalars(select(AttentionEvent)).all()) == 2
            assert len(db.scalars(select(EscalationEpisode).where(
                EscalationEpisode.status == "active")).all()) == 1
    finally:
        app.dependency_overrides.pop(get_session, None)


def test_postgres_dump_restore_preserves_event_lifecycle(pg_database, tmp_path):
    engine, schema, url = pg_database
    run_migrations(engine)
    _, sessions = _client_for(engine)
    try:
        _seed_event(sessions)
        with sessions() as db:
            event = db.get(AttentionEvent, "evt_postgres")
            event.status = "RESOLVED"
            event.acknowledged_at = utcnow()
            event.acknowledged_by = "DR001"
            event.resolved_at = utcnow()
            event.resolved_by = "DR002"
            db.commit()
        backup = tmp_path / "synthetic-p0.dump"
        restore_name = f"r26_restore_{uuid.uuid4().hex}"
        env = {**os.environ, "PGPASSWORD": url.password or ""}
        cli_url = URL.create("postgresql", username=url.username, host=url.host,
                             port=url.port, database=url.database, query=url.query)
        restore_url = cli_url.set(database=restore_name)
        admin = create_engine(url, isolation_level="AUTOCOMMIT")
        try:
            subprocess.run(["pg_dump", "--dbname", str(cli_url), "--schema", schema,
                            "--format", "custom", "--file", str(backup)], env=env, check=True)
            with admin.connect() as conn:
                conn.execute(text(f"CREATE DATABASE {restore_name}"))
            subprocess.run(["pg_restore", "--dbname", str(restore_url),
                            "--no-owner", "--no-privileges", "--exit-on-error",
                            str(backup)], env=env, check=True)
            restored = create_engine(url.set(database=restore_name),
                                     connect_args={"options": f"-csearch_path={schema}"})
            try:
                with restored.connect() as conn:
                    row = conn.execute(text("SELECT status, acknowledged_by, resolved_by, "
                                            "acknowledged_at, resolved_at FROM attention_events "
                                            "WHERE id = 'evt_postgres'")).one()
                    assert row.status == "RESOLVED"
                    assert row.acknowledged_by == "DR001" and row.resolved_by == "DR002"
                    assert row.acknowledged_at and row.resolved_at
                    assert conn.execute(text("SELECT count(*) FROM schema_migrations")).scalar_one() == 2
            finally:
                restored.dispose()
        finally:
            with admin.connect() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS {restore_name} WITH (FORCE)"))
            admin.dispose()
    finally:
        app.dependency_overrides.pop(get_session, None)
