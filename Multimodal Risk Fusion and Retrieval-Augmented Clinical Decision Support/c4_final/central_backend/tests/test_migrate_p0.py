"""An already deployed database can be upgraded without dropping clinical rows."""

from sqlalchemy import create_engine, inspect, text

from migrate_p0 import run_migrations


def test_upgrade_existing_forecasts_is_versioned_idempotent_and_preserves_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'upgrade.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE subjects (subject_id VARCHAR(36) PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE modality_readings (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE fusion_results (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE forecast_results (forecast_result_id VARCHAR(48) PRIMARY KEY, subject_id VARCHAR(36))"))
        conn.execute(text("INSERT INTO subjects (subject_id) VALUES ('patient-a')"))
        conn.execute(text("INSERT INTO forecast_results (forecast_result_id, subject_id) VALUES ('fcst_old', 'patient-a')"))

    run_migrations(engine)
    run_migrations(engine)
    inspector = inspect(engine)
    assert "patient_credentials" in inspector.get_table_names()
    assert "clinician_assignment_invites" in inspector.get_table_names()
    assert "source_fusion_result_id" in {column["name"] for column in inspector.get_columns("forecast_results")}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT forecast_result_id FROM forecast_results")).scalar_one() == "fcst_old"
        assert conn.execute(text("SELECT count(*) FROM schema_migrations")).scalar_one() == 2


def test_clean_database_bootstraps_core_schema_before_p0_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'clean.db'}")
    run_migrations(engine)
    tables = set(inspect(engine).get_table_names())
    assert {"subjects", "subject_aliases", "modality_readings", "fusion_results",
            "clinicians", "clinician_subject_assignments", "patient_credentials",
            "forecast_results", "escalation_episodes", "attention_events",
            "clinician_assignment_invites"} <= tables
