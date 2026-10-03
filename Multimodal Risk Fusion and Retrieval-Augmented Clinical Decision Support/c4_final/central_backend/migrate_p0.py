"""Additive, repeatable P0 upgrades for an existing Central Backend database.

Run before deploying new application code. Never deletes clinical rows.
"""

from __future__ import annotations

from sqlalchemy import Column, MetaData, String, Table, inspect, insert, select, text

from db_models import Base, engine

_migration_metadata = MetaData()
_revisions = Table("schema_migrations", _migration_metadata,
                   Column("revision", String(96), primary_key=True))

P0_TABLES = ("clinicians", "clinician_subject_assignments", "patient_credentials",
             "forecast_results", "escalation_episodes", "attention_events")
CORE_TABLES = ("subjects", "subject_aliases", "pairing_codes", "modality_readings",
               "fusion_results", "verdicts", "audit_log", "support_bank_notes")


def run_migrations(bind=engine) -> None:
    """Upgrade an existing DB in revision order, preserving all patient data."""
    with bind.begin() as conn:
        _migration_metadata.create_all(conn)
    with bind.begin() as conn:
        applied = set(conn.scalars(select(_revisions)).all())
        if "0001_p0_tables" not in applied:
            Base.metadata.create_all(conn, tables=[Base.metadata.tables[name]
                                                   for name in CORE_TABLES + P0_TABLES])
            conn.execute(insert(_revisions).values(revision="0001_p0_tables"))

    with bind.begin() as conn:
        applied = set(conn.scalars(select(_revisions)).all())
        if "0002_assignment_invites_and_forecast_link" not in applied:
            Base.metadata.create_all(conn, tables=[Base.metadata.tables["clinician_assignment_invites"]])
            if "source_fusion_result_id" not in {
                column["name"] for column in inspect(conn).get_columns("forecast_results")
            }:
                conn.execute(text("ALTER TABLE forecast_results ADD COLUMN source_fusion_result_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_forecast_results_source_fusion_result_id "
                              "ON forecast_results (source_fusion_result_id)"))
            if conn.dialect.name == "postgresql" and not any(
                fk.get("referred_table") == "fusion_results" and
                "source_fusion_result_id" in fk.get("constrained_columns", [])
                for fk in inspect(conn).get_foreign_keys("forecast_results")
            ):
                conn.execute(text("ALTER TABLE forecast_results ADD CONSTRAINT "
                                  "fk_forecast_source_fusion_result "
                                  "FOREIGN KEY (source_fusion_result_id) REFERENCES fusion_results (id)"))
            conn.execute(insert(_revisions).values(revision="0002_assignment_invites_and_forecast_link"))


if __name__ == "__main__":
    run_migrations()
    print("Central Backend P0 schema revisions applied")
