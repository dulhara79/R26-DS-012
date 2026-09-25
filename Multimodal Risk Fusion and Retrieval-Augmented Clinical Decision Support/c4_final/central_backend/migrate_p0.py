"""Create the additive ClinAnx P0 tables. Safe to run repeatedly."""
from db_models import Base, engine

P0_TABLES = ["clinicians", "clinician_subject_assignments", "forecast_results",
             "escalation_episodes", "attention_events"]

if __name__ == "__main__":
    Base.metadata.create_all(engine, tables=[Base.metadata.tables[name] for name in P0_TABLES])
    print("created/verified: " + ", ".join(P0_TABLES))
