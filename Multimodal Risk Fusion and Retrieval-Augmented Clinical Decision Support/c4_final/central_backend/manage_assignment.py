import argparse
from sqlalchemy import select
from db_models import ClinicianSubjectAssignment, SessionLocal, init_db, utcnow

parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["assign", "unassign"])
parser.add_argument("clinician_id"); parser.add_argument("subject_id"); args = parser.parse_args(); init_db()
with SessionLocal() as db:
    row = db.scalar(select(ClinicianSubjectAssignment).where(
        ClinicianSubjectAssignment.clinician_id == args.clinician_id,
        ClinicianSubjectAssignment.subject_id == args.subject_id))
    if row is None:
        if args.action == "unassign": raise SystemExit("assignment not found")
        row = ClinicianSubjectAssignment(clinician_id=args.clinician_id, subject_id=args.subject_id)
        db.add(row)
    else:
        row.active = args.action == "assign"; row.ended_at = None if row.active else utcnow()
    db.commit()
print(f"{args.action}ed {args.subject_id} for {args.clinician_id}")
