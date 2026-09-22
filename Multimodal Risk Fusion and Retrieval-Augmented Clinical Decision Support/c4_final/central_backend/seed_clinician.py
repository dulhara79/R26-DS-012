import argparse
from db_models import Clinician, SessionLocal, init_db
from clinician_api import hash_password

parser = argparse.ArgumentParser()
parser.add_argument("clinician_id"); parser.add_argument("display_name"); parser.add_argument("password")
parser.add_argument("--role", default="clinician")
args = parser.parse_args()
init_db()
with SessionLocal() as db:
    row = db.get(Clinician, args.clinician_id)
    if row is None:
        row = Clinician(clinician_id=args.clinician_id, display_name=args.display_name,
                        role=args.role, password_hash=hash_password(args.password))
        db.add(row)
    else:
        row.display_name=args.display_name; row.role=args.role
        row.password_hash=hash_password(args.password); row.active=True
    db.commit()
print(f"seeded clinician {args.clinician_id}")
