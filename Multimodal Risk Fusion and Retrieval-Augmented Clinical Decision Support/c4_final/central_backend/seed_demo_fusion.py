#!/usr/bin/env python3
"""
seed_demo_fusion.py — put REAL fusion results in the database without
depending on C1, C3 or C4 being reachable.

WHY THIS EXISTS
---------------
Every modality in this system is an external HTTP call to a Hugging Face
Space. If any Space is asleep, rate-limited, or rejecting credentials, no
composite can be produced and every downstream screen sits empty. That
blocks demonstrating the fusion engine, the dashboard, and the chart —
none of which are actually broken.

This script writes modality_readings rows directly, then calls the real
run_fusion() from main.py. The fusion, gate, calibration, banding and
trend are all genuinely computed by your own code. Only the component
scores are supplied by hand instead of fetched over the network.

WHAT THIS IS NOT
----------------
This is a DEVELOPMENT AND DEMONSTRATION FIXTURE. The scores it inserts are
made up. Do not present a composite produced from these rows as a model
result, do not put a screenshot of it in your thesis as evidence of
performance, and do not leave these rows in the database when you record
real measurements. Every seeded reading is tagged
model_version='SEEDED-FIXTURE' so you can find and delete them later:

    DELETE FROM modality_readings WHERE model_version='SEEDED-FIXTURE';

Run from the central_backend folder with the backend STOPPED:
    ./.venv/bin/python3 seed_demo_fusion.py DEMO-001
"""

import sys, os, datetime as dt

if not os.path.exists("main.py"):
    sys.exit("Run this from the central_backend folder (no main.py here).")

os.environ.setdefault("BACKEND_API_TOKEN", "")
os.environ.setdefault("MRN_PEPPER", "seed")

from db_models import Subject, SubjectAlias, ModalityReading   # noqa: E402
from main import run_fusion, get_session                        # noqa: E402
from sqlalchemy import select                                   # noqa: E402

MRN = sys.argv[1] if len(sys.argv) > 1 else "DEMO-001"
TAG = "SEEDED-FIXTURE"

# Two assessments so the dashboard's trajectory tiles have something to
# compare. First pass is milder, second is worse -> "Worsening".
PASSES = [
    {"days_ago": 3, "c1": 0.42, "c3": 0.51, "c4": 0.38},
    {"days_ago": 0, "c1": 0.71, "c3": 0.83, "c4": 0.38},
]

db = next(get_session())

# ── find or create the subject ───────────────────────────────────────────────
alias = db.scalar(select(SubjectAlias).where(SubjectAlias.alias_value.like(f"%{MRN}%")))
if alias:
    subject_id = alias.subject_id
    print(f"Found existing subject for {MRN}: {subject_id}")
else:
    subj = db.scalar(select(Subject).limit(1))
    if subj:
        subject_id = subj.subject_id
        print(f"No alias matched {MRN}; using first subject in DB: {subject_id}")
    else:
        sys.exit(
            f"No subjects in the database at all.\n"
            f"Open the patient's chart in the doctor app once first — that\n"
            f"enrols them — then re-run this script."
        )

print()

for i, p in enumerate(PASSES, 1):
    when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=p["days_ago"])
    print(f"Pass {i}  ({p['days_ago']} days ago)")

    for modality, score, conf, cov in [
        ("c1_physiological", p["c1"], 0.80, 0.95),
        ("c3_clinical_nlp",  p["c3"], 0.75, 1.00),
        ("c4_demographic",   p["c4"], 0.70, 1.00),
    ]:
        db.add(ModalityReading(
            subject_id=subject_id,
            modality=modality,
            raw_score=score,
            status="ok",
            confidence=conf,
            coverage=cov,
            captured_at=when,          # datetime object, NOT an ISO string
            model_version=TAG,
            detail={"note": "seeded development fixture - not a model output"},
        ))
        print(f"    {modality:20s} raw_score={score}")

    db.commit()

    try:
        row = run_fusion(db, subject_id, f"seed-pass-{i}")
        db.commit()
        print(f"    -> composite={row.composite}  tier={row.tier}  band={row.band}")
        if row.reason:
            print(f"       reason: {row.reason}")
    except Exception as exc:
        print(f"    -> FUSION FAILED: {type(exc).__name__}: {exc}")
        print("       (the readings are stored; the fusion step is what failed)")
    print()

print("=" * 68)
print("Done. Restart the backend, then in the doctor app open the patient")
print("chart once (this pulls the timeline), and open the Dashboard.")
print()
print("These rows are tagged SEEDED-FIXTURE. Remove them before recording")
print("any real result:")
print("    DELETE FROM modality_readings WHERE model_version='SEEDED-FIXTURE';")
print("=" * 68)
