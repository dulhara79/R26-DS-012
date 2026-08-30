import os, sys, datetime as dt
os.environ.setdefault("BACKEND_API_TOKEN", "")
os.environ.setdefault("MRN_PEPPER", "seed")

from db_models import Subject, ModalityReading
from main import run_fusion, get_session
from sqlalchemy import select

db = next(get_session())
subj = db.scalar(select(Subject).limit(1))
if not subj:
    sys.exit("No subjects in DB. Open a patient chart in the app first.")

sid = subj.subject_id
print(f"Subject: {sid}\n")

# Values taken from the ACTUAL quartiles of each reference distribution.
# Pass 1 at Q1 (~25th percentile) -> should be low-mid band.
# Pass 2 at Q3 (~75th percentile) -> should be higher band.
# This gives the dashboard a real trajectory (worsening).
PASSES = [
    {"label": "3 days ago", "days_ago": 3, "scores": {
        "c1_physiological": 0.058066,   # Q1 of ref (max=0.529)
        "c3_clinical_nlp":  0.314173,   # Q1 of ref (max=0.981)
        "c4_demographic":   0.035176,   # Q1 of ref (max=0.451)
    }},
    {"label": "today", "days_ago": 0, "scores": {
        "c1_physiological": 0.161367,   # Q3
        "c3_clinical_nlp":  0.638600,   # Q3
        "c4_demographic":   0.134885,   # Q3
    }},
]

CONF = {"c1_physiological": 0.80, "c3_clinical_nlp": 0.75, "c4_demographic": 0.70}
COV  = {"c1_physiological": 0.95, "c3_clinical_nlp": 1.00, "c4_demographic": 1.00}

for i, p in enumerate(PASSES, 1):
    when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=p["days_ago"])
    print(f"Pass {i}  ({p['label']})")
    for mod, score in p["scores"].items():
        db.add(ModalityReading(
            subject_id=sid, modality=mod, raw_score=score,
            status="ok", confidence=CONF[mod], coverage=COV[mod],
            captured_at=when, model_version="SEEDED-FIXTURE",
            detail={"note": "seeded fixture - not a model output"},
        ))
        print(f"    {mod:20s} raw={score:.6f}")
    db.commit()
    try:
        row = run_fusion(db, sid, f"seed-pass-{i}")
        db.commit()
        print(f"    -> composite={row.composite}  tier={row.tier}  band={row.band}")
    except Exception as exc:
        print(f"    -> FUSION FAILED: {exc}")
    print()

print("Done. Two passes seeded at Q1 and Q3 of the reference distributions.")
