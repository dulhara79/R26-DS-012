"""
apply_index_time.py — Stage 1b, between build_clean_cohort and tokenize_cohort.

    python -m scripts.apply_index_time \
        --cohort data/clean/cohort_psych_mimic4.csv \
        --patients data/clean/patients_psych_mimic4.csv \
        --policy at_or_before \
        --out data/clean/cohort_psych_mimic4_idx.csv

Additive by design: build_clean_cohort.py, tokenize_cohort.py, the sampler, the
model and every test are untouched. This stage reads a finished cohort CSV,
fixes an index time per patient from structured tables, drops notes outside the
admitted window, and writes a new cohort CSV in the identical schema.

IMPORTANT — the temporal field
------------------------------
tokenize_cohort.py reads the column `days_before_patient_last_note` and stores it
as the delta that drives w^T. Under an index protocol that column is the wrong
quantity, so this script OVERWRITES it with `days_before_index` and also keeps
both original and new values in dedicated columns:

    days_before_index                     the new, correct delta
    days_before_patient_last_note         := days_before_index   (what the model reads)
    days_before_last_note_preindex        recomputed within the admitted window
    days_before_patient_last_note_raw     the original pre-filter value

The substitution is recorded in the output JSON under `temporal_reference` so
the provenance is explicit rather than implied. If you would rather make this
visible in code, add a `--temporal-field` argument to tokenize_cohort.py and
point it at `days_before_index`; the one-line change is noted in the README.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn import indexing as IX          # noqa: E402
from tcwpn.io_paths import resolve, MIMIC4, MIMIC3   # noqa: E402


def load_structured(source: str, mimic4_path: str, mimic3_path: str):
    """Returns (diagnoses, admissions) with lowercase columns."""
    if source == "mimic4":
        diagnoses = pd.read_csv(
            resolve(mimic4_path, *MIMIC4["diagnoses"]),
            usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
        admissions = pd.read_csv(
            resolve(mimic4_path, *MIMIC4["admissions"]),
            usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
        return diagnoses, admissions

    diagnoses = pd.read_csv(resolve(mimic3_path, *MIMIC3["diagnoses"]),
                            usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
    diagnoses.columns = [c.lower() for c in diagnoses.columns]
    diagnoses = diagnoses.rename(columns={"icd9_code": "icd_code"})
    diagnoses["icd_version"] = 9
    admissions = pd.read_csv(resolve(mimic3_path, *MIMIC3["admissions"]),
                             usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"])
    admissions.columns = [c.lower() for c in admissions.columns]
    return diagnoses, admissions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--patients", required=True,
                    help="patients_<tag>.csv from build_clean_cohort.py")
    ap.add_argument("--policy", choices=list(IX.POLICIES), default="at_or_before")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["mimic4", "mimic3"], default="mimic4")
    ap.add_argument("--mimic4-path", default=os.getenv("MIMIC_IV_DATASET_PATH", ""))
    ap.add_argument("--mimic3-path", default=os.getenv("MIMIC_III_DATASET_PATH", ""))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-notes-per-patient", type=int, default=1)
    args = ap.parse_args()

    notes = pd.read_csv(args.cohort, low_memory=False)
    per_patient = pd.read_csv(args.patients)
    diagnoses, admissions = load_structured(
        args.source, args.mimic4_path, args.mimic3_path)

    print(f"input notes: {len(notes):,}  patients: {notes['subject_id'].nunique():,}")

    index_times = IX.compute_index_times(
        diagnoses, admissions, per_patient, seed=args.seed)
    print(f"index times fixed for {len(index_times):,} patients")
    print(index_times["index_rule"].value_counts().to_string())

    notes["days_before_patient_last_note_raw"] = notes.get(
        "days_before_patient_last_note")

    kept, report = IX.apply_index_policy(notes, index_times, args.policy)
    kept = IX.recompute_within_patient_features(kept)
    kept = kept.rename(
        columns={"days_before_patient_last_note": "days_before_last_note_preindex"})

    if args.min_notes_per_patient > 1:
        sizes = kept.groupby("subject_id")["note_id"].transform("count")
        kept = kept[sizes >= args.min_notes_per_patient]

    # The substitution tokenize_cohort.py depends on.
    kept["days_before_patient_last_note"] = kept["days_before_index"]
    report["temporal_reference"] = (
        "index_time" if args.policy != "none" else "index_dischtime_descriptive")
    report["model_temporal_field"] = "days_before_patient_last_note := days_before_index"

    # ---- guard rails --------------------------------------------------------
    assert (kept["days_before_index"] >= -1e-9).all(), "negative delta survived"
    for split in ("train", "val", "test"):
        sub = kept[kept["split"] == split]
        if sub.empty:
            print(f"  WARNING: split '{split}' is now empty")
    n_pat = kept.groupby(["split", "label"])["subject_id"].nunique()
    report["patients_per_split_after"] = {
        f"{s}_label{l}": int(v) for (s, l), v in n_pat.items()}

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(outp, index=False)
    (outp.parent / f"{outp.stem}_index_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    print("\n" + json.dumps(report, indent=2, default=str))
    print(f"\nwrote {outp}")

    drift = abs(report.get("differential_patient_retention_pp", 0.0))
    if drift > 10:
        print(
            f"\n  CAUTION: the policy retained cases and controls at rates "
            f"differing by {drift} percentage points. The surviving cohort is "
            f"not the cohort you defined; report this number in the paper and "
            f"consider whether the index rule for the control arm is fair."
        )
    print("\nNEXT: python -m scripts.audit_cohort --cohort "
          f"{outp}\n  then tokenize with --cohort {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
