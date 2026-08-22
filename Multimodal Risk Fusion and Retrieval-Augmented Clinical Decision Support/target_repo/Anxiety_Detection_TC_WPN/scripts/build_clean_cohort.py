"""
build_clean_cohort.py — Stage 1 of the publication-clean benchmark.

    python -m scripts.build_clean_cohort --out data/clean --arm psych
    python -m scripts.build_clean_cohort --out data/clean --arm clean
    python -m scripts.build_clean_cohort --out data/clean --arm psych --source mimic3

ORDER OF OPERATIONS (this order is the point)
=============================================
  1. age eligibility          <- structured
  2. cohort arms from ICD     <- structured
  3. patient-level split      <- hash of subject_id only
  4. THEN load note text
  5. drop notes that are too short / empty                 (label-independent)
  6. drop text duplicated across splits                    (label-independent)
  7. temporal features, computed within each patient
  8. write parquet/csv + an audit JSON

At no point does a label, a split assignment, or a cohort membership decision
depend on the note text. Steps 5 and 6 do touch text but apply the identical
rule to both classes, and the audit records how many notes each rule removed
per class so a reviewer can confirm the rule was not differential.

OUTPUT
  <out>/cohort_<arm>_<source>.csv        one row per note
  <out>/patients_<arm>_<source>.csv      one row per patient
  <out>/audit_<arm>_<source>.json        machine-readable audit record

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn import cohort as C            # noqa: E402
from tcwpn import splits as S            # noqa: E402
from tcwpn.io_paths import resolve, MIMIC4, MIMIC3   # noqa: E402

MIN_NOTE_CHARS = 250


# =============================================================================
# LOADERS
# =============================================================================
def load_mimic4(base, note_base):
    patients = pd.read_csv(resolve(base, *MIMIC4["patients"]),
                           usecols=["subject_id", "gender", "anchor_age"])
    diagnoses = pd.read_csv(
        resolve(base, *MIMIC4["diagnoses"]),
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    return patients, diagnoses, resolve(note_base, *MIMIC4["discharge"])


def load_mimic3(base):
    patients = pd.read_csv(resolve(base, *MIMIC3["patients"]),
                           usecols=["SUBJECT_ID", "GENDER", "DOB"])
    admissions = pd.read_csv(resolve(base, *MIMIC3["admissions"]),
                             usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"])
    diagnoses = pd.read_csv(resolve(base, *MIMIC3["diagnoses"]),
                            usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
    diagnoses.columns = [c.lower() for c in diagnoses.columns]
    diagnoses = diagnoses.rename(columns={"icd9_code": "icd_code"})
    diagnoses["icd_version"] = 9
    return patients, admissions, diagnoses, resolve(base, *MIMIC3["noteevents"])


def stream_mimic4_notes(path, subject_ids, chunksize=100_000):
    keep = []
    reader = pd.read_csv(
        path, usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        chunksize=chunksize, low_memory=False,
    )
    for chunk in reader:
        sel = chunk[chunk["subject_id"].isin(subject_ids)]
        if len(sel):
            keep.append(sel)
    if not keep:
        return pd.DataFrame(
            columns=["note_id", "subject_id", "hadm_id", "charttime", "text"]
        )
    df = pd.concat(keep, ignore_index=True)
    df["note_source"] = "discharge"
    return df


def stream_mimic3_notes(path, subject_ids, categories, chunksize=100_000):
    """
    MIMIC-III NOTEEVENTS. `categories` restricts note type. Category is
    structured metadata, not note text, so filtering on it does not violate
    the no-text-filtering rule — but it must be applied identically to both
    classes, which it is.
    """
    cats = {c.lower() for c in categories}
    keep = []
    reader = pd.read_csv(
        path,
        usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME",
                 "CATEGORY", "ISERROR", "TEXT"],
        chunksize=chunksize, low_memory=False,
    )
    for chunk in reader:
        chunk.columns = [c.lower() for c in chunk.columns]
        m = (
            chunk["subject_id"].isin(subject_ids)
            & chunk["category"].astype(str).str.lower().str.strip().isin(cats)
            & (chunk["iserror"].isna() | (chunk["iserror"] == 0))
        )
        sel = chunk[m]
        if len(sel):
            keep.append(sel)
    if not keep:
        return pd.DataFrame(
            columns=["note_id", "subject_id", "hadm_id", "charttime", "text"]
        )
    df = pd.concat(keep, ignore_index=True)
    df["charttime"] = df["charttime"].fillna(df["chartdate"])
    df["note_id"] = "M3-" + df["row_id"].astype(str)
    df["note_source"] = df["category"].str.lower().str.strip().str.replace(
        r"[ /]", "_", regex=True
    )
    return df[["note_id", "subject_id", "hadm_id", "charttime", "text",
               "note_source"]]


# =============================================================================
# TEXT NORMALISATION (identical for both classes)
# =============================================================================
def normalise_text(t: str) -> str:
    """
    De-identification placeholders and structural headers are removed. This is
    applied to every note regardless of label. Note that we do NOT lowercase
    here and we do NOT strip punctuation aggressively as the archived pipeline
    did: Bio_ClinicalBERT is an uncased model so lowercasing is redundant, and
    stripping characters damages tokenisation of clinical shorthand.
    """
    import re

    if not isinstance(t, str):
        return ""
    t = re.sub(r"\[\*\*.*?\*\*\]", " ", t)          # MIMIC PHI placeholders
    t = re.sub(r"_{3,}", " ", t)                     # MIMIC-IV ___ redactions
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def add_temporal_features(notes: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient temporal features. All are defined WITHIN a patient, so they
    are invariant to which other patients are in the cohort or the episode.
    """
    n = notes.copy()
    n["charttime"] = pd.to_datetime(n["charttime"], errors="coerce")
    n = n[n["charttime"].notna()].sort_values(["subject_id", "charttime"])

    g = n.groupby("subject_id")["charttime"]
    last = g.transform("max")
    first = g.transform("min")

    n["days_before_patient_last_note"] = (last - n["charttime"]).dt.days.astype(float)
    n["days_since_patient_first_note"] = (n["charttime"] - first).dt.days.astype(float)
    n["note_index_within_patient"] = n.groupby("subject_id").cumcount()
    n["n_notes_patient"] = n.groupby("subject_id")["note_id"].transform("count")
    n["is_patient_last_note"] = n["charttime"].eq(last)
    return n


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--source", choices=["mimic4", "mimic3"], default="mimic4")
    ap.add_argument("--arm", choices=["psych", "clean"], default="psych",
                    help="psych = anxiety vs other psychiatric (PRIMARY); "
                         "clean = anxiety vs no-psychiatric-history (SECONDARY)")
    ap.add_argument("--age-min", type=int, default=18)
    ap.add_argument("--age-max", type=int, default=50)
    ap.add_argument("--anxiety-seq-num", type=int, default=None,
                    help="require an anxiety code at seq_num <= N; omit for "
                         "any-position (default, more inclusive)")
    ap.add_argument("--max-notes-per-patient", type=int, default=8,
                    help="cap notes per patient (applied by recency, before "
                         "any text is inspected) so heavy utilisers do not "
                         "dominate the note pool")
    ap.add_argument("--train-control-ratio", type=float, default=None,
                    help="cap TRAIN control patients at N x case patients; "
                         "val/test are never resampled")
    ap.add_argument("--split-salt", default="tcwpn-clean-v1")
    ap.add_argument("--mimic4-path", default=os.getenv("MIMIC_IV_DATASET_PATH", ""))
    ap.add_argument("--mimic4-note-path",
                    default=os.getenv("MIMIC_IV_NOTE_DATASET_PATH", ""))
    ap.add_argument("--mimic3-path", default=os.getenv("MIMIC_III_DATASET_PATH", ""))
    ap.add_argument("--mimic3-categories", nargs="+",
                    default=["Discharge summary"],
                    help="MIMIC-III note categories to keep. Keep this narrow "
                         "and matched to the MIMIC-IV note type (discharge) or "
                         "the transfer experiment confounds note type with "
                         "dataset.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "args": vars(args)}

    print("=" * 74)
    print(f"CLEAN COHORT BUILD — source={args.source}  arm={args.arm}")
    print("=" * 74)

    # ---- 1/2. structured cohort ------------------------------------------------
    if args.source == "mimic4":
        if not args.mimic4_path or not args.mimic4_note_path:
            raise SystemExit("set --mimic4-path and --mimic4-note-path "
                             "(or MIMIC_IV_DATASET_PATH / MIMIC_IV_NOTE_DATASET_PATH)")
        patients, diagnoses, note_path = load_mimic4(args.mimic4_path,
                                                     args.mimic4_note_path)
        eligible = C.eligible_by_age(patients, args.age_min, args.age_max)
        demo = patients[["subject_id", "gender", "anchor_age"]].rename(
            columns={"anchor_age": "age"})
    else:
        if not args.mimic3_path:
            raise SystemExit("set --mimic3-path (or MIMIC_III_DATASET_PATH)")
        p3, a3, diagnoses, note_path = load_mimic3(args.mimic3_path)
        eligible = C.eligible_by_age_mimic3(p3, a3, args.age_min, args.age_max)
        demo = p3.rename(columns=str.lower)[["subject_id", "gender"]].assign(age=np.nan)

    print(f"  eligible patients aged {args.age_min}-{args.age_max}: {len(eligible):,}")

    per_patient = C.build_patient_cohort(
        diagnoses, eligible_subject_ids=eligible,
        require_anxiety_seq_num=args.anxiety_seq_num,
    )
    arm_counts = per_patient["arm"].value_counts().to_dict()
    print("  arm assignment:", arm_counts)
    audit["arm_counts_all"] = {k: int(v) for k, v in arm_counts.items()}

    control_arm = "psych_control" if args.arm == "psych" else "clean_control"
    keep = per_patient[per_patient["arm"].isin(["case", control_arm])].copy()
    if keep.empty:
        raise SystemExit("no patients in the selected arms")
    print(f"  cases: {(keep.label == 1).sum():,}   "
          f"controls ({control_arm}): {(keep.label == 0).sum():,}")

    # ---- 3. split BEFORE any text ---------------------------------------------
    split_map = S.assign_splits(keep["subject_id"], salt=args.split_salt)
    keep = keep.merge(split_map, on="subject_id", how="left")

    if args.train_control_ratio:
        before = len(keep)
        keep = S.downsample_control_patients(
            keep, ratio=args.train_control_ratio, seed=42, only_split="train")
        print(f"  train control down-sampling: {before:,} -> {len(keep):,} patients")
        audit["train_control_ratio_applied"] = args.train_control_ratio

    audit["patients_per_split"] = (
        keep.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    )

    # ---- 4. NOW load text ------------------------------------------------------
    subject_ids = set(keep["subject_id"].unique())
    print(f"\n  loading notes for {len(subject_ids):,} patients from {note_path.name} ...")
    if args.source == "mimic4":
        notes = stream_mimic4_notes(note_path, subject_ids)
    else:
        notes = stream_mimic3_notes(note_path, subject_ids, args.mimic3_categories)
    print(f"  raw notes: {len(notes):,}")
    audit["n_notes_raw"] = int(len(notes))

    notes = notes.merge(
        keep[["subject_id", "label", "arm", "split"]], on="subject_id", how="inner")
    notes["text"] = notes["text"].map(normalise_text)

    # ---- 5. length filter, identical rule per class ----------------------------
    before_by_label = notes.groupby("label").size().to_dict()
    notes = notes[notes["text"].str.len() >= MIN_NOTE_CHARS]
    after_by_label = notes.groupby("label").size().to_dict()
    audit["length_filter"] = {
        "min_chars": MIN_NOTE_CHARS,
        "removed_case": int(before_by_label.get(1, 0) - after_by_label.get(1, 0)),
        "removed_control": int(before_by_label.get(0, 0) - after_by_label.get(0, 0)),
        "removed_case_pct": round(
            100 * (before_by_label.get(1, 0) - after_by_label.get(1, 0))
            / max(before_by_label.get(1, 1), 1), 2),
        "removed_control_pct": round(
            100 * (before_by_label.get(0, 0) - after_by_label.get(0, 0))
            / max(before_by_label.get(0, 1), 1), 2),
    }
    print(f"  after length filter (>={MIN_NOTE_CHARS} chars): {len(notes):,}")

    # ---- 6. cross-split duplicate text -----------------------------------------
    notes["_text_hash"] = notes["text"].map(
        lambda t: hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest())
    spans = notes.groupby("_text_hash")["split"].nunique()
    cross = set(spans[spans > 1].index)
    if cross:
        n_before = len(notes)
        notes = notes[~notes["_text_hash"].isin(cross)]
        print(f"  removed {n_before - len(notes):,} notes whose exact text "
              f"appeared in more than one split")
    audit["cross_split_duplicate_text_hashes"] = int(len(cross))

    # within-split exact duplicates: keep one
    n_before = len(notes)
    notes = notes.drop_duplicates(subset=["subject_id", "_text_hash"], keep="first")
    audit["within_patient_duplicate_notes_dropped"] = int(n_before - len(notes))

    # ---- 7. temporal + per-patient note cap ------------------------------------
    notes = add_temporal_features(notes)
    if args.max_notes_per_patient:
        notes = (
            notes.sort_values(["subject_id", "charttime"], ascending=[True, False])
            .groupby("subject_id", group_keys=False)
            .head(args.max_notes_per_patient)
        )
        notes = add_temporal_features(notes)   # recompute after the cap
    print(f"  final notes: {len(notes):,}")

    # ---- metadata that must NOT be used for filtering ---------------------------
    anx_hadms = C.anxiety_admissions(diagnoses)
    notes["anx_coded_this_adm"] = notes["hadm_id"].isin(anx_hadms)

    notes = notes.merge(demo, on="subject_id", how="left")

    # ---- 8. verify + write ------------------------------------------------------
    overlaps = S.assert_disjoint(notes)
    S.assert_no_duplicate_notes(notes)
    audit["split_overlaps"] = overlaps
    audit["cross_split_text_hashes_remaining"] = S.assert_no_duplicate_text_hashes(notes)

    summary = S.split_summary(notes)
    print("\n" + summary.to_string(index=False))
    audit["split_summary"] = summary.to_dict(orient="records")

    cols = [
        "note_id", "subject_id", "hadm_id", "charttime", "note_source",
        "label", "arm", "split", "gender", "age",
        "days_before_patient_last_note", "days_since_patient_first_note",
        "note_index_within_patient", "n_notes_patient", "is_patient_last_note",
        "anx_coded_this_adm", "text",
    ]
    cols = [c for c in cols if c in notes.columns]
    tag = f"{args.arm}_{args.source}"
    notes_path = out_dir / f"cohort_{tag}.csv"
    notes[cols].to_csv(notes_path, index=False)
    keep.to_csv(out_dir / f"patients_{tag}.csv", index=False)
    with open(out_dir / f"audit_{tag}.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"\n  wrote {notes_path}")
    print(f"  wrote {out_dir / f'patients_{tag}.csv'}")
    print(f"  wrote {out_dir / f'audit_{tag}.json'}")
    print("\nNEXT: python -m scripts.audit_cohort --cohort "
          f"{notes_path}")


if __name__ == "__main__":
    main()
