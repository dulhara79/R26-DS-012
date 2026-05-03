"""
mimic_extract_v2.py
MIMIC-IV Data Extraction V2 — Publication-Grade
TC-WPN Research Pipeline
Author: Dulhara Kaushalya

FIXES IN THIS VERSION:
- NOTE_SOURCES now includes discharge_detail (addendum notes)
- verify_and_clean_notes no longer drops valid notes via template filter
- penalize_control_noise fixed (was causing 100% control contamination)
- assign_anxiety_confidence adds family_history + situational filters
- high_conf filter uses label_confidence >= 0.9 for controls (stricter)

RUN ORDER:
  python -m scripts.mimic_extract_v2
  python -m scripts.convert_csv_to_pkl_v2

THEN verify with diagnostic before uploading PKLs to Kaggle.
"""

import sys
from pathlib import Path
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIG
# =============================================================================
from config.settings import (
    MIMIC_IV_DATASET_PATH,
    MIMIC_IV_NOTE_DATASET_PATH,
    MIMIC_PROCESSED_BASE_DIR,
)

from tc_wpn.data.extraction_v2 import (
    load_csv_safe,
    identify_anxiety_patients,
    identify_control_patients,
    compute_temporal_features,
    clean_note_text,
    verify_and_clean_notes,
    assign_anxiety_confidence,
    penalize_control_noise,
    compute_section_quality,
)

warnings.filterwarnings("ignore")

MIMIC_IV_PATH = Path(MIMIC_IV_DATASET_PATH)
MIMIC_IV_NOTE_PATH = Path(MIMIC_IV_NOTE_DATASET_PATH)
OUTPUT_DIR = Path(MIMIC_PROCESSED_BASE_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# NOTE SOURCE CONFIGURATION
# Add or remove sources here. Each must have the same usecols.
# discharge_detail = addendum/amendment notes (denser clinical reasoning)
# =============================================================================
NOTE_SOURCES = [
    {
        "file": "discharge.csv.gz",
        "name": "discharge",
        "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    },
    {
        "file": "discharge_detail.csv.gz",
        "name": "discharge_detail",
        "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    },
    # Uncomment below if you have these files in your MIMIC note path:
    # {
    #     "file": "physician.csv.gz",
    #     "name": "physician",
    #     "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    # },
    # {
    #     "file": "nursing.csv.gz",
    #     "name": "nursing",
    #     "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    # },
]


# =============================================================================
# LOAD NOTES FROM ALL CONFIGURED SOURCES
# =============================================================================
def load_relevant_notes(target_hadm_ids):
    all_notes = []

    for source in NOTE_SOURCES:
        note_path = MIMIC_IV_NOTE_PATH / "note" / source["file"]

        if not note_path.exists():
            print(f"  ⚠  Missing note source: {source['file']} — skipping.")
            continue

        print(f"\nLoading {source['name']} notes...")

        chunk_iter = pd.read_csv(
            note_path,
            usecols=source["usecols"],
            chunksize=50000,
            low_memory=False,
        )

        source_notes = []
        for i, chunk in enumerate(chunk_iter):
            relevant = chunk[chunk["hadm_id"].isin(target_hadm_ids)].copy()
            if len(relevant) > 0:
                relevant["source_type"] = source["name"]
                source_notes.append(relevant)
            if (i + 1) % 10 == 0:
                print(f"    Processed {(i + 1) * 50_000:,} rows...")

        if source_notes:
            source_df = pd.concat(source_notes, ignore_index=True)
            print(f"  ✓ {source['name']}: {len(source_df):,} notes")
            all_notes.append(source_df)
        else:
            print(f"  ⚠  {source['name']}: 0 relevant notes found.")

    if not all_notes:
        raise ValueError(
            "No note data loaded. Check NOTE_SOURCES and MIMIC_IV_NOTE_DATASET_PATH."
        )

    combined = pd.concat(all_notes, ignore_index=True)

    # Remove cross-source exact duplicates (same note_id from multiple sources)
    combined = combined.drop_duplicates(subset=["note_id"])

    return combined


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("MIMIC-IV DATA EXTRACTION V2 — PUBLICATION GRADE")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — LOAD HOSPITAL TABLES
    # =========================================================================
    print("\nSTEP 1: Loading hospital tables...")

    patients = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    if patients is None:
        raise FileNotFoundError("Cannot load patients.csv.gz")

    admissions = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
        ],
    )

    diagnoses = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )

    # Young adult cohort only
    patients = patients[(patients["anchor_age"] >= 18) & (patients["anchor_age"] <= 30)]
    print(f"✓ Young adult patients (18–30): {len(patients):,}")

    # =========================================================================
    # STEP 2 — BUILD COHORTS
    # =========================================================================
    print("\nSTEP 2: Building anxiety and control cohorts...")

    anxiety_cases = identify_anxiety_patients(diagnoses)
    control_cases = identify_control_patients(diagnoses, anxiety_cases)

    # Filter to young adult patients only
    young_ids = set(patients["subject_id"])
    anxiety_cases = anxiety_cases[anxiety_cases["subject_id"].isin(young_ids)]
    control_cases = control_cases[control_cases["subject_id"].isin(young_ids)]

    all_cases = pd.concat([anxiety_cases, control_cases], ignore_index=True)

    print(f"✓ Anxiety cases (young adult): {len(anxiety_cases):,}")
    print(f"✓ Control cases (young adult, psych-clean): {len(control_cases):,}")

    # =========================================================================
    # STEP 3 — LOAD NOTES
    # =========================================================================
    print("\nSTEP 3: Loading clinical notes...")

    target_hadm_ids = set(all_cases["hadm_id"].unique())
    notes = load_relevant_notes(target_hadm_ids)

    print(f"\n✓ Total loaded notes (all sources): {len(notes):,}")
    print(f"  Source breakdown:\n{notes['source_type'].value_counts().to_string()}")

    # =========================================================================
    # STEP 4 — MERGE
    # =========================================================================
    print("\nSTEP 4: Merging notes + labels + demographics...")

    merged = notes.merge(
        all_cases[["hadm_id", "has_anxiety"]].drop_duplicates(),
        on="hadm_id",
        how="inner",
    )
    merged = merged.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    )
    merged = merged.merge(
        admissions[["hadm_id", "admittime", "dischtime"]],
        on="hadm_id",
        how="left",
    )

    print(f"✓ Merged notes: {len(merged):,}")

    # =========================================================================
    # STEP 5 — TEXT CLEANING
    # =========================================================================
    print("\nSTEP 5: Cleaning notes...")

    merged["clinical_note_text"] = merged["text"].apply(clean_note_text)

    # verify_and_clean_notes only removes empty/short/duplicate notes
    merged = verify_and_clean_notes(merged)

    merged["note_length"] = merged["clinical_note_text"].str.len()

    print(f"✓ Post-clean notes: {len(merged):,}")

    # =========================================================================
    # STEP 6 — TEMPORAL FEATURES
    # =========================================================================
    print("\nSTEP 6: Computing temporal features...")

    merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")
    merged = merged[merged["charttime"].notna()]

    temporal_df = compute_temporal_features(merged)

    final_df = merged.merge(
        temporal_df,
        on=["note_id", "subject_id", "charttime"],
        how="left",
    )

    # =========================================================================
    # STEP 7 — LABEL CONFIDENCE + SECTION QUALITY
    # =========================================================================
    print("\nSTEP 7: Assigning label confidence + section quality...")

    final_df["label_confidence"] = 1.0
    final_df["anxiety_context"] = "control"

    anxiety_mask = final_df["has_anxiety"] == 1
    control_mask = final_df["has_anxiety"] == 0

    # Anxiety notes
    anxiety_results = final_df.loc[anxiety_mask, "clinical_note_text"].apply(
        assign_anxiety_confidence
    )
    final_df.loc[anxiety_mask, "label_confidence"] = [x[0] for x in anxiety_results]
    final_df.loc[anxiety_mask, "anxiety_context"] = [x[1] for x in anxiety_results]

    # Control notes — penalise only if strong diagnostic language present
    final_df.loc[control_mask, "label_confidence"] = final_df.loc[
        control_mask, "clinical_note_text"
    ].apply(penalize_control_noise)

    # Section quality
    final_df["section_quality"] = final_df["clinical_note_text"].apply(
        compute_section_quality
    )

    # Combined training weight
    final_df["training_weight"] = (
        final_df["label_confidence"] * final_df["section_quality"]
    )

    final_df["has_text_signal"] = final_df["label_confidence"] > 0.5

    # =========================================================================
    # DIAGNOSTIC PRINT — verify contamination before saving
    # =========================================================================
    print("\n--- LABEL QUALITY DIAGNOSTIC ---")
    ctrl_df = final_df[final_df["has_anxiety"] == 0]
    contam = ctrl_df[ctrl_df["label_confidence"] < 0.9]
    print(
        f"Control contamination: {len(contam):,} / {len(ctrl_df):,} "
        f"({100 * len(contam) / max(len(ctrl_df), 1):.1f}%)"
    )
    print(
        f"Anxiety context breakdown:\n"
        f"{final_df[final_df['has_anxiety']==1]['anxiety_context'].value_counts().to_string()}"
    )
    print(
        f"Anxiety confidence distribution:\n"
        f"{final_df[final_df['has_anxiety']==1]['label_confidence'].value_counts().sort_index().to_string()}"
    )
    print("--------------------------------")

    # =========================================================================
    # STEP 8 — PATIENT-LEVEL SPLIT (leakage-safe)
    # =========================================================================
    print("\nSTEP 8: Patient-level leakage-safe split...")

    final_df = final_df.sort_values(by=["subject_id", "charttime"]).reset_index(
        drop=True
    )

    unique_patients = final_df["subject_id"].dropna().unique()

    train_val_ids, test_ids = train_test_split(
        unique_patients, test_size=0.10, random_state=42, shuffle=True
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.1111, random_state=42, shuffle=True
    )

    final_df["dataset_split"] = "train"
    final_df.loc[final_df["subject_id"].isin(val_ids), "dataset_split"] = "val"
    final_df.loc[final_df["subject_id"].isin(test_ids), "dataset_split"] = "test"

    # =========================================================================
    # STEP 9 — SAVE DATASETS
    # =========================================================================
    print("\nSTEP 9: Saving datasets...")

    output_columns = [
        "note_id",
        "subject_id",
        "hadm_id",
        "charttime",
        "source_type",
        "has_anxiety",
        "has_text_signal",
        "label_confidence",
        "section_quality",
        "training_weight",
        "anxiety_context",
        "dataset_split",
        "gender",
        "anchor_age",
        "days_since_first_visit",
        "days_since_last_visit",
        "visit_number",
        "total_visits",
        "note_age_days",
        "is_most_recent",
        "note_length",
        "clinical_note_text",
    ]

    final_dataset = final_df[output_columns].copy()
    final_dataset.rename(columns={"anchor_age": "age_at_admission"}, inplace=True)

    train_df = final_dataset[final_dataset["dataset_split"] == "train"]
    val_df = final_dataset[final_dataset["dataset_split"] == "val"]
    test_df = final_dataset[final_dataset["dataset_split"] == "test"]

    # Real-world splits (natural distribution)
    train_df.to_csv(OUTPUT_DIR / "mimic_anxiety_train_real_world.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "mimic_anxiety_val_real_world.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "mimic_anxiety_test_real_world.csv", index=False)

    # Balanced train (equal anxiety/control)
    train_anx = train_df[train_df["has_anxiety"] == 1]
    train_ctrl = train_df[train_df["has_anxiety"] == 0]
    n_samples = min(len(train_anx), len(train_ctrl))

    train_balanced = pd.concat(
        [
            train_anx.sample(n=n_samples, random_state=42),
            train_ctrl.sample(n=n_samples, random_state=42),
        ]
    ).sample(frac=1, random_state=42)
    train_balanced.to_csv(OUTPUT_DIR / "mimic_anxiety_train_balanced.csv", index=False)

    # High-confidence train
    # Anxiety: conf >= 0.7 (named disorder or active general mention)
    # Control: conf >= 0.9 (clean, no diagnostic anxiety language)
    train_high_conf = train_balanced[
        (
            (train_balanced["has_anxiety"] == 1)
            & (train_balanced["label_confidence"] >= 0.7)
        )
        | (
            (train_balanced["has_anxiety"] == 0)
            & (train_balanced["label_confidence"] >= 0.9)
        )
    ]
    train_high_conf.to_csv(
        OUTPUT_DIR / "mimic_anxiety_train_high_conf.csv", index=False
    )

    # High-confidence test
    test_high_conf = test_df[
        ((test_df["has_anxiety"] == 1) & (test_df["label_confidence"] >= 0.7))
        | ((test_df["has_anxiety"] == 0) & (test_df["label_confidence"] >= 0.9))
    ]
    test_high_conf.to_csv(OUTPUT_DIR / "mimic_anxiety_test_high_conf.csv", index=False)

    # =========================================================================
    # STEP 10 — LEAKAGE CHECKS
    # =========================================================================
    print("\nSTEP 10: Leakage checks...")

    leak_tv = len(set(train_ids) & set(val_ids))
    leak_tt = len(set(train_ids) & set(test_ids))
    leak_vt = len(set(val_ids) & set(test_ids))

    print(f"  Leak Train↔Val  : {leak_tv}")
    print(f"  Leak Train↔Test : {leak_tt}")
    print(f"  Leak Val↔Test   : {leak_vt}")

    if leak_tv == 0 and leak_tt == 0 and leak_vt == 0:
        print("  ✅ ZERO PATIENT LEAKAGE CONFIRMED")
    else:
        print("  ❌ LEAKAGE DETECTED — check train_test_split logic")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ EXTRACTION V2 COMPLETE")
    print("=" * 80)
    print(f"Master dataset   : {len(final_dataset):,}")
    print(f"Train (raw)      : {len(train_df):,}")
    print(f"Val              : {len(val_df):,}")
    print(f"Test             : {len(test_df):,}")
    print(f"Balanced Train   : {len(train_balanced):,}")
    print(f"High-Conf Train  : {len(train_high_conf):,}")
    print(f"High-Conf Test   : {len(test_high_conf):,}")
    print(f"\nSource breakdown (full dataset):")
    print(final_dataset["source_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
