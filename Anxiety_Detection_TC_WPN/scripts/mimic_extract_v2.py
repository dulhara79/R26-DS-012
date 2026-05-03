"""
updated extract_data.py mimic_extract_v2.py
MIMIC-IV Data Extraction V2 (Publication-Grade)
TC-WPN Research Pipeline
Author: Dulhara Kaushalya
Goal:
- Cleaner anxiety phenotype
- Cleaner psych-free controls
- Better temporal fidelity
- Section-aware weighting
- Multi-source note support
- Leakage-safe patient splits
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
# =============================================================================
NOTE_SOURCES = [
    {
        "file": "discharge.csv.gz",
        "name": "discharge",
        "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    },
]


# =============================================================================
# LOAD NOTES
# =============================================================================
def load_relevant_notes(target_hadm_ids):
    all_notes = []

    for source in NOTE_SOURCES:
        note_path = MIMIC_IV_NOTE_PATH / "note" / source["file"]

        if not note_path.exists():
            print(f"  ⚠ Missing note source: {source['file']}")
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
            relevant_chunk = chunk[chunk["hadm_id"].isin(target_hadm_ids)]

            if len(relevant_chunk) > 0:
                relevant_chunk["source_type"] = source["name"]
                source_notes.append(relevant_chunk)

            if (i + 1) % 10 == 0:
                print(f"    Processed {(i + 1) * 50000:,} rows...")

        if source_notes:
            source_df = pd.concat(source_notes, ignore_index=True)
            print(f"  ✓ {source['name']}: {len(source_df):,} notes")
            all_notes.append(source_df)

    if not all_notes:
        raise ValueError("No note data loaded.")

    return pd.concat(all_notes, ignore_index=True)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("MIMIC-IV DATA EXTRACTION V2 — PUBLICATION GRADE")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — LOAD HOSPITAL DATA
    # =========================================================================
    print("\nSTEP 1: Loading hospital tables...")

    patients = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )

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

    # Young adults only
    patients = patients[(patients["anchor_age"] >= 18) & (patients["anchor_age"] <= 30)]

    print(f"✓ Young adult patients: {len(patients):,}")

    # =========================================================================
    # STEP 2 — COHORT CONSTRUCTION
    # =========================================================================
    print("\nSTEP 2: Building anxiety and control cohorts...")

    anxiety_cases = identify_anxiety_patients(diagnoses)
    control_cases = identify_control_patients(diagnoses, anxiety_cases)

    all_cases = pd.concat([anxiety_cases, control_cases], ignore_index=True)

    print(f"✓ Anxiety cases: {len(anxiety_cases):,}")
    print(f"✓ Control cases: {len(control_cases):,}")

    # =========================================================================
    # STEP 3 — LOAD NOTES
    # =========================================================================
    print("\nSTEP 3: Loading clinical notes...")

    target_hadm_ids = set(all_cases["hadm_id"].unique())

    notes = load_relevant_notes(target_hadm_ids)

    print(f"✓ Total loaded notes: {len(notes):,}")

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
    # STEP 5 — CLEANING
    # =========================================================================
    print("\nSTEP 5: Cleaning notes...")

    merged["clinical_note_text"] = merged["text"].apply(clean_note_text)

    merged = verify_and_clean_notes(merged)

    merged["note_length"] = merged["clinical_note_text"].str.len()

    print(f"✓ Post-clean notes: {len(merged):,}")

    # =========================================================================
    # STEP 6 — TEMPORAL FEATURES
    # =========================================================================
    print("\nSTEP 6: Computing temporal features...")

    merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")

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

    anxiety_results = final_df.loc[anxiety_mask, "clinical_note_text"].apply(
        assign_anxiety_confidence
    )

    final_df.loc[anxiety_mask, "label_confidence"] = [x[0] for x in anxiety_results]

    final_df.loc[anxiety_mask, "anxiety_context"] = [x[1] for x in anxiety_results]

    final_df.loc[control_mask, "label_confidence"] = final_df.loc[
        control_mask, "clinical_note_text"
    ].apply(penalize_control_noise)

    final_df["section_quality"] = final_df["clinical_note_text"].apply(
        compute_section_quality
    )

    # Final weight
    final_df["training_weight"] = (
        final_df["label_confidence"] * final_df["section_quality"]
    )

    final_df["has_text_signal"] = final_df["label_confidence"] > 0.5

    # =========================================================================
    # STEP 8 — SORT + SPLIT
    # =========================================================================
    print("\nSTEP 8: Patient-level leakage-safe split...")

    final_df = final_df.sort_values(by=["subject_id", "charttime"]).reset_index(
        drop=True
    )

    unique_patients = final_df["subject_id"].dropna().unique()

    train_val_ids, test_ids = train_test_split(
        unique_patients,
        test_size=0.10,
        random_state=42,
        shuffle=True,
    )

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.1111,
        random_state=42,
        shuffle=True,
    )

    final_df["dataset_split"] = "train"

    final_df.loc[final_df["subject_id"].isin(val_ids), "dataset_split"] = "val"

    final_df.loc[final_df["subject_id"].isin(test_ids), "dataset_split"] = "test"

    # =========================================================================
    # STEP 9 — OUTPUT DATASETS
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

    final_dataset.rename(
        columns={"anchor_age": "age_at_admission"},
        inplace=True,
    )

    train_df = final_dataset[final_dataset["dataset_split"] == "train"]
    val_df = final_dataset[final_dataset["dataset_split"] == "val"]
    test_df = final_dataset[final_dataset["dataset_split"] == "test"]

    # -------------------------------------------------------------------------
    # REAL-WORLD SPLITS
    # -------------------------------------------------------------------------
    train_df.to_csv(
        OUTPUT_DIR / "mimic_anxiety_train_real_world.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR / "mimic_anxiety_val_real_world.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "mimic_anxiety_test_real_world.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # BALANCED TRAIN
    # -------------------------------------------------------------------------
    train_anx = train_df[train_df["has_anxiety"] == 1]
    train_ctrl = train_df[train_df["has_anxiety"] == 0]

    n_samples = min(len(train_anx), len(train_ctrl))

    train_balanced = pd.concat(
        [
            train_anx.sample(n=n_samples, random_state=42),
            train_ctrl.sample(n=n_samples, random_state=42),
        ]
    ).sample(frac=1, random_state=42)

    train_balanced.to_csv(
        OUTPUT_DIR / "mimic_anxiety_train_balanced.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # HIGH CONF TRAIN
    # -------------------------------------------------------------------------
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
        OUTPUT_DIR / "mimic_anxiety_train_high_conf.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # HIGH CONF TEST
    # -------------------------------------------------------------------------
    test_high_conf = test_df[
        ((test_df["has_anxiety"] == 1) & (test_df["label_confidence"] >= 0.7))
        | ((test_df["has_anxiety"] == 0) & (test_df["label_confidence"] >= 0.9))
    ]

    test_high_conf.to_csv(
        OUTPUT_DIR / "mimic_anxiety_test_high_conf.csv",
        index=False,
    )

    # =========================================================================
    # STEP 10 — SANITY CHECKS
    # =========================================================================
    print("\nSTEP 10: Leakage checks...")

    leak_train_val = len(set(train_ids) & set(val_ids))
    leak_train_test = len(set(train_ids) & set(test_ids))
    leak_val_test = len(set(val_ids) & set(test_ids))

    print(f"Leak Train-Val: {leak_train_val}")
    print(f"Leak Train-Test: {leak_train_test}")
    print(f"Leak Val-Test: {leak_val_test}")

    if leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0:
        print("✅ ZERO PATIENT LEAKAGE CONFIRMED")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ EXTRACTION V2 COMPLETE")
    print("=" * 80)

    print(f"Master dataset: {len(final_dataset):,}")
    print(f"Train: {len(train_df):,}")
    print(f"Val: {len(val_df):,}")
    print(f"Test: {len(test_df):,}")
    print(f"Balanced Train: {len(train_balanced):,}")
    print(f"High-Conf Train: {len(train_high_conf):,}")
    print(f"High-Conf Test: {len(test_high_conf):,}")


if __name__ == "__main__":
    main()
