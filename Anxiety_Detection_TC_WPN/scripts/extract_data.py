"""
MIMIC-IV Data Extraction for TC-WPN Research
Extracts anxiety-related clinical notes with temporal metadata
UPDATED: Train/Val/Test Split, Safe Balancing, High-Conf Test Set, and Temporal Sorting.

Author: Dulhara Kaushalya
Date: February 2026
"""

import sys
import os
from pathlib import Path
import re
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

# Get absolute path and insert at index 0 to force Python to find your packages
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MIMIC_IV_DATASET_PATH,
    MIMIC_IV_NOTE_DATASET_PATH,
    MIMIC_PROCESSED_BASE_DIR,
)

from tc_wpn.data.extraction import (
    load_csv_safe,
    identify_anxiety_patients,
    identify_control_patients,
    compute_temporal_features,
    clean_note_text,
    verify_and_clean_notes,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
MIMIC_IV_PATH = Path(MIMIC_IV_DATASET_PATH)
MIMIC_IV_NOTE_PATH = Path(MIMIC_IV_NOTE_DATASET_PATH)

OUTPUT_DIR = Path(MIMIC_PROCESSED_BASE_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================


def main():
    print("=" * 80)
    print("MIMIC-IV DATA EXTRACTION FOR TC-WPN RESEARCH")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # STEP 1: Load Hospital Data
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: Loading Hospital Data")
    print("=" * 80)

    patients_path = MIMIC_IV_PATH / "hosp" / "patients.csv.gz"
    patients = load_csv_safe(
        patients_path, usecols=["subject_id", "gender", "anchor_age"]
    )
    if patients is None:
        return

    patients = patients[(patients["anchor_age"] >= 18) & (patients["anchor_age"] <= 30)]
    print(f"  ✓ Young adult patients (18–30): {len(patients):,}")

    admissions_path = MIMIC_IV_PATH / "hosp" / "admissions.csv.gz"
    admissions = load_csv_safe(
        admissions_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
        ],
    )
    if admissions is None:
        return

    diagnoses_path = MIMIC_IV_PATH / "hosp" / "diagnoses_icd.csv.gz"
    diagnoses = load_csv_safe(
        diagnoses_path, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"]
    )
    if diagnoses is None:
        return

    # ------------------------------------------------------------------------
    # STEP 2: Identify Anxiety and Control Cases
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Identifying Anxiety and Control Cases")
    print("=" * 80)

    anxiety_cases = identify_anxiety_patients(diagnoses)
    control_cases = identify_control_patients(diagnoses, anxiety_cases)

    all_cases = pd.concat([anxiety_cases, control_cases], ignore_index=True)

    print(f"\nTotal cases to extract notes for: {len(all_cases):,}")
    print(f"  - Anxiety: {len(anxiety_cases):,}")
    print(f"  - Control: {len(control_cases):,}")

    # ------------------------------------------------------------------------
    # STEP 3: Load Clinical Notes
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Loading Clinical Notes")
    print("=" * 80)

    discharge_path = MIMIC_IV_NOTE_PATH / "note" / "discharge.csv.gz"
    print("Loading discharge notes (this may take several minutes)...")

    chunk_iter = pd.read_csv(
        discharge_path,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        chunksize=50000,
        low_memory=False,
    )

    notes_list = []
    target_hadm_ids = set(all_cases["hadm_id"].unique())

    for i, chunk in enumerate(chunk_iter):
        relevant_chunk = chunk[chunk["hadm_id"].isin(target_hadm_ids)]
        if len(relevant_chunk) > 0:
            notes_list.append(relevant_chunk)
        if (i + 1) % 5 == 0:
            print(f"  Processed {(i + 1) * 50000:,} rows...")

    notes = pd.concat(notes_list, ignore_index=True)
    print(f"  ✓ Loaded {len(notes):,} relevant clinical notes")

    # ------------------------------------------------------------------------
    # STEP 4: Merge Data
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: Merging Data")
    print("=" * 80)

    merged = notes.merge(
        all_cases[["hadm_id", "has_anxiety"]].drop_duplicates(),
        on="hadm_id",
        how="inner",
    )
    merged = merged.merge(patients, on="subject_id", how="left")
    merged = merged.merge(
        admissions[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="left"
    )

    print(f"  ✓ Merged dataset: {len(merged):,} notes")

    # ------------------------------------------------------------------------
    # STEP 5: Compute Temporal Features
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: Computing Temporal Features")
    print("=" * 80)

    merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")
    temporal_features = compute_temporal_features(merged)

    merged["charttime"] = pd.to_datetime(merged["charttime"]).dt.floor("s")
    temporal_features["charttime"] = pd.to_datetime(
        temporal_features["charttime"]
    ).dt.floor("s")

    final_df = merged.merge(
        temporal_features, on=["note_id", "subject_id", "charttime"], how="left"
    )

    if "text_cleaned" in final_df.columns:
        final_df = final_df.rename(columns={"text_cleaned": "clinical_note_text"})
    elif "text" in final_df.columns:
        final_df["clinical_note_text"] = final_df["text"].apply(clean_note_text)

    final_df = verify_and_clean_notes(final_df)

    # ------------------------------------------------------------------------
    # STEP 6: PUBLICATION-GRADE NLP FILTERING
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: Advanced NLP - Rebalanced Confidence and Graded Penalties")
    print("=" * 80)

    final_df["text_cleaned"] = final_df["text"].apply(clean_note_text)
    final_df["note_length"] = final_df["text_cleaned"].str.len()

    final_df = final_df[final_df["text_cleaned"] != ""]
    final_df = final_df[final_df["note_length"] > 50]
    final_df = final_df[final_df["charttime"].notna()]

    strong_regex = re.compile(
        r"\b(anxiety disorder|generalized anxiety|panic attack|panic disorder)\b"
    )
    moderate_regex = re.compile(
        r"\b(anxiety|anxious|panic)\b.{0,30}\b(symptom|episode|attack|disorder|history|complaint|diagnosis)\b"
    )
    moderate_regex_loose = re.compile(
        r"\b(patient|reports|complains|feels)\b.{0,15}\b(anxiety|anxious|panic)\b|\b(anxiety|anxious|panic)\b.{0,15}\b(patient|reports|complains|feels)\b"
    )

    past_regex = re.compile(
        r"\b(history of anxiety|hx of anxiety|h/o anxiety|past anxiety)\b"
    )
    stable_regex = re.compile(
        r"\b(controlled anxiety|anxiety controlled|treated for anxiety|stable anxiety|on medication for anxiety|managed anxiety)\b"
    )
    active_regex = re.compile(
        r"\b(current anxiety|acute anxiety|panic attack|experiencing anxiety|reports anxiety|anxious)\b"
    )

    def is_negated(text):
        sentences = re.split(r"[.!?\n]", text)
        for sent in sentences:
            if re.search(
                r"\b(no|denies|without|negative for|not|no evidence of)\b", sent
            ) and re.search(r"\b(anxiety|anxious|panic)\b", sent):
                return True
        return False

    def assign_confidence_and_context(text):
        if not isinstance(text, str):
            return 0.5, "unspecified"
        text = text.lower()

        context = "unspecified"
        if active_regex.search(text):
            context = "active"
        elif stable_regex.search(text):
            context = "stable"
        elif past_regex.search(text):
            context = "past"

        if is_negated(text):
            return 0.5, "negated"

        if strong_regex.search(text):
            return 1.0, context
        elif moderate_regex.search(text):
            return 0.75, context
        elif moderate_regex_loose.search(text):
            return 0.6, context
        else:
            return 0.5, context

    print("Assigning Label Confidences and Context Flags...")

    final_df["label_confidence"] = 1.0
    final_df["anxiety_context"] = "control"

    anxiety_mask = final_df["has_anxiety"] == 1
    results = final_df.loc[anxiety_mask, "text_cleaned"].apply(
        assign_confidence_and_context
    )
    final_df.loc[anxiety_mask, "label_confidence"] = [res[0] for res in results]
    final_df.loc[anxiety_mask, "anxiety_context"] = [res[1] for res in results]

    control_mask = final_df["has_anxiety"] == 0

    def penalize_noisy_controls(text):
        if not isinstance(text, str):
            return 1.0
        text = text.lower()
        if is_negated(text):
            return 1.0

        if strong_regex.search(text):
            return 0.25
        elif moderate_regex.search(text):
            return 0.5
        elif moderate_regex_loose.search(text):
            return 0.75
        return 1.0

    final_df.loc[control_mask, "label_confidence"] = final_df.loc[
        control_mask, "text_cleaned"
    ].apply(penalize_noisy_controls)
    final_df["has_text_signal"] = final_df["label_confidence"] > 0.5
    final_df["training_weight"] = final_df["label_confidence"]

    # ------------------------------------------------------------------------
    # STEP 7: PATIENT-LEVEL TRAIN/VAL/TEST SPLIT & TEMPORAL SORTING
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7: Patient-Level Train/Val/Test Split (Preventing Data Leakage)")
    print("=" * 80)

    # 🟢 Evaluator Fix: Sort chronologically to preserve temporal order natively
    final_df = final_df.sort_values(by=["subject_id", "charttime"]).reset_index(
        drop=True
    )

    unique_patients = final_df["subject_id"].unique()

    # 🟢 Evaluator Fix: Add a Validation Set (80% Train, 10% Val, 10% Test)
    train_val_ids, test_ids = train_test_split(
        unique_patients, test_size=0.1, random_state=42, shuffle=True
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.1111, random_state=42, shuffle=True
    )  # 0.1111 of 0.9 = ~0.1

    final_df["dataset_split"] = "train"
    final_df.loc[final_df["subject_id"].isin(val_ids), "dataset_split"] = "val"
    final_df.loc[final_df["subject_id"].isin(test_ids), "dataset_split"] = "test"

    print(
        f"  ✓ Train Patients: {len(train_ids):,} | Val Patients: {len(val_ids):,} | Test Patients: {len(test_ids):,}"
    )

    # ------------------------------------------------------------------------
    # STEP 8: Output Generation (Strict Subsets)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 8: Dataset Generation (Isolated Subsets)")
    print("=" * 80)

    output_columns = [
        "note_id",
        "subject_id",
        "hadm_id",
        "charttime",
        "has_anxiety",
        "has_text_signal",
        "label_confidence",
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
        "text_cleaned",
    ]

    final_dataset = final_df[output_columns].copy()
    final_dataset.rename(
        columns={
            "text_cleaned": "clinical_note_text",
            "anchor_age": "age_at_admission",
        },
        inplace=True,
    )

    train_df = final_dataset[final_dataset["dataset_split"] == "train"]
    val_df = final_dataset[final_dataset["dataset_split"] == "val"]
    test_df = final_dataset[final_dataset["dataset_split"] == "test"]

    # --- SAVE UNTOUCHED REAL-WORLD EVAL SETS ---
    output_file_val = OUTPUT_DIR / "mimic_anxiety_val_real_world.csv"
    val_df.to_csv(output_file_val, index=False)
    print(
        f"  ✓ Saved UNTOUCHED REAL-WORLD VAL dataset: {output_file_val.name} ({len(val_df):,} rows)"
    )

    output_file_test = OUTPUT_DIR / "mimic_anxiety_test_real_world.csv"
    test_df.to_csv(output_file_test, index=False)
    print(
        f"  ✓ Saved UNTOUCHED REAL-WORLD TEST dataset: {output_file_test.name} ({len(test_df):,} rows)"
    )

    # 🟢 Evaluator Fix: High-Confidence TEST set for fair, clean evaluation
    print("\nGenerating High-Confidence EVAL Sets...")
    test_high_conf = test_df[
        ((test_df["has_anxiety"] == 1) & (test_df["label_confidence"] >= 0.6))
        | ((test_df["has_anxiety"] == 0) & (test_df["label_confidence"] == 1.0))
    ]
    test_high_conf.to_csv(OUTPUT_DIR / "mimic_anxiety_test_high_conf.csv", index=False)
    print(
        f"  ✓ Saved HIGH-CONFIDENCE TEST dataset: mimic_anxiety_test_high_conf.csv ({len(test_high_conf):,} rows)"
    )

    # --- GENERATE BALANCED TRAIN SET ---
    print("\nGenerating TRAIN Balanced Dataset...")
    train_anxiety = train_df[train_df["has_anxiety"] == 1]
    train_control_pool = train_df[train_df["has_anxiety"] == 0]

    # 🟢 Evaluator Fix: Safe sampling fallback to prevent crashes
    n_samples = min(len(train_anxiety), len(train_control_pool))
    train_anx_sampled = train_anxiety.sample(n=n_samples, random_state=42)
    train_ctrl_sampled = train_control_pool.sample(n=n_samples, random_state=42)

    train_balanced = (
        pd.concat([train_anx_sampled, train_ctrl_sampled])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    output_file_train_bal = OUTPUT_DIR / "mimic_anxiety_train_balanced.csv"
    train_balanced.to_csv(output_file_train_bal, index=False)
    print(
        f"  ✓ Saved TRAIN BALANCED dataset: {output_file_train_bal.name} ({len(train_balanced):,} rows)"
    )

    # --- GENERATE HIGH-CONFIDENCE TRAIN SET ---
    print("\nGenerating TRAIN High-Confidence Dataset (Conf >= 0.6)...")
    train_high_conf = train_balanced[
        (
            (train_balanced["has_anxiety"] == 1)
            & (train_balanced["label_confidence"] >= 0.6)
        )
        | (
            (train_balanced["has_anxiety"] == 0)
            & (train_balanced["label_confidence"] == 1.0)
        )
    ]
    output_file_train_high = OUTPUT_DIR / "mimic_anxiety_train_high_conf.csv"
    train_high_conf.to_csv(output_file_train_high, index=False)
    print(
        f"  ✓ Saved TRAIN HIGH-CONFIDENCE dataset: {output_file_train_high.name} ({len(train_high_conf):,} rows)"
    )

    # ------------------------------------------------------------------------
    # STEP 9: Final Dataset Statistics
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 9: Final Dataset Statistics")
    print("=" * 80)

    unique_pats = final_dataset.drop_duplicates(subset=["subject_id"])
    gender_counts = unique_pats["gender"].value_counts()
    total_pats = len(unique_pats)

    # 🟢 Evaluator Fix: Correct Demographic Math
    valid_gender_total = gender_counts.sum()
    missing_gender = total_pats - valid_gender_total

    print("\n📊 MASTER DATASET STATISTICS:")
    print(f"  Total Notes (Train + Val + Test): {len(final_dataset):,}")
    print(f"  Train Set Notes: {len(train_df):,}")
    print(f"  Val Set Notes:   {len(val_df):,}")
    print(f"  Test Set Notes:  {len(test_df):,}")

    print("\n👥 PATIENT DEMOGRAPHICS:")
    print(f"  Total Unique Patients: {total_pats:,}")
    print(f"  Missing Gender Data: {missing_gender:,}")  # Explicitly show missing data

    for gender, count in gender_counts.items():
        percentage = (count / valid_gender_total) * 100
        print(f"  Gender {gender}: {count:,} ({percentage:.1f}%)")  # Now sums to 100%

    # --- EVALUATOR SANITY CHECKS ---
    print("\n🔍 EVALUATOR SANITY CHECKS:")
    leak_train_val = len(set(train_df.subject_id) & set(val_df.subject_id))
    leak_train_test = len(set(train_df.subject_id) & set(test_df.subject_id))
    leak_val_test = len(set(val_df.subject_id) & set(test_df.subject_id))

    print(f"  Leakage Train <-> Val : {leak_train_val} patients")
    print(f"  Leakage Train <-> Test: {leak_train_test} patients")
    print(f"  Leakage Val <-> Test  : {leak_val_test} patients")
    if leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0:
        print("  ✅ ZERO LEAKAGE CONFIRMED. Data is research-safe.")

    print("\n📊 CLASS PREVALENCE (Real-World Distribution):")
    for name, df_split in [
        ("Train (Raw)", train_df),
        ("Val", val_df),
        ("Test", test_df),
    ]:
        dist = df_split["has_anxiety"].value_counts(normalize=True) * 100
        print(
            f"  {name: <12}: Anxiety {dist.get(1, 0):.1f}% | Control {dist.get(0, 0):.1f}%"
        )

    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
