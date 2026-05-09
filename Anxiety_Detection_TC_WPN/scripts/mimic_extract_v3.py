"""
mimic_extract_v3.py
MIMIC-IV Data Extraction V3 — Publication-Grade
TC-WPN Research Pipeline
Author: Dulhara Kaushalya

KEY CHANGES FROM mimic_extract_v2.py:
1. filter_primary_anxiety_admissions() applied BEFORE building cohorts.
   Only admissions where anxiety is a top-3 diagnosis are included.
   This eliminates notes from ICU/surgery patients with incidental
   anxiety ICD codes — the single biggest cause of label noise.

2. has_psychiatric_content() gate applied to all anxiety notes.
   Notes without ≥2 psychiatric keywords are EXCLUDED even if they
   have an anxiety ICD code. Eliminates ~40-60% of low-signal notes.

3. Age filter tightened: 18-50 (was 18-65).
   Reduces comorbidity noise — elderly patients have many confounding
   diagnoses that contaminate discharge summary text.

4. Imports from extraction_v3.py (not extraction_v2.py).

RUN ORDER:
  python -m scripts.mimic_extract_v3
  python -m scripts.convert_csv_to_pkl_v2   (unchanged — reuse v2)

THEN verify with validation.py before uploading PKLs to Kaggle.
"""

import sys
from pathlib import Path
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MIMIC_IV_DATASET_PATH,
    MIMIC_IV_NOTE_DATASET_PATH,
    MIMIC_PROCESSED_BASE_DIR,
)

# v3: import from extraction_v3
from src.tc_wpn.data.extraction_v3 import (
    load_csv_safe,
    filter_primary_anxiety_admissions,  # NEW v3
    has_psychiatric_content,  # NEW v3
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

MIMIC_IV_PATH      = Path(MIMIC_IV_DATASET_PATH)
MIMIC_IV_NOTE_PATH = Path(MIMIC_IV_NOTE_DATASET_PATH)
OUTPUT_DIR         = Path(MIMIC_PROCESSED_BASE_DIR)
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
    {
        "file": "discharge_detail.csv.gz",
        "name": "discharge_detail",
        "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    },
    # Uncomment if available — physician notes have denser psychiatric content:
    # {
    #     "file": "physician.csv.gz",
    #     "name": "physician",
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
            note_path, usecols=source["usecols"],
            chunksize=50000, low_memory=False,
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
    combined = combined.drop_duplicates(subset=["note_id"])
    return combined


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("MIMIC-IV DATA EXTRACTION V3 — PUBLICATION GRADE")
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
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "hospital_expire_flag"],
    )

    diagnoses = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )

    # v3: tightened to 18-50 to reduce elderly comorbidity noise
    patients = patients[
        (patients["anchor_age"] >= 18) & (patients["anchor_age"] <= 50)
    ]
    print(f"✓ Young adult patients (18–50): {len(patients):,}")

    # =========================================================================
    # STEP 2 — PRIMARY DIAGNOSIS FILTER  (v3 NEW)
    # =========================================================================
    print("\nSTEP 2: Filtering to primary anxiety admissions (seq_num ≤ 3)...")
    primary_anxiety_hadm = filter_primary_anxiety_admissions(
        diagnoses, max_seq_num=3
    )

    # Filter diagnoses table to only those admissions for anxiety cohort
    diagnoses_primary = diagnoses[
        diagnoses["hadm_id"].isin(primary_anxiety_hadm)
        | (diagnoses["hadm_id"].isin(
            diagnoses[~diagnoses["hadm_id"].isin(primary_anxiety_hadm)]["hadm_id"]
        ))
    ]
    # For controls, keep ALL admissions (no primary filter needed — they have no anxiety)
    # For anxiety, restrict to primary_anxiety_hadm
    diagnoses_for_anxiety = diagnoses[diagnoses["hadm_id"].isin(primary_anxiety_hadm)]
    diagnoses_for_controls = diagnoses.copy()

    # =========================================================================
    # STEP 3 — BUILD COHORTS
    # =========================================================================
    print("\nSTEP 3: Building anxiety and control cohorts...")

    anxiety_cases = identify_anxiety_patients(diagnoses_for_anxiety)
    control_cases = identify_control_patients(diagnoses_for_controls, anxiety_cases)

    young_ids = set(patients["subject_id"])
    anxiety_cases = anxiety_cases[anxiety_cases["subject_id"].isin(young_ids)]
    control_cases = control_cases[control_cases["subject_id"].isin(young_ids)]

    all_cases = pd.concat([anxiety_cases, control_cases], ignore_index=True)

    print(f"✓ Anxiety cases (primary, young adult): {len(anxiety_cases):,}")
    print(f"✓ Control cases (psych-clean, young adult): {len(control_cases):,}")

    # =========================================================================
    # STEP 4 — LOAD NOTES
    # =========================================================================
    print("\nSTEP 4: Loading clinical notes...")
    target_hadm_ids = set(all_cases["hadm_id"].unique())
    notes = load_relevant_notes(target_hadm_ids)
    print(f"\n✓ Total loaded notes (all sources): {len(notes):,}")

    # =========================================================================
    # STEP 5 — MERGE AND CLEAN
    # =========================================================================
    print("\nSTEP 5: Merging and cleaning...")

    notes = notes.rename(columns={"text": "clinical_note_text"})
    merged = notes.merge(all_cases, on=["subject_id", "hadm_id"], how="left")
    merged = merged[merged["has_anxiety"].notna()].copy()
    merged["has_anxiety"] = merged["has_anxiety"].astype(int)

    merged = merged.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id", how="left",
    )

    merged["clinical_note_text"] = merged["clinical_note_text"].apply(clean_note_text)
    merged = verify_and_clean_notes(merged)
    merged["note_length"] = merged["clinical_note_text"].str.len()

    print(f"✓ Post-clean notes: {len(merged):,}")

    # =========================================================================
    # STEP 6 — PSYCHIATRIC CONTENT GATE  (v3 NEW)
    #
    # Apply to ANXIETY notes only. Require ≥2 psychiatric keywords.
    # This removes discharge summaries for e.g. liver cirrhosis admissions
    # where anxiety appears as a secondary ICD code but the note has no
    # meaningful anxiety content.
    # Controls are NOT filtered here — clean controls intentionally have
    # few psychiatric keywords (that's what makes them controls).
    # =========================================================================
    print("\nSTEP 6: Applying psychiatric content gate to anxiety notes...")

    anxiety_mask = merged["has_anxiety"] == 1
    control_mask = merged["has_anxiety"] == 0

    n_anxiety_before = anxiety_mask.sum()

    # Apply gate: keep anxiety notes with ≥2 psychiatric keywords
    has_content = merged["clinical_note_text"].apply(
        lambda t: has_psychiatric_content(t, min_keywords=2)
    )
    # Keep: (anxiety AND has content) OR control
    keep_mask = (anxiety_mask & has_content) | control_mask
    merged = merged[keep_mask].copy()

    n_anxiety_after = (merged["has_anxiety"] == 1).sum()
    n_dropped = n_anxiety_before - n_anxiety_after
    print(f"  Anxiety notes before gate: {n_anxiety_before:,}")
    print(f"  Anxiety notes after  gate: {n_anxiety_after:,}")
    print(f"  Low-signal notes removed:  {n_dropped:,} "
          f"({100 * n_dropped / max(n_anxiety_before, 1):.1f}%)")

    print(f"✓ Post-gate total notes: {len(merged):,}")

    # =========================================================================
    # STEP 7 — TEMPORAL FEATURES
    # =========================================================================
    print("\nSTEP 7: Computing temporal features...")

    merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")
    merged = merged[merged["charttime"].notna()]

    temporal_df = compute_temporal_features(merged)
    final_df = merged.merge(
        temporal_df, on=["note_id", "subject_id", "charttime"], how="left",
    )

    # =========================================================================
    # STEP 8 — LABEL CONFIDENCE + SECTION QUALITY
    # =========================================================================
    print("\nSTEP 8: Assigning label confidence + section quality...")

    final_df["label_confidence"] = 1.0
    final_df["anxiety_context"]  = "control"

    anxiety_mask_final = final_df["has_anxiety"] == 1
    control_mask_final = final_df["has_anxiety"] == 0

    anxiety_results = final_df.loc[anxiety_mask_final, "clinical_note_text"].apply(
        assign_anxiety_confidence
    )
    final_df.loc[anxiety_mask_final, "label_confidence"] = [x[0] for x in anxiety_results]
    final_df.loc[anxiety_mask_final, "anxiety_context"]  = [x[1] for x in anxiety_results]

    final_df.loc[control_mask_final, "label_confidence"] = final_df.loc[
        control_mask_final, "clinical_note_text"
    ].apply(penalize_control_noise)

    final_df["section_quality"] = final_df["clinical_note_text"].apply(
        compute_section_quality
    )
    final_df["training_weight"] = (
        final_df["label_confidence"] * final_df["section_quality"]
    )
    final_df["has_text_signal"] = final_df["label_confidence"] > 0.5

    # Diagnostic print
    print("\n--- LABEL QUALITY DIAGNOSTIC ---")
    ctrl_df = final_df[final_df["has_anxiety"] == 0]
    contam  = ctrl_df[ctrl_df["label_confidence"] < 0.9]
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
    # STEP 9 — PATIENT-LEVEL SPLIT (leakage-safe)
    # =========================================================================
    print("\nSTEP 9: Patient-level leakage-safe split...")

    final_df = final_df.sort_values(
        by=["subject_id", "charttime"]
    ).reset_index(drop=True)

    unique_patients = final_df["subject_id"].dropna().unique()
    train_val_ids, test_ids = train_test_split(
        unique_patients, test_size=0.10, random_state=42, shuffle=True
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.1111, random_state=42, shuffle=True
    )

    final_df["dataset_split"] = "train"
    final_df.loc[final_df["subject_id"].isin(val_ids),  "dataset_split"] = "val"
    final_df.loc[final_df["subject_id"].isin(test_ids), "dataset_split"] = "test"

    # =========================================================================
    # STEP 10 — SAVE DATASETS
    # =========================================================================
    print("\nSTEP 10: Saving datasets...")

    output_columns = [
        "note_id", "subject_id", "hadm_id", "charttime", "source_type",
        "has_anxiety", "has_text_signal", "label_confidence", "section_quality",
        "training_weight", "anxiety_context", "dataset_split",
        "gender", "anchor_age",
        "days_since_first_visit", "days_since_last_visit",
        "visit_number", "total_visits", "note_age_days", "is_most_recent",
        "note_length", "clinical_note_text",
    ]

    final_dataset = final_df[output_columns].copy()
    final_dataset.rename(columns={"anchor_age": "age_at_admission"}, inplace=True)

    train_df = final_dataset[final_dataset["dataset_split"] == "train"]
    val_df   = final_dataset[final_dataset["dataset_split"] == "val"]
    test_df  = final_dataset[final_dataset["dataset_split"] == "test"]

    # Real-world splits
    train_df.to_csv(OUTPUT_DIR / "mimic_anxiety_train_real_world.csv", index=False)
    val_df.to_csv(  OUTPUT_DIR / "mimic_anxiety_val_real_world.csv",   index=False)
    test_df.to_csv( OUTPUT_DIR / "mimic_anxiety_test_real_world.csv",  index=False)

    # Balanced train
    train_anx  = train_df[train_df["has_anxiety"] == 1]
    train_ctrl = train_df[train_df["has_anxiety"] == 0]
    n_samples  = min(len(train_anx), len(train_ctrl))
    train_balanced = pd.concat([
        train_anx.sample(n=n_samples, random_state=42),
        train_ctrl.sample(n=n_samples, random_state=42),
    ]).sample(frac=1, random_state=42)
    train_balanced.to_csv(OUTPUT_DIR / "mimic_anxiety_train_balanced.csv", index=False)

    # High-confidence train — v3: raised confidence thresholds slightly
    train_high_conf = train_balanced[
        ((train_balanced["has_anxiety"] == 1)
         & (train_balanced["label_confidence"] >= 0.75))
        | ((train_balanced["has_anxiety"] == 0)
           & (train_balanced["label_confidence"] >= 0.9))
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
    # STEP 11 — LEAKAGE CHECKS
    # =========================================================================
    print("\nSTEP 11: Leakage checks...")
    leak_tv = len(set(train_ids) & set(val_ids))
    leak_tt = len(set(train_ids) & set(test_ids))
    leak_vt = len(set(val_ids)   & set(test_ids))

    print(f"  Leak Train↔Val  : {leak_tv}")
    print(f"  Leak Train↔Test : {leak_tt}")
    print(f"  Leak Val↔Test   : {leak_vt}")

    if leak_tv == 0 and leak_tt == 0 and leak_vt == 0:
        print("  ✅ ZERO PATIENT LEAKAGE CONFIRMED")
    else:
        print("  ❌ LEAKAGE DETECTED — check train_test_split logic")

    # Final summary
    print("\n" + "=" * 80)
    print("✅ EXTRACTION V3 COMPLETE")
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
    print(f"\nAnxiety / Control split (balanced train):")
    print(train_balanced["has_anxiety"].value_counts().to_string())


if __name__ == "__main__":
    main()
