"""
mimic_extract_v4.py
TC-WPN Multi-Source Data Extraction — Publication-Grade
Author: Dulhara Kaushalya

DATA SOURCES INTEGRATED:
1. MIMIC-IV discharge notes (discharge.csv.gz, discharge_detail.csv.gz)
2. MIMIC-IV prescriptions (prescriptions.csv.gz) — confidence boosting
3. MIMIC-IV OMR (omr.csv.gz) — GAD-7/PHQ gold labels
4. MIMIC-III NOTEEVENTS (NOTEEVENTS.csv.gz) — psychiatric note categories
5. MIMIC-III PRESCRIPTIONS (PRESCRIPTIONS.csv.gz) — confidence boosting
6. MIMIC-III-Ext-Notes (labels.csv) — negation pattern validation

EXPECTED TRAINING CORPUS SIZE (after filtering):
  MIMIC-IV discharge (primary anxiety admissions): ~8,000–12,000 notes
  MIMIC-III psychiatry + social work notes:        ~15,000–25,000 notes
  MIMIC-III nursing/physician for psych patients:  ~20,000–30,000 notes
  Total high-conf anxiety notes:                   ~15,000–20,000
  Total high-conf control notes:                   ~15,000–20,000
  TOTAL:                                           ~30,000–40,000

DIRECTORY STRUCTURE EXPECTED:
  MIMIC_IV_DATASET_PATH/
    hosp/
      patients.csv.gz
      admissions.csv.gz
      diagnoses_icd.csv.gz
      prescriptions.csv.gz     ← NEW
      omr.csv.gz               ← NEW
  MIMIC_IV_NOTE_DATASET_PATH/
    note/
      discharge.csv.gz
      discharge_detail.csv.gz
  MIMIC_III_DATASET_PATH/      ← NEW env var
    NOTEEVENTS.csv.gz
    DIAGNOSES_ICD.csv.gz
    PRESCRIPTIONS.csv.gz
    PATIENTS.csv.gz
    ADMISSIONS.csv.gz
  MIMIC_III_EXT_NOTES_PATH/    ← NEW env var
    labels.csv
    notes.csv

RUN ORDER:
  python -m scripts.mimic_extract_v4
  python -m scripts.convert_csv_to_pkl_v2   (unchanged)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
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

import os

# New paths — add to your .env file
MIMIC_III_DATASET_PATH = os.getenv("MIMIC_III_DATASET_PATH", "")
MIMIC_III_EXT_NOTES_PATH = os.getenv("MIMIC_III_EXT_NOTES_PATH", "")

from src.tc_wpn.data.extraction_v4 import (
    load_csv_safe,
    filter_primary_anxiety_admissions,
    has_psychiatric_content,
    identify_anxiety_patients,
    identify_control_patients,
    load_prescription_confirmed_subjects,
    load_omr_gold_subjects,
    load_ext_notes_negation_patterns,
    compute_temporal_features,
    clean_note_text,
    verify_and_clean_notes,
    assign_anxiety_confidence,
    penalize_control_noise,
    compute_section_quality,
    MIMIC3_HIGH_SIGNAL_CATEGORIES,
)

warnings.filterwarnings("ignore")

MIMIC_IV_PATH = Path(MIMIC_IV_DATASET_PATH)
MIMIC_IV_NOTE_PATH = Path(MIMIC_IV_NOTE_DATASET_PATH)
MIMIC_III_PATH = Path(MIMIC_III_DATASET_PATH) if MIMIC_III_DATASET_PATH else None
EXT_NOTES_PATH = Path(MIMIC_III_EXT_NOTES_PATH) if MIMIC_III_EXT_NOTES_PATH else None
OUTPUT_DIR = Path(MIMIC_PROCESSED_BASE_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# MIMIC-IV NOTE SOURCES
# =============================================================================
MIMIC4_NOTE_SOURCES = [
    {
        "file": "discharge.csv.gz",
        "name": "discharge",
        # Confirmed columns (MIMIC-IV-Note 2.2):
        # note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
        "usecols": ["note_id", "subject_id", "hadm_id", "charttime", "text"],
    },
    # discharge_detail.csv.gz EXCLUDED:
    # Its columns are note_id, subject_id, field_name, field_value, field_ordinal
    # — metadata fields (author, cosigner), NOT clinical note text.
    # Trying to load it with hadm_id/text usecols causes a KeyError.
]


# =============================================================================
# LOAD MIMIC-IV NOTES
# =============================================================================
def load_mimic4_notes(target_hadm_ids):
    all_notes = []
    for source in MIMIC4_NOTE_SOURCES:
        note_path = MIMIC_IV_NOTE_PATH / "note" / source["file"]
        if not note_path.exists():
            print(f"  ⚠  Missing: {source['file']} — skipping.")
            continue
        print(f"\nLoading MIMIC-IV {source['name']} notes...")
        chunks = pd.read_csv(
            note_path,
            usecols=source["usecols"],
            chunksize=50000,
            low_memory=False,
        )
        src_notes = []
        for i, chunk in enumerate(chunks):
            rel = chunk[chunk["hadm_id"].isin(target_hadm_ids)].copy()
            if len(rel) > 0:
                rel["source_type"] = source["name"]
                rel["mimic_version"] = "mimic4"
                src_notes.append(rel)
            if (i + 1) % 10 == 0:
                print(f"    {(i+1)*50000:,} rows processed...")
        if src_notes:
            df = pd.concat(src_notes, ignore_index=True)
            print(f"  ✓ {source['name']}: {len(df):,} notes")
            all_notes.append(df)
    if not all_notes:
        return pd.DataFrame()
    combined = pd.concat(all_notes, ignore_index=True)
    return combined.drop_duplicates(subset=["note_id"])


# =============================================================================
# LOAD MIMIC-III NOTES
# NOTEEVENTS.csv.gz columns:
#   ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME,
#   CATEGORY, DESCRIPTION, CGID, ISERROR, TEXT
# =============================================================================
def load_mimic3_notes(target_subject_ids):
    """
    Loads MIMIC-III NOTEEVENTS filtered to:
    1. High-signal note categories (Psychiatry, Social Work, Physician, Nursing)
    2. Target anxiety/control subject_ids
    3. No error notes (ISERROR is null or 0)

    Note: MIMIC-III uses SUBJECT_ID and HADM_ID (uppercase).
    We normalise to lowercase after loading.
    """
    if MIMIC_III_PATH is None:
        print("  ⚠  MIMIC_III_DATASET_PATH not set — skipping MIMIC-III notes.")
        return pd.DataFrame()

    noteevents_path = MIMIC_III_PATH / "NOTEEVENTS.csv.gz"
    if not noteevents_path.exists():
        # Also try lowercase filename
        noteevents_path = MIMIC_III_PATH / "noteevents.csv.gz"
    if not noteevents_path.exists():
        print(f"  ⚠  NOTEEVENTS.csv.gz not found at {MIMIC_III_PATH} — skipping.")
        return pd.DataFrame()

    print(f"\nLoading MIMIC-III NOTEEVENTS from {noteevents_path}...")

    valid_categories = set(c.lower() for c in MIMIC3_HIGH_SIGNAL_CATEGORIES)

    chunks = pd.read_csv(
        noteevents_path,
        usecols=[
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTTIME",
            "CATEGORY",
            "ISERROR",
            "TEXT",
        ],
        chunksize=50000,
        low_memory=False,
    )

    all_chunks = []
    for i, chunk in enumerate(chunks):
        # Normalise column names to lowercase
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = chunk.rename(
            columns={
                "row_id": "note_id",
                "charttime": "charttime",
                "text": "text",
            }
        )

        # Filter: target subjects, valid category, no errors
        mask_subject = chunk["subject_id"].isin(target_subject_ids)
        mask_category = chunk["category"].astype(str).str.lower().isin(valid_categories)
        mask_error = chunk["iserror"].isna() | (chunk["iserror"] == 0)

        rel = chunk[mask_subject & mask_category & mask_error].copy()
        if len(rel) > 0:
            # Tag source type from category
            rel["source_type"] = (
                rel["category"]
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace("/", "_", regex=False)
            )
            rel["mimic_version"] = "mimic3"
            # Create a unique note_id to avoid collision with MIMIC-IV note_ids
            rel["note_id"] = "M3-" + rel["note_id"].astype(str)
            all_chunks.append(
                rel[
                    [
                        "note_id",
                        "subject_id",
                        "hadm_id",
                        "charttime",
                        "text",
                        "source_type",
                        "mimic_version",
                    ]
                ]
            )
        if (i + 1) % 20 == 0:
            print(f"    {(i+1)*50000:,} MIMIC-III rows processed...")

    if not all_chunks:
        print("  ⚠  No MIMIC-III notes found for target subjects.")
        return pd.DataFrame()

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["note_id"])
    print(f"  ✓ MIMIC-III notes loaded: {len(combined):,}")
    print(
        f"    Category breakdown:\n"
        f"    {combined['source_type'].value_counts().to_string()}"
    )
    return combined


# =============================================================================
# LOAD MIMIC-III DIAGNOSES AND PATIENTS
# Column names in MIMIC-III are UPPERCASE
# =============================================================================
def load_mimic3_cohorts(young_age_min=18, young_age_max=50):
    """
    Returns (anxiety_cases_df, control_cases_df, mimic3_subject_ids)
    using MIMIC-III ICD-9 diagnoses.
    """
    if MIMIC_III_PATH is None:
        return pd.DataFrame(), pd.DataFrame(), set()

    print("\nLoading MIMIC-III hospital tables...")

    # Patients
    pts_path = MIMIC_III_PATH / "PATIENTS.csv.gz"
    if not pts_path.exists():
        pts_path = MIMIC_III_PATH / "patients.csv.gz"
    if not pts_path.exists():
        print("  ⚠  MIMIC-III PATIENTS.csv.gz not found.")
        return pd.DataFrame(), pd.DataFrame(), set()

    patients3 = pd.read_csv(
        pts_path,
        usecols=["SUBJECT_ID", "GENDER", "DOB"],
        compression="gzip",
        low_memory=False,
    )
    patients3.columns = [c.lower() for c in patients3.columns]

    # MIMIC-III DOB is a datetime — anchor_age not available directly
    # Use admissions to compute approximate age
    adm_path = MIMIC_III_PATH / "ADMISSIONS.csv.gz"
    if not adm_path.exists():
        adm_path = MIMIC_III_PATH / "admissions.csv.gz"

    admissions3 = pd.read_csv(
        adm_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
        compression="gzip",
        low_memory=False,
    )
    admissions3.columns = [c.lower() for c in admissions3.columns]
    admissions3["admittime"] = pd.to_datetime(admissions3["admittime"], errors="coerce")
    patients3["dob"] = pd.to_datetime(patients3["dob"], errors="coerce")

    # Approximate age at first admission
    adm_merged = admissions3.merge(
        patients3[["subject_id", "dob"]], on="subject_id", how="left"
    )
    adm_merged["age"] = (
        (adm_merged["admittime"] - adm_merged["dob"]).dt.days / 365.25
    ).round(0)
    young_subjects = set(
        adm_merged[
            (adm_merged["age"] >= young_age_min) & (adm_merged["age"] <= young_age_max)
        ]["subject_id"].unique()
    )
    print(
        f"  ✓ MIMIC-III young patients ({young_age_min}–{young_age_max}): "
        f"{len(young_subjects):,}"
    )

    # Diagnoses
    diag_path = MIMIC_III_PATH / "DIAGNOSES_ICD.csv.gz"
    if not diag_path.exists():
        diag_path = MIMIC_III_PATH / "diagnoses_icd.csv.gz"

    diagnoses3 = pd.read_csv(
        diag_path,
        usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        compression="gzip",
        low_memory=False,
    )
    diagnoses3.columns = [c.lower() for c in diagnoses3.columns]
    diagnoses3 = diagnoses3.rename(
        columns={"icd9_code": "icd_code", "seq_num": "seq_num"}
    )
    diagnoses3["icd_version"] = 9  # All MIMIC-III codes are ICD-9

    # Filter to young patients
    diagnoses3 = diagnoses3[diagnoses3["subject_id"].isin(young_subjects)]

    # Build cohorts
    anxiety3 = identify_anxiety_patients(diagnoses3)
    control3 = identify_control_patients(diagnoses3, anxiety3)

    print(f"  ✓ MIMIC-III anxiety cases: {len(anxiety3):,}")
    print(f"  ✓ MIMIC-III control cases: {len(control3):,}")

    # Add prefix to subject_ids to prevent collision with MIMIC-IV IDs
    # MIMIC-III subject_ids are integers up to ~100k
    # MIMIC-IV subject_ids start at 10000000+ — no actual collision
    # but we tag them for traceability
    anxiety3["mimic_version"] = "mimic3"
    control3["mimic_version"] = "mimic3"

    m3_subjects = set(anxiety3["subject_id"].unique()) | set(
        control3["subject_id"].unique()
    )
    return anxiety3, control3, m3_subjects


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("TC-WPN MULTI-SOURCE DATA EXTRACTION V4 — PUBLICATION GRADE")
    print("Sources: MIMIC-IV + MIMIC-III + Prescriptions + OMR + Ext-Notes")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — LOAD MIMIC-IV HOSPITAL TABLES
    # =========================================================================
    print("\nSTEP 1: Loading MIMIC-IV hospital tables...")

    patients4 = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    if patients4 is None:
        raise FileNotFoundError("Cannot load MIMIC-IV patients.csv.gz")

    admissions4 = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
        ],
    )

    diagnoses4 = load_csv_safe(
        MIMIC_IV_PATH / "hosp" / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )

    patients4 = patients4[
        (patients4["anchor_age"] >= 18) & (patients4["anchor_age"] <= 50)
    ]
    print(f"✓ MIMIC-IV young adult patients (18–50): {len(patients4):,}")

    # =========================================================================
    # STEP 2 — PRESCRIPTION + OMR GOLD LABELS
    # =========================================================================
    print("\nSTEP 2: Loading prescription and OMR gold labels...")

    # MIMIC-IV prescriptions
    rx4_path = MIMIC_IV_PATH / "hosp" / "prescriptions.csv.gz"
    prescription_confirmed_v4 = load_prescription_confirmed_subjects(rx4_path)

    # MIMIC-III prescriptions
    prescription_confirmed_v3 = set()
    if MIMIC_III_PATH:
        rx3_path = MIMIC_III_PATH / "PRESCRIPTIONS.csv.gz"
        if not rx3_path.exists():
            rx3_path = MIMIC_III_PATH / "prescriptions.csv.gz"
        if rx3_path.exists():
            prescription_confirmed_v3 = load_prescription_confirmed_subjects(rx3_path)

    # All prescription-confirmed subjects (both versions)
    all_rx_confirmed = prescription_confirmed_v4 | prescription_confirmed_v3

    # MIMIC-IV OMR gold labels
    omr_path = MIMIC_IV_PATH / "hosp" / "omr.csv.gz"
    omr_gold_subjects = load_omr_gold_subjects(omr_path)

    # =========================================================================
    # STEP 3 — EXT-NOTES NEGATION PATTERNS
    # =========================================================================
    print("\nSTEP 3: Loading Ext-Notes negation patterns...")
    ext_negation_triggers = set()
    if EXT_NOTES_PATH:
        ext_labels_path = EXT_NOTES_PATH / "labels.csv"
        if ext_labels_path.exists():
            ext_negation_triggers = load_ext_notes_negation_patterns(ext_labels_path)
    if not ext_negation_triggers:
        print("  Using default regex-based negation patterns only.")

    # =========================================================================
    # STEP 4 — MIMIC-IV PRIMARY DIAGNOSIS FILTER
    # =========================================================================
    print("\nSTEP 4: Filtering MIMIC-IV to primary anxiety admissions...")
    primary_anxiety_hadm4 = filter_primary_anxiety_admissions(diagnoses4, max_seq_num=3)

    diagnoses4_anxiety = diagnoses4[diagnoses4["hadm_id"].isin(primary_anxiety_hadm4)]
    diagnoses4_controls = diagnoses4.copy()

    # =========================================================================
    # STEP 5 — MIMIC-IV COHORTS
    # =========================================================================
    print("\nSTEP 5: Building MIMIC-IV cohorts...")
    young_ids4 = set(patients4["subject_id"])

    anxiety4 = identify_anxiety_patients(diagnoses4_anxiety)
    control4 = identify_control_patients(diagnoses4_controls, anxiety4)

    anxiety4 = anxiety4[anxiety4["subject_id"].isin(young_ids4)]
    control4 = control4[control4["subject_id"].isin(young_ids4)]
    anxiety4["mimic_version"] = "mimic4"
    control4["mimic_version"] = "mimic4"

    print(f"✓ MIMIC-IV anxiety: {len(anxiety4):,}")
    print(f"✓ MIMIC-IV control: {len(control4):,}")

    # =========================================================================
    # STEP 6 — MIMIC-III COHORTS
    # =========================================================================
    print("\nSTEP 6: Loading MIMIC-III cohorts...")
    anxiety3, control3, m3_subjects = load_mimic3_cohorts(18, 50)

    # =========================================================================
    # STEP 7 — COMBINE COHORTS
    # =========================================================================
    print("\nSTEP 7: Combining cohorts...")
    all_anxiety = (
        pd.concat([anxiety4, anxiety3], ignore_index=True)
        if not anxiety3.empty
        else anxiety4
    )
    all_control = (
        pd.concat([control4, control3], ignore_index=True)
        if not control3.empty
        else control4
    )

    all_cases = pd.concat([all_anxiety, all_control], ignore_index=True)

    print(f"✓ Total anxiety cases: {len(all_anxiety):,}")
    print(f"✓ Total control cases: {len(all_control):,}")

    # =========================================================================
    # STEP 8 — LOAD NOTES FROM ALL SOURCES
    # =========================================================================
    print("\nSTEP 8: Loading clinical notes...")

    # MIMIC-IV: load notes for ALL anxiety + sampled control subjects
    # We load by subject_id (not hadm_id) so ALL notes for a patient are
    # included, not just notes from the specific anxiety admission.
    anxiety_subjects4 = set(anxiety4["subject_id"].unique())
    control_subjects4 = set(control4["subject_id"].unique())
    # Sample controls to cap at 5× anxiety subjects (avoids 90k+ control notes)
    max_ctrl4 = min(len(control_subjects4), len(anxiety_subjects4) * 5)
    rng = np.random.default_rng(42)
    sampled_ctrl4 = set(
        rng.choice(list(control_subjects4), size=max_ctrl4, replace=False).tolist()
    )
    target_subjects4 = anxiety_subjects4 | sampled_ctrl4
    print(
        f"  MIMIC-IV target: {len(anxiety_subjects4):,} anxiety + "
        f"{len(sampled_ctrl4):,} control subjects"
    )

    # Load by subject_id — discharge.csv.gz has hadm_id so we filter by
    # the hadm_ids that belong to our target subjects
    target_hadm4 = set(
        all_cases[
            (all_cases["mimic_version"] == "mimic4")
            & (all_cases["subject_id"].isin(target_subjects4))
        ]["hadm_id"].unique()
    )
    notes4 = load_mimic4_notes(target_hadm4)
    print(f"✓ MIMIC-IV notes: {len(notes4):,}")

    # MIMIC-III notes
    target_subjects3 = set(
        all_cases[all_cases["mimic_version"] == "mimic3"]["subject_id"].unique()
    )
    notes3 = load_mimic3_notes(target_subjects3)

    # Standardise column name
    if not notes4.empty and "text" in notes4.columns:
        notes4 = notes4.rename(columns={"text": "clinical_note_text"})
    if not notes3.empty and "text" in notes3.columns:
        notes3 = notes3.rename(columns={"text": "clinical_note_text"})
        if "hadm_id" not in notes3.columns:
            notes3["hadm_id"] = np.nan

    all_notes_list = []
    if not notes4.empty:
        all_notes_list.append(
            notes4[
                [
                    "note_id",
                    "subject_id",
                    "hadm_id",
                    "charttime",
                    "clinical_note_text",
                    "source_type",
                    "mimic_version",
                ]
            ]
        )
    if not notes3.empty:
        all_notes_list.append(
            notes3[
                [
                    "note_id",
                    "subject_id",
                    "hadm_id",
                    "charttime",
                    "clinical_note_text",
                    "source_type",
                    "mimic_version",
                ]
            ]
        )

    if not all_notes_list:
        raise ValueError("No notes loaded from any source.")

    notes_combined = pd.concat(all_notes_list, ignore_index=True)
    print(f"\n✓ Total notes from all sources: {len(notes_combined):,}")

    # =========================================================================
    # STEP 9 — MERGE LABELS
    # Merge on subject_id ONLY for both versions.
    # The label (anxiety/control) is a patient-level property:
    # - An anxiety patient's every note is labeled 1
    # - A control patient's every note is labeled 0
    # Using hadm_id caused most MIMIC-IV anxiety notes to be dropped
    # (notes from non-anxiety admissions of anxiety patients got NaN).
    # =========================================================================
    print("\nSTEP 9: Merging labels...")

    # Build a clean subject_id → has_anxiety lookup (one row per subject)
    label_lookup = pd.concat(
        [
            all_anxiety[["subject_id", "has_anxiety", "mimic_version"]],
            all_control[["subject_id", "has_anxiety", "mimic_version"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["subject_id"])

    merged = notes_combined.merge(
        label_lookup[["subject_id", "has_anxiety"]],
        on="subject_id",
        how="inner",  # inner: drop notes with no matching label
    )

    merged["has_anxiety"] = merged["has_anxiety"].astype(int)

    # Add demographics for MIMIC-IV patients
    merged = merged.merge(
        patients4[["subject_id", "gender", "anchor_age"]].rename(
            columns={"anchor_age": "age_at_admission"}
        ),
        on="subject_id",
        how="left",
    )

    print(f"✓ Merged notes: {len(merged):,}")
    print(f"  Anxiety: {(merged['has_anxiety']==1).sum():,}")
    print(f"  Control: {(merged['has_anxiety']==0).sum():,}")

    # =========================================================================
    # STEP 10 — CLEAN NOTES
    # =========================================================================
    print("\nSTEP 10: Cleaning notes...")
    merged["clinical_note_text"] = merged["clinical_note_text"].apply(clean_note_text)
    merged = verify_and_clean_notes(merged)
    merged["note_length"] = merged["clinical_note_text"].str.len()
    print(f"✓ Post-clean: {len(merged):,}")

    # =========================================================================
    # STEP 11 — PSYCHIATRIC CONTENT GATE
    # Apply to ANXIETY notes only — removes notes with no anxiety signal.
    # Controls are NOT filtered: a clean control with no psychiatric keywords
    # is exactly what we want as a negative example.
    # MIMIC-III psychiatry/social_work: min 1 keyword (category already guarantees signal)
    # All other sources: min 2 keywords
    # =========================================================================
    print("\nSTEP 11: Applying psychiatric content gate to anxiety notes only...")
    anxiety_mask = merged["has_anxiety"] == 1
    control_mask = merged["has_anxiety"] == 0
    n_before = anxiety_mask.sum()

    def gate_anxiety_note(row):
        src = str(row.get("source_type", "")).lower()
        min_kw = 1 if src in ("psychiatry", "social_work") else 2
        return has_psychiatric_content(row["clinical_note_text"], min_keywords=min_kw)

    # Only compute gate for anxiety notes (controls always pass)
    anx_rows = merged[anxiety_mask].copy()
    anx_has_content = anx_rows.apply(gate_anxiety_note, axis=1)
    anx_kept = anx_rows[anx_has_content]

    # Cap controls at 4× kept anxiety notes to maintain useful ratio
    ctrl_rows = merged[control_mask].copy()
    max_ctrl = min(len(ctrl_rows), len(anx_kept) * 4)
    ctrl_kept = ctrl_rows.sample(n=max_ctrl, random_state=42)

    merged = pd.concat([anx_kept, ctrl_kept], ignore_index=True)
    n_after = (merged["has_anxiety"] == 1).sum()
    n_ctrl = (merged["has_anxiety"] == 0).sum()
    print(
        f"  Anxiety notes: {n_before:,} → {n_after:,} "
        f"(removed {n_before - n_after:,} low-signal)"
    )
    print(f"  Control notes kept: {n_ctrl:,} (4× anxiety)")
    print(f"✓ Post-gate: {len(merged):,}")

    # =========================================================================
    # STEP 12 — TEMPORAL FEATURES
    # =========================================================================
    print("\nSTEP 12: Computing temporal features...")
    merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")
    merged = merged[merged["charttime"].notna()]

    temporal_df = compute_temporal_features(merged)
    merged = merged.merge(
        temporal_df,
        on=["note_id", "subject_id", "charttime"],
        how="left",
    )

    # =========================================================================
    # STEP 13 — LABEL CONFIDENCE + SECTION QUALITY
    # =========================================================================
    print("\nSTEP 13: Assigning label confidence + section quality...")

    merged["label_confidence"] = 1.0
    merged["anxiety_context"] = "control"

    anx_mask = merged["has_anxiety"] == 1
    ctrl_mask = merged["has_anxiety"] == 0

    # Anxiety confidence — v4: passes subject_id for prescription/OMR boosting
    def get_confidence(row):
        return assign_anxiety_confidence(
            row["clinical_note_text"],
            subject_id=row["subject_id"],
            prescription_confirmed_subjects=all_rx_confirmed,
            omr_gold_subjects=omr_gold_subjects,
        )

    anxiety_results = merged[anx_mask].apply(get_confidence, axis=1)
    merged.loc[anx_mask, "label_confidence"] = [x[0] for x in anxiety_results]
    merged.loc[anx_mask, "anxiety_context"] = [x[1] for x in anxiety_results]

    # Control confidence
    merged.loc[ctrl_mask, "label_confidence"] = merged.loc[
        ctrl_mask, "clinical_note_text"
    ].apply(penalize_control_noise)

    # Section quality — pass source_type for priority bonus
    merged["section_quality"] = merged.apply(
        lambda row: compute_section_quality(
            row["clinical_note_text"], row.get("source_type", "unknown")
        ),
        axis=1,
    )

    merged["training_weight"] = merged["label_confidence"] * merged["section_quality"]
    merged["has_text_signal"] = merged["label_confidence"] > 0.5

    # Diagnostic
    print("\n--- LABEL QUALITY DIAGNOSTIC ---")
    ctrl_df = merged[merged["has_anxiety"] == 0]
    contam = ctrl_df[ctrl_df["label_confidence"] < 0.9]
    print(
        f"Control contamination: {len(contam):,} / {len(ctrl_df):,} "
        f"({100*len(contam)/max(len(ctrl_df),1):.1f}%)"
    )
    anx_ctx = merged[merged["has_anxiety"] == 1]["anxiety_context"].value_counts()
    print(f"Anxiety context breakdown:\n{anx_ctx.to_string()}")
    src_breakdown = merged.groupby(["mimic_version", "source_type"]).size()
    print(f"Source breakdown:\n{src_breakdown.to_string()}")
    print("--------------------------------")

    # =========================================================================
    # STEP 14 — PATIENT-LEVEL SPLIT (leakage-safe)
    # =========================================================================
    print("\nSTEP 14: Patient-level leakage-safe split...")
    merged = merged.sort_values(["subject_id", "charttime"]).reset_index(drop=True)

    unique_patients = merged["subject_id"].dropna().unique()
    train_val_ids, test_ids = train_test_split(
        unique_patients, test_size=0.10, random_state=42, shuffle=True
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.1111, random_state=42, shuffle=True
    )

    merged["dataset_split"] = "train"
    merged.loc[merged["subject_id"].isin(val_ids), "dataset_split"] = "val"
    merged.loc[merged["subject_id"].isin(test_ids), "dataset_split"] = "test"

    # =========================================================================
    # STEP 15 — SAVE DATASETS
    # =========================================================================
    print("\nSTEP 15: Saving datasets...")

    output_cols = [
        "note_id",
        "subject_id",
        "hadm_id",
        "charttime",
        "source_type",
        "mimic_version",
        "has_anxiety",
        "has_text_signal",
        "label_confidence",
        "section_quality",
        "training_weight",
        "anxiety_context",
        "dataset_split",
        "gender",
        "age_at_admission",
        "days_since_first_visit",
        "days_since_last_visit",
        "visit_number",
        "total_visits",
        "note_age_days",
        "is_most_recent",
        "note_length",
        "clinical_note_text",
    ]
    # Only keep columns that exist (age_at_admission may be NaN for MIMIC-III subjects)
    output_cols = [c for c in output_cols if c in merged.columns]
    final = merged[output_cols].copy()

    train_df = final[final["dataset_split"] == "train"]
    val_df = final[final["dataset_split"] == "val"]
    test_df = final[final["dataset_split"] == "test"]

    # Real-world splits
    train_df.to_csv(OUTPUT_DIR / "mimic_anxiety_train_real_world.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "mimic_anxiety_val_real_world.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "mimic_anxiety_test_real_world.csv", index=False)

    # Balanced train
    t_anx = train_df[train_df["has_anxiety"] == 1]
    t_ctrl = train_df[train_df["has_anxiety"] == 0]
    n = min(len(t_anx), len(t_ctrl))
    train_balanced = pd.concat(
        [
            t_anx.sample(n=n, random_state=42),
            t_ctrl.sample(n=n, random_state=42),
        ]
    ).sample(frac=1, random_state=42)
    train_balanced.to_csv(OUTPUT_DIR / "mimic_anxiety_train_balanced.csv", index=False)

    # High-confidence train
    train_hc = train_balanced[
        (
            (train_balanced["has_anxiety"] == 1)
            & (train_balanced["label_confidence"] >= 0.75)
        )
        | (
            (train_balanced["has_anxiety"] == 0)
            & (train_balanced["label_confidence"] >= 0.9)
        )
    ]
    train_hc.to_csv(OUTPUT_DIR / "mimic_anxiety_train_high_conf.csv", index=False)

    # High-confidence test
    test_hc = test_df[
        ((test_df["has_anxiety"] == 1) & (test_df["label_confidence"] >= 0.7))
        | ((test_df["has_anxiety"] == 0) & (test_df["label_confidence"] >= 0.9))
    ]
    test_hc.to_csv(OUTPUT_DIR / "mimic_anxiety_test_high_conf.csv", index=False)

    # =========================================================================
    # STEP 16 — LEAKAGE CHECKS
    # =========================================================================
    print("\nSTEP 16: Leakage checks...")
    print(f"  Train↔Val  : {len(set(train_ids) & set(val_ids))}")
    print(f"  Train↔Test : {len(set(train_ids) & set(test_ids))}")
    print(f"  Val↔Test   : {len(set(val_ids)   & set(test_ids))}")

    if not any(
        [
            len(set(train_ids) & set(val_ids)),
            len(set(train_ids) & set(test_ids)),
            len(set(val_ids) & set(test_ids)),
        ]
    ):
        print("  ✅ ZERO PATIENT LEAKAGE CONFIRMED")
    else:
        print("  ❌ LEAKAGE DETECTED")

    print("\n" + "=" * 80)
    print("✅ EXTRACTION V4 COMPLETE")
    print("=" * 80)
    print(f"Total notes          : {len(final):,}")
    print(f"  Anxiety            : {(final['has_anxiety']==1).sum():,}")
    print(f"  Control            : {(final['has_anxiety']==0).sum():,}")
    print(f"Train (raw)          : {len(train_df):,}")
    print(f"Val                  : {len(val_df):,}")
    print(f"Test                 : {len(test_df):,}")
    print(f"Balanced Train       : {len(train_balanced):,}")
    print(f"High-Conf Train      : {len(train_hc):,}")
    print(f"High-Conf Test       : {len(test_hc):,}")
    print(f"\nSource × version breakdown:")
    if "mimic_version" in final.columns and "source_type" in final.columns:
        print(
            final.groupby(["mimic_version", "source_type", "has_anxiety"])
            .size()
            .rename("count")
            .to_string()
        )
    print(f"\nAnxiety confidence breakdown (balanced train):")
    print(
        train_balanced[train_balanced["has_anxiety"] == 1]["anxiety_context"]
        .value_counts()
        .to_string()
    )
    print(f"\nHigh-Conf Train anxiety / control split:")
    print(train_hc["has_anxiety"].value_counts().to_string())
    print(f"\n✅ Ready for: python -m scripts.convert_csv_to_pkl_v2")


if __name__ == "__main__":
    main()
