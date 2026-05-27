"""
convert_csv_to_pkl_v3.py
Publication-Grade CSV → PKL Conversion for TC-WPN
Author: Dulhara Kaushalya

CHANGES FROM convert_csv_to_pkl_v2.py:
=======================================
1. DEDUPLICATION CHECK added before saving.
   Verifies zero duplicate note_ids in each PKL.
   Raises ValueError if duplicates are found — prevents bad PKL from
   being uploaded to Kaggle silently.

2. PREVALENCE STATS reported for every file.
   Anxiety%, control%, and patient counts are printed and saved to a
   summary CSV. This is required for the methods section of the paper.

3. NOTE: This script NEVER resamples records.
   The old fix_val_pkl.py upsampled test sets. That is now gone.
   This script converts CSV → PKL faithfully, one record per row.
   Rebalancing for the train set only happens inside the episodic
   dataset (curriculum filtering), never at the PKL level.
"""

import pandas as pd
import pickle
import csv
from pathlib import Path
import os
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer

# =============================================================================
# ENV CONFIG
# =============================================================================
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# PROJECT CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "mimic_processed"
PKL_DIR = PROJECT_ROOT / "mimic_pkl"
PKL_DIR.mkdir(exist_ok=True)

# =============================================================================
# TARGET DATASETS
# Note: NO upsampled versions here.
# fix_val_pkl.py creates balanced_supp variants by DOWN-sampling controls.
# =============================================================================
TARGET_FILES = [
    "mimic_anxiety_train_balanced.csv",
    "mimic_anxiety_train_high_conf.csv",
    "mimic_anxiety_train_real_world.csv",
    "mimic_anxiety_val_real_world.csv",
    "mimic_anxiety_test_real_world.csv",
    "mimic_anxiety_test_high_conf.csv",
]

# =============================================================================
# TOKENIZER CONFIG
# =============================================================================
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 512
WINDOW_OVERLAP = 128
STRIDE = MAX_LENGTH - WINDOW_OVERLAP - 2


# =============================================================================
# TOKENIZATION ENGINE
# =============================================================================
def sliding_window_tokenize(text: str, tokenizer):
    if not isinstance(text, str) or not text.strip():
        text = "empty note"

    raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    if len(raw_ids) <= STRIDE:
        enc = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": [enc["input_ids"]],
            "attention_mask": [enc["attention_mask"]],
            "n_chunks": 1,
            "raw_token_count": len(raw_ids),
        }

    chunk_ids, chunk_masks = [], []
    for start in range(0, len(raw_ids), STRIDE):
        chunk = raw_ids[start : start + STRIDE]
        if not chunk:
            continue
        chunk_text = tokenizer.decode(
            chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        enc = tokenizer(
            chunk_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )
        chunk_ids.append(enc["input_ids"])
        chunk_masks.append(enc["attention_mask"])
        if start + STRIDE >= len(raw_ids):
            break

    return {
        "input_ids": chunk_ids,
        "attention_mask": chunk_masks,
        "n_chunks": len(chunk_ids),
        "raw_token_count": len(raw_ids),
    }


# =============================================================================
# SAFE TYPE HELPERS
# =============================================================================
def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except:
        return default


def safe_str(value, default=""):
    try:
        if pd.isna(value):
            return default
        return str(value)
    except:
        return default


# =============================================================================
# DEDUPLICATION VERIFICATION
# =============================================================================
def verify_no_duplicates(records, filename):
    """
    Verifies no duplicate note_ids exist in the generated PKL.
    Also verifies no duplicate (subject_id, charttime) pairs.
    Raises ValueError if duplicates found — fail loud, not silently.
    """
    note_ids = [r["note_id"] for r in records]
    dup_count = len(note_ids) - len(set(note_ids))

    if dup_count > 0:
        dup_examples = [nid for nid in set(note_ids) if note_ids.count(nid) > 1][:5]
        raise ValueError(
            f"DEDUPLICATION FAILURE in {filename}: "
            f"{dup_count} duplicate note_ids found. "
            f"Examples: {dup_examples}. "
            f"Check CSV source for duplicate rows."
        )

    print(f"  ✅ Deduplication check passed: 0 duplicate note_ids")


# =============================================================================
# PREVALENCE AND PATIENT STATS
# =============================================================================
def compute_stats(records, filename):
    anxiety = [r for r in records if r["label"] == 1]
    control = [r for r in records if r["label"] == 0]

    unique_anxiety_patients = len(set(r["subject_id"] for r in anxiety))
    unique_control_patients = len(set(r["subject_id"] for r in control))
    total_patients = len(set(r["subject_id"] for r in records))

    prevalence = len(anxiety) / max(len(records), 1)
    avg_weight = sum(r["weight"] for r in records) / max(len(records), 1)
    avg_chunks = sum(r["n_chunks"] for r in records) / max(len(records), 1)

    stats = {
        "filename": filename,
        "total_records": len(records),
        "anxiety_records": len(anxiety),
        "control_records": len(control),
        "prevalence_pct": round(100 * prevalence, 2),
        "total_patients": total_patients,
        "anxiety_patients": unique_anxiety_patients,
        "control_patients": unique_control_patients,
        "avg_weight": round(avg_weight, 4),
        "avg_chunks": round(avg_chunks, 3),
    }
    return stats


# =============================================================================
# CSV → PKL PIPELINE
# =============================================================================
def convert_csv_to_pkl():
    print("=" * 90)
    print("PUBLICATION-GRADE MIMIC CSV → PKL CONVERSION (v3)")
    print("No resampling. No duplicates. Prevalence reported for methods section.")
    print("=" * 90)

    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    all_stats = []

    for filename in TARGET_FILES:
        csv_path = PROCESSED_DIR / filename

        if not csv_path.exists():
            print(f"\n⚠ Missing file: {filename} — skipped.")
            continue

        print("\n" + "-" * 90)
        print(f"Processing: {filename}")
        print("-" * 90)

        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded rows: {len(df):,}")

        # Check for source-level duplicates and warn
        n_dup_source = df.duplicated(subset=["note_id"]).sum()
        if n_dup_source > 0:
            print(
                f"  ⚠  {n_dup_source} duplicate note_ids in CSV — dropping duplicates."
            )
            df = df.drop_duplicates(subset=["note_id"])
            print(f"  After dedup: {len(df):,} rows")

        dataset_records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {filename}"):
            text = safe_str(row.get("clinical_note_text", ""))
            token_data = sliding_window_tokenize(text, tokenizer)

            record = {
                # Core IDs
                "note_id": safe_str(row.get("note_id", "unknown")),
                "subject_id": safe_str(row.get("subject_id", "unknown")),
                "hadm_id": safe_str(row.get("hadm_id", "unknown")),
                # Labels
                "label": safe_int(row.get("has_anxiety", 0)),
                "label_confidence": safe_float(row.get("label_confidence", 1.0)),
                "section_quality": safe_float(row.get("section_quality", 1.0)),
                "weight": safe_float(row.get("training_weight", 1.0)),
                # Domain / context
                "source_type": safe_str(row.get("source_type", "unknown")),
                "anxiety_context": safe_str(row.get("anxiety_context", "unspecified")),
                "dataset_split": safe_str(row.get("dataset_split", "unknown")),
                # Demographics
                "gender": safe_str(row.get("gender", "unknown")),
                "age_at_admission": safe_float(row.get("age_at_admission", 0.0)),
                # Temporal
                "note_timestamp": safe_str(row.get("charttime", "")),
                "visit_number": safe_int(row.get("visit_number", 1)),
                "days_since_first_visit": safe_float(
                    row.get("days_since_first_visit", 0.0)
                ),
                "days_since_last_visit": safe_float(
                    row.get("days_since_last_visit", 0.0)
                ),
                "total_visits": safe_int(row.get("total_visits", 1)),
                "note_age_days": safe_float(row.get("note_age_days", 0.0)),
                "is_most_recent": bool(row.get("is_most_recent", False)),
                # Note features
                "note_length": safe_int(row.get("note_length", len(text))),
                "cleaned_text": text,
                # Tokenization
                "input_ids": token_data["input_ids"],
                "attention_mask": token_data["attention_mask"],
                "n_chunks": token_data["n_chunks"],
                "raw_token_count": token_data["raw_token_count"],
            }
            dataset_records.append(record)

        # DEDUPLICATION CHECK — fail loud if duplicates found
        verify_no_duplicates(dataset_records, filename)

        # SAVE
        pkl_filename = filename.replace(".csv", ".pkl")
        pkl_path = PKL_DIR / pkl_filename
        with open(pkl_path, "wb") as f:
            pickle.dump(dataset_records, f)

        # STATS
        stats = compute_stats(dataset_records, pkl_filename)
        all_stats.append(stats)

        print(f"\n✓ Saved: {pkl_filename}")
        print(f"  Records:    {stats['total_records']:,}")
        print(
            f"  Anxiety:    {stats['anxiety_records']:,}  ({stats['prevalence_pct']:.1f}%)"
        )
        print(f"  Control:    {stats['control_records']:,}")
        print(
            f"  Patients:   {stats['total_patients']:,}  "
            f"(anxiety={stats['anxiety_patients']:,}  control={stats['control_patients']:,})"
        )
        print(f"  Avg Weight: {stats['avg_weight']:.3f}")
        print(f"  Avg Chunks: {stats['avg_chunks']:.2f}")

    # =========================================================================
    # SAVE SUMMARY STATS CSV (use in paper methods section)
    # =========================================================================
    summary_path = PKL_DIR / "pkl_dataset_summary.csv"
    if all_stats:
        keys = all_stats[0].keys()
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n✅ Summary stats saved: {summary_path}")
        print("   Use these numbers in your paper's Data section (Table 1).")

    print("\n" + "=" * 90)
    print("✅ ALL PKL DATASETS GENERATED — NO RESAMPLING APPLIED")
    print(f"Output Directory: {PKL_DIR}")
    print()
    print("NEXT STEPS:")
    print("  1. Run scripts/fix_val_pkl.py to generate balanced_supp variants")
    print("     (down-samples controls for supplementary eval — anxiety unchanged)")
    print("  2. Upload ALL *.pkl files in mimic_pkl/ to Kaggle dataset")
    print("  3. Run tc_wpn_kaggle_notebook_v9_publication.ipynb")
    print("=" * 90)


if __name__ == "__main__":
    convert_csv_to_pkl()
