"""
updated convert_pkl.py file convert_csv_to_pkl_v2.py
Publication-Grade CSV → PKL Conversion for TC-WPN
Author: Dulhara Kaushalya

Purpose:
- Converts processed MIMIC CSV splits into PKL
- Preserves:
    * Bio_ClinicalBERT tokenization
    * Sliding-window chunking
    * Label confidence
    * Section quality
    * Training weights
    * Source type
    * Anxiety context
    * Temporal metadata
- Supports domain adaptation + confidence-aware episodic learning
"""

import pandas as pd
import pickle
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

# Effective stride after CLS + SEP
STRIDE = MAX_LENGTH - WINDOW_OVERLAP - 2


# =============================================================================
# TOKENIZATION ENGINE
# =============================================================================
def sliding_window_tokenize(text: str, tokenizer):
    """
    Handles ultra-long clinical notes with token-space chunking.
    Preserves:
    - input_ids
    - attention_masks
    - chunk count
    - raw token count
    """

    if not isinstance(text, str) or not text.strip():
        text = "empty note"

    # Raw tokenization without truncation
    raw_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    # -------------------------------------------------------------------------
    # SHORT NOTE
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # LONG NOTE
    # -------------------------------------------------------------------------
    chunk_ids = []
    chunk_masks = []

    for start in range(0, len(raw_ids), STRIDE):
        chunk = raw_ids[start : start + STRIDE]

        if not chunk:
            continue

        # Rebuild text from tokens
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
# RECORD VALIDATION
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
# CSV → PKL PIPELINE
# =============================================================================
def convert_csv_to_pkl():
    print("=" * 90)
    print("PUBLICATION-GRADE MIMIC CSV → PKL CONVERSION")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # TOKENIZER LOAD
    # -------------------------------------------------------------------------
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # -------------------------------------------------------------------------
    # LOOP FILES
    # -------------------------------------------------------------------------
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

        dataset_records = []

        # =====================================================================
        # ITERATE RECORDS
        # =====================================================================
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Tokenizing {filename}",
        ):

            text = safe_str(row.get("clinical_note_text", ""))

            # -----------------------------------------------------------------
            # TOKENIZE
            # -----------------------------------------------------------------
            token_data = sliding_window_tokenize(text, tokenizer)

            # -----------------------------------------------------------------
            # RECORD
            # -----------------------------------------------------------------
            record = {
                # =============================================================
                # CORE IDS
                # =============================================================
                "note_id": safe_str(row.get("note_id", "unknown")),
                "subject_id": safe_str(row.get("subject_id", "unknown")),
                "hadm_id": safe_str(row.get("hadm_id", "unknown")),
                # =============================================================
                # LABELS
                # =============================================================
                "label": safe_int(row.get("has_anxiety", 0)),
                "label_confidence": safe_float(row.get("label_confidence", 1.0)),
                "section_quality": safe_float(row.get("section_quality", 1.0)),
                "weight": safe_float(row.get("training_weight", 1.0)),
                # =============================================================
                # DOMAIN / CONTEXT
                # =============================================================
                "source_type": safe_str(row.get("source_type", "unknown")),
                "anxiety_context": safe_str(row.get("anxiety_context", "unspecified")),
                "dataset_split": safe_str(row.get("dataset_split", "unknown")),
                # =============================================================
                # DEMOGRAPHICS
                # =============================================================
                "gender": safe_str(row.get("gender", "unknown")),
                "age_at_admission": safe_float(row.get("age_at_admission", 0.0)),
                # =============================================================
                # TEMPORAL
                # =============================================================
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
                # =============================================================
                # NOTE FEATURES
                # =============================================================
                "note_length": safe_int(row.get("note_length", len(text))),
                "cleaned_text": text,
                # =============================================================
                # TOKENIZATION
                # =============================================================
                "input_ids": token_data["input_ids"],
                "attention_mask": token_data["attention_mask"],
                "n_chunks": token_data["n_chunks"],
                "raw_token_count": token_data["raw_token_count"],
            }

            dataset_records.append(record)

        # ---------------------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------------------
        pkl_filename = filename.replace(".csv", ".pkl")
        pkl_path = PKL_DIR / pkl_filename

        with open(pkl_path, "wb") as f:
            pickle.dump(dataset_records, f)

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------
        anxiety_count = sum(r["label"] for r in dataset_records)
        control_count = len(dataset_records) - anxiety_count

        avg_weight = sum(r["weight"] for r in dataset_records) / max(
            len(dataset_records), 1
        )

        avg_chunks = sum(r["n_chunks"] for r in dataset_records) / max(
            len(dataset_records), 1
        )

        print(f"\n✓ Saved: {pkl_filename}")
        print(f"  Records: {len(dataset_records):,}")
        print(f"  Anxiety: {anxiety_count:,}")
        print(f"  Control: {control_count:,}")
        print(f"  Avg Weight: {avg_weight:.3f}")
        print(f"  Avg Chunks: {avg_chunks:.2f}")

    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 90)
    print("✅ ALL PKL DATASETS GENERATED SUCCESSFULLY")
    print(f"Output Directory: {PKL_DIR}")
    print("=" * 90)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    convert_csv_to_pkl()
