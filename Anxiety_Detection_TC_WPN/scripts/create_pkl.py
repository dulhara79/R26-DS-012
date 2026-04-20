"""
Convert MIMIC-IV CSVs to PKL for PyTorch Training
Formats data exactly to Evaluator Specs, including Bio_ClinicalBERT tokenization.
"""

import pandas as pd
import pickle
from pathlib import Path
import os
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "mimic_processed"
PKL_DIR = PROJECT_ROOT / "mimic_pkl"
PKL_DIR.mkdir(exist_ok=True)

TARGET_FILES = [
    "mimic_anxiety_train_balanced.csv",
    "mimic_anxiety_train_high_conf.csv",
    "mimic_anxiety_val_real_world.csv",
    "mimic_anxiety_test_real_world.csv",
    "mimic_anxiety_test_high_conf.csv",
]

TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 512
WINDOW_OVERLAP = 128


# =============================================================================
# TOKENIZATION LOGIC (Optimized Token-Space windowing)
# =============================================================================
def sliding_window_tokenize(text: str, tokenizer) -> dict:
    """Tokenizes text into chunks to handle sequences longer than 512 tokens."""
    if not isinstance(text, str) or not text.strip():
        text = "empty note"

    stride = MAX_LENGTH - WINDOW_OVERLAP - 2
    raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    if len(raw_ids) <= stride:
        enc = tokenizer(
            text, max_length=MAX_LENGTH, padding="max_length", truncation=True
        )
        return {
            "input_ids": [enc["input_ids"]],
            "attention_mask": [enc["attention_mask"]],
            "n_chunks": 1,
            "raw_token_count": len(raw_ids),
        }

    chunk_ids, chunk_masks = [], []
    for start in range(0, len(raw_ids), stride):
        chunk = raw_ids[start : start + stride]

        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

        enc = tokenizer(
            chunk_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )

        chunk_ids.append(enc["input_ids"])
        chunk_masks.append(enc["attention_mask"])

        if start + stride >= len(raw_ids):
            break

    return {
        "input_ids": chunk_ids,
        "attention_mask": chunk_masks,
        "n_chunks": len(chunk_ids),
        "raw_token_count": len(raw_ids),
    }


# =============================================================================
# MAIN CONVERSION PIPELINE
# =============================================================================
def convert_csv_to_pkl():
    print("=" * 80)
    print("GENERATING PyTorch .pkl DATASETS (WITH TOKENIZATION)")
    print("=" * 80)

    print(f"\nLoading Tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for filename in TARGET_FILES:
        csv_path = PROCESSED_DIR / filename

        if not csv_path.exists():
            print(f"\n⚠️ WARNING: Could not find {filename}. Skipping.")
            continue

        print(f"\nProcessing {filename}...")
        df = pd.read_csv(csv_path)
        dataset_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            text = str(row.get("clinical_note_text", ""))
            tokens = sliding_window_tokenize(text, tokenizer)

            record = {
                "note_id": str(row.get("note_id", "unknown")),
                "subject_id": str(row.get("subject_id", "unknown")),
                "label": int(row.get("has_anxiety", 0)),
                # EVALUATOR FIX: Map to training_weight, NOT section_quality!
                "weight": float(row.get("training_weight", 1.0)),
                "note_timestamp": str(row.get("charttime", "")),
                "visit_number": int(row.get("visit_number", 1)),
                "days_since_first_visit": float(row.get("days_since_first_visit", 0.0)),
                "days_since_last_visit": float(row.get("days_since_last_visit", 0.0)),
                "total_visits": int(row.get("total_visits", 1)),
                "note_age_days": float(row.get("note_age_days", 0.0)),
                "section_quality": float(row.get("section_quality", 1.0)),
                "cleaned_text": text,
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "n_chunks": tokens["n_chunks"],
                "raw_token_count": tokens["raw_token_count"],
            }
            dataset_list.append(record)

        pkl_filename = filename.replace(".csv", ".pkl")
        pkl_path = PKL_DIR / pkl_filename
        with open(pkl_path, "wb") as f:
            pickle.dump(dataset_list, f)
        print(f"  ✓ Saved: {pkl_filename} ({len(dataset_list):,} records)")

    print("\n" + "=" * 80)
    print(f"✅ ALL PKL FILES GENERATED IN: {PKL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    convert_csv_to_pkl()
