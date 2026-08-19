"""
tokenize_cohort.py — Stage 3. Cohort CSV -> per-split tokenised pkl.

    python -m scripts.tokenize_cohort \
        --cohort data/clean/cohort_psych_mimic4.csv \
        --out data/clean/pkl --max-chunks 1

Each record is exactly:

    note_id, subject_id, label, split, note_source,
    input_ids [n_chunks][max_len], attention_mask [n_chunks][max_len],
    days_before_patient_last_note, note_index_within_patient, n_notes_patient

There is deliberately NO weight / label_confidence / training_weight field.
RecordStore raises if it finds one.

--blind lets the same script emit the lexically blinded variant from the SAME
cohort rows, so the blinded and unblinded pkls contain identical note_ids and
can share one episode plan. That is what removes the blinded/unblinded patient
mismatch from the earlier robustness analysis.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# BLINDING VOCABULARIES
# Deletion, not placeholder substitution: a placeholder token would appear
# almost exclusively in case notes and would become a NEW shortcut.
# ---------------------------------------------------------------------------
ANXIETY_TERMS = [
    "anxiety", "anxieties", "anxious", "anxiously",
    "panic", "panicky", "panicked", "panic attack", "panic attacks",
    "gad", "gad-7", "gad7", "generalized anxiety disorder",
    "generalised anxiety disorder", "panic disorder",
    "agoraphobia", "agoraphobic", "phobia", "phobias", "phobic",
    "social anxiety", "separation anxiety", "nervousness",
]
MED_TERMS = [
    "lorazepam", "ativan", "alprazolam", "xanax", "clonazepam", "klonopin",
    "diazepam", "valium", "oxazepam", "temazepam", "restoril",
    "buspirone", "buspar", "hydroxyzine", "vistaril", "atarax",
    "sertraline", "zoloft", "escitalopram", "lexapro", "fluoxetine", "prozac",
    "paroxetine", "paxil", "citalopram", "celexa", "fluvoxamine", "luvox",
    "venlafaxine", "effexor", "duloxetine", "cymbalta", "pregabalin", "lyrica",
    "benzodiazepine", "benzodiazepines", "ssri", "ssris", "snri", "snris",
]
PSYCH_TERMS = [
    "psychiatry", "psychiatric", "psychiatrist", "psych",
    "depression", "depressed", "depressive", "phq", "phq-9", "phq9",
    "bipolar", "ptsd", "post-traumatic", "posttraumatic",
    "schizophrenia", "schizoaffective", "psychotherapy", "cbt",
    "cognitive behavioral therapy", "cognitive behavioural therapy",
    "mental health", "axis i",
]

BLIND_LEVELS = {
    "none": [],
    "anxiety": ANXIETY_TERMS,
    "meds": MED_TERMS,
    "anx_meds": ANXIETY_TERMS + MED_TERMS,
    "psych": ANXIETY_TERMS + MED_TERMS + PSYCH_TERMS,
}


def build_blind_pattern(terms):
    if not terms:
        return None
    ordered = sorted(set(terms), key=len, reverse=True)   # longest match first
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ordered) + r")\b",
                      flags=re.IGNORECASE)


def blind(text, pattern):
    if pattern is None:
        return text
    return re.sub(r"\s{2,}", " ", pattern.sub(" ", text)).strip()


def chunk_tokenize(text, tokenizer, max_len, max_chunks, stride):
    """
    Sliding-window tokenisation. Returns (input_ids, attention_mask) as lists
    of lists, at most `max_chunks` windows. max_chunks=1 keeps only the first
    window — cheap, and on discharge summaries the opening sections carry the
    chief complaint and HPI.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_overflowing_tokens=max_chunks > 1,
        stride=stride if max_chunks > 1 else 0,
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if isinstance(ids[0], int):        # single window -> wrap
        ids, mask = [ids], [mask]
    return ids[:max_chunks], mask[:max_chunks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", default="data/clean/pkl")
    ap.add_argument("--encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=1)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--blind", choices=sorted(BLIND_LEVELS), default="none")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.encoder)
    pattern = build_blind_pattern(BLIND_LEVELS[args.blind])

    df = pd.read_csv(args.cohort, low_memory=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.blind == "none" else f"_blind-{args.blind}"
    stem = Path(args.cohort).stem.replace("cohort_", "")

    n_masked_case = n_masked_ctrl = 0
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if sub.empty:
            print(f"  [{split}] empty, skipping")
            continue

        records = []
        for row in tqdm(sub.itertuples(index=False), total=len(sub),
                        desc=f"tokenising {split}{suffix}"):
            text = row.text if isinstance(row.text, str) else ""
            if pattern is not None:
                hits = len(pattern.findall(text))
                if int(row.label) == 1:
                    n_masked_case += hits
                else:
                    n_masked_ctrl += hits
                text = blind(text, pattern)
            ids, mask = chunk_tokenize(text, tok, args.max_len,
                                       args.max_chunks, args.stride)
            records.append({
                "note_id": str(row.note_id),
                "subject_id": str(row.subject_id),
                "label": int(row.label),
                "split": split,
                "note_source": getattr(row, "note_source", "unknown"),
                "input_ids": ids,
                "attention_mask": mask,
                "days_before_patient_last_note":
                    float(getattr(row, "days_before_patient_last_note", 0.0) or 0.0),
                "note_index_within_patient":
                    int(getattr(row, "note_index_within_patient", 0) or 0),
                "n_notes_patient": int(getattr(row, "n_notes_patient", 1) or 1),
            })

        out = out_dir / f"{stem}_{split}{suffix}.pkl"
        with open(out, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {out}  ({len(records):,} records)")

    if pattern is not None:
        print(f"\n  blinding level '{args.blind}': removed "
              f"{n_masked_case:,} term occurrences from case notes and "
              f"{n_masked_ctrl:,} from control notes")
        print("  Report both counts — a blinding that only touches one class")
        print("  changes the input distribution asymmetrically.")


if __name__ == "__main__":
    main()
