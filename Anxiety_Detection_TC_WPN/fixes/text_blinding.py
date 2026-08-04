"""
text_blinding.py
TC-WPN — Lexical Leakage Control via Term Blinding
Author: (for Dulhara Kaushalya's TC-WPN pipeline)

WHY THIS EXISTS
===============
Your own TF-IDF baseline reaches AUROC 0.91-0.96, which means the anxiety
vs control task is largely separable by the literal presence of anxiety
vocabulary in the note. The model may be reading the diagnosis off the page
rather than inferring it. This module removes that shortcut so you can
measure whether any REAL signal remains.

WHAT IT DOES
============
Deletes (not placeholder-substitutes) anxiety/psychiatric diagnosis terms
and anxiety-medication names from the note text, then RE-TOKENIZES with the
same Bio_ClinicalBERT config used in convert_csv_to_pkl.py, producing a
parallel *_blinded.pkl for every input PKL.

IMPORTANT DESIGN CHOICE — WHY DELETION, NOT A PLACEHOLDER
========================================================
If you replace "anxiety" with a distinctive token like [REDACTED], then only
POSITIVE notes get that token (controls rarely mention anxiety), so the token
PRESENCE becomes a NEW leak and the model just learns "[REDACTED] -> positive".
Deleting the term avoids creating a new positional cue. This is the standard
redaction approach. It is not perfect (surrounding context can still hint),
which is exactly why you ALSO re-run TF-IDF on the blinded text: if TF-IDF
collapses toward chance after blinding, the lexical shortcut is gone.

BLINDING LEVELS
===============
  'anxiety'   : mask only anxiety/panic/phobia terms          (the fair test:
                "can you detect anxiety WITHOUT the word anxiety?")
  'meds'      : mask only anxiety-medication names            (kills the
                prescription_confirmed leak specifically)
  'anx_meds'  : anxiety terms + medication names              (RECOMMENDED main run)
  'psych'     : anxiety + broad psychiatric vocabulary + meds (strictest;
                tests detection without ANY explicit psych cue)

USAGE
=====
  python text_blinding.py --pkl_dir /kaggle/input/.../tc-wpn-data \
                          --out_dir /kaggle/working/blinded \
                          --level anx_meds
Then point your training/eval CONFIG at the *_blinded.pkl files and compare
AUROC against the originals. Report BOTH in the paper.
"""

import re
import os
import sys
import pickle
import argparse
from pathlib import Path

# =============================================================================
# TERM LISTS
# =============================================================================
ANXIETY_TERMS = [
    "anxiety", "anxious", "anxiously", "anxieties",
    "panic", "panicky", "panicked",
    "gad", "gad-7", "gad7",
    "generalized anxiety disorder", "generalised anxiety disorder",
    "panic disorder", "panic attack", "panic attacks",
    "agoraphobia", "agoraphobic",
    "phobia", "phobic", "phobias",
    "social anxiety", "separation anxiety",
    "nervousness", "nervous",          # weaker, but common anxiety proxies
    "worry", "worries", "worried",
]

# Broad psychiatric vocabulary (only used at level='psych').
# These leak because controls are psych-CLEAN (no F20-F43 at all),
# so ANY psychiatric word is class-predictive in this cohort design.
PSYCH_TERMS = [
    "psychiatric", "psychiatry", "psychiatrist",
    "depression", "depressed", "depressive",
    "phq", "phq-9", "phq9",
    "ssri", "snri", "benzodiazepine", "benzodiazepines",
    "psychotherapy", "cbt", "cognitive behavioral", "cognitive behavioural",
    "mental health", "mental status", "mood disorder",
    "axis i", "axis 1", "dsm", "dsm-5", "dsm-iv",
]

# Anxiety-medication names — these directly leak the prescription_confirmed
# gold label, which DOMINATES your positive test set (n~225-233).
MED_TERMS = [
    "sertraline", "escitalopram", "fluoxetine", "paroxetine", "citalopram",
    "fluvoxamine", "venlafaxine", "duloxetine", "desvenlafaxine", "buspirone",
    "lorazepam", "clonazepam", "alprazolam", "diazepam", "oxazepam",
    "temazepam", "hydroxyzine", "pregabalin",
    "zoloft", "lexapro", "prozac", "paxil", "celexa", "effexor",
    "cymbalta", "buspar", "ativan", "klonopin", "xanax", "valium",
]

LEVEL_TO_TERMS = {
    "anxiety":  ANXIETY_TERMS,
    "meds":     MED_TERMS,
    "anx_meds": ANXIETY_TERMS + MED_TERMS,
    "psych":    ANXIETY_TERMS + PSYCH_TERMS + MED_TERMS,
}


def build_pattern(terms):
    """Compile one case-insensitive, word-boundary regex for all terms.
    Longer phrases first so 'generalized anxiety disorder' is matched before 'anxiety'."""
    terms_sorted = sorted(set(terms), key=len, reverse=True)
    escaped = [re.escape(t) for t in terms_sorted]
    # \b...\b with optional trailing 's' already covered by explicit plurals.
    # Use lookarounds that treat hyphen as part of the token (gad-7).
    pattern = r"(?<![a-z0-9])(?:" + "|".join(escaped) + r")(?![a-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def blind_text(text, pattern):
    """Delete every matched term; collapse the whitespace it leaves behind."""
    if not isinstance(text, str) or not text:
        return text, 0
    n_hits = len(pattern.findall(text))
    out = pattern.sub(" ", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out, n_hits


# =============================================================================
# RE-TOKENIZATION (mirrors convert_csv_to_pkl.py exactly)
# =============================================================================
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 512
WINDOW_OVERLAP = 128
STRIDE = MAX_LENGTH - WINDOW_OVERLAP - 2


def sliding_window_tokenize(text, tokenizer):
    if not isinstance(text, str) or not text.strip():
        text = "empty note"
    raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(raw_ids) <= STRIDE:
        enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True)
        return {"input_ids": [enc["input_ids"]], "attention_mask": [enc["attention_mask"]],
                "n_chunks": 1, "raw_token_count": len(raw_ids)}
    chunk_ids, chunk_masks = [], []
    for start in range(0, len(raw_ids), STRIDE):
        chunk = raw_ids[start:start + STRIDE]
        if not chunk:
            continue
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
        enc = tokenizer(chunk_text, max_length=MAX_LENGTH, padding="max_length", truncation=True)
        chunk_ids.append(enc["input_ids"])
        chunk_masks.append(enc["attention_mask"])
        if start + STRIDE >= len(raw_ids):
            break
    return {"input_ids": chunk_ids, "attention_mask": chunk_masks,
            "n_chunks": len(chunk_ids), "raw_token_count": len(raw_ids)}


def blind_pkl(in_path, out_path, pattern, tokenizer):
    with open(in_path, "rb") as f:
        records = pickle.load(f)
    total_hits, n_anx_hit, n_ctrl_hit = 0, 0, 0
    for r in records:
        text = r.get("cleaned_text", "")
        blinded, hits = blind_text(text, pattern)
        total_hits += hits
        if hits > 0:
            if int(r.get("label", 0)) == 1:
                n_anx_hit += 1
            else:
                n_ctrl_hit += 1
        r["cleaned_text"] = blinded
        tok = sliding_window_tokenize(blinded, tokenizer)
        r["input_ids"] = tok["input_ids"]
        r["attention_mask"] = tok["attention_mask"]
        r["n_chunks"] = tok["n_chunks"]
        r["raw_token_count"] = tok["raw_token_count"]
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    n_anx = sum(1 for r in records if int(r.get("label", 0)) == 1)
    n_ctrl = len(records) - n_anx
    print(f"  [{os.path.basename(in_path)}] {len(records):,} records | "
          f"masked terms removed: {total_hits:,}")
    print(f"     anxiety notes with >=1 masked term: {n_anx_hit:,}/{n_anx:,} "
          f"({100*n_anx_hit/max(n_anx,1):.1f}%)")
    print(f"     control notes with >=1 masked term: {n_ctrl_hit:,}/{n_ctrl:,} "
          f"({100*n_ctrl_hit/max(n_ctrl,1):.1f}%)")
    print(f"     -> saved {os.path.basename(out_path)}")
    # The asymmetry (anxiety% >> control%) is the lexical shortcut, quantified.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--level", default="anx_meds", choices=list(LEVEL_TO_TERMS.keys()))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer {TOKENIZER_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    pattern = build_pattern(LEVEL_TO_TERMS[args.level])
    print(f"Blinding level: {args.level}  ({len(set(LEVEL_TO_TERMS[args.level]))} terms)")

    in_dir, out_dir = Path(args.pkl_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(in_dir.glob("*.pkl"))
    if not pkls:
        print(f"No PKL files in {in_dir}"); sys.exit(1)
    for p in pkls:
        out_name = p.stem + f"_blinded_{args.level}.pkl"
        blind_pkl(p, out_dir / out_name, pattern, tokenizer)
    print("\nDONE. Re-run training + baselines on the *_blinded_* PKLs and compare.")


if __name__ == "__main__":
    main()
