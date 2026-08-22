"""
splits.py — Deterministic, patient-disjoint train/val/test assignment.

Two properties matter here and both were weak in the archived pipeline:

1. The split must be a function of subject_id ONLY. Not of the text, not of
   any label-correlated score, not of the number of notes a patient has.
   We therefore hash the subject_id. A hash split is reproducible without
   storing an id list, is stable if the cohort grows, and cannot accidentally
   depend on row order.

2. The split must be verifiable after the fact. `assert_disjoint()` is called
   by build_clean_cohort.py, by validate_splits.py, and by the unit tests, so
   a reviewer can re-run the check independently.

Class balance is checked but NOT enforced by moving patients between splits,
because moving patients to hit a target prevalence makes the split a function
of the label. Prevalence is reported as-is.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def _hash_unit_interval(subject_id, salt: str) -> float:
    """Map (subject_id, salt) deterministically into [0, 1)."""
    h = hashlib.sha256(f"{salt}::{subject_id}".encode("utf-8")).hexdigest()
    # Use 16 hex chars (64 bits) for plenty of resolution.
    return int(h[:16], 16) / float(1 << 64)


def assign_splits(
    subject_ids,
    salt: str = "tcwpn-clean-v1",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Returns DataFrame[subject_id, split] with split in {train, val, test}.

    test_frac is implied as 1 - train_frac - val_frac.
    Deterministic: same subject_id + same salt always lands in the same split,
    regardless of cohort size or ordering.
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        raise ValueError("train_frac/val_frac out of range")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test split")

    rows = []
    for sid in pd.unique(pd.Series(list(subject_ids))):
        u = _hash_unit_interval(sid, salt)
        if u < train_frac:
            split = "train"
        elif u < train_frac + val_frac:
            split = "val"
        else:
            split = "test"
        rows.append((sid, split))
    return pd.DataFrame(rows, columns=["subject_id", "split"])


def assert_disjoint(notes: pd.DataFrame, subject_col="subject_id",
                    split_col="split") -> dict:
    """
    Hard check: no subject_id appears in more than one split.
    Raises AssertionError on any overlap. Returns the overlap counts (all zero)
    so the caller can print them into the audit report.
    """
    by_split = {
        s: set(notes.loc[notes[split_col] == s, subject_col].unique())
        for s in SPLIT_NAMES
    }
    overlaps = {
        "train_val": len(by_split["train"] & by_split["val"]),
        "train_test": len(by_split["train"] & by_split["test"]),
        "val_test": len(by_split["val"] & by_split["test"]),
    }
    bad = {k: v for k, v in overlaps.items() if v > 0}
    assert not bad, f"PATIENT LEAKAGE BETWEEN SPLITS: {bad}"
    return overlaps


def assert_no_duplicate_notes(notes: pd.DataFrame) -> int:
    """
    A note_id must appear exactly once. Duplicated notes across splits are a
    second, subtler leakage channel (identical text in train and test).
    Returns the number of note_ids checked.
    """
    dup = notes["note_id"].duplicated().sum()
    assert dup == 0, f"{dup} duplicate note_id values in the cohort"
    return len(notes)


def assert_no_duplicate_text_hashes(notes: pd.DataFrame,
                                    text_col="text") -> int:
    """
    Different note_ids can still carry byte-identical text (MIMIC contains
    copy-forwarded notes). If the same text appears in train and test the
    benchmark is contaminated even though note_ids differ.

    Returns the number of text hashes that span more than one split. The
    caller decides whether to drop them; build_clean_cohort.py drops them.
    """
    h = notes[text_col].fillna("").map(
        lambda t: hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()
    )
    tmp = notes.assign(_text_hash=h)
    spans = tmp.groupby("_text_hash")["split"].nunique()
    return int((spans > 1).sum())


def split_summary(notes: pd.DataFrame) -> pd.DataFrame:
    """Per-split patient/note counts and prevalence, for the audit report."""
    rows = []
    for s in SPLIT_NAMES:
        sub = notes[notes["split"] == s]
        pat = sub.drop_duplicates("subject_id")
        n_pat = len(pat)
        n_case_pat = int((pat["label"] == 1).sum())
        rows.append(
            {
                "split": s,
                "patients": n_pat,
                "case_patients": n_case_pat,
                "control_patients": n_pat - n_case_pat,
                "patient_prevalence": round(n_case_pat / n_pat, 4) if n_pat else 0.0,
                "notes": len(sub),
                "case_notes": int((sub["label"] == 1).sum()),
                "control_notes": int((sub["label"] == 0).sum()),
                "note_prevalence": round(float(sub["label"].mean()), 4) if len(sub) else 0.0,
                "notes_per_patient": round(len(sub) / n_pat, 2) if n_pat else 0.0,
            }
        )
    return pd.DataFrame(rows)


def downsample_control_patients(
    patient_table: pd.DataFrame,
    ratio: float,
    seed: int = 42,
    split_col: str = "split",
    only_split: str = "train",
) -> pd.DataFrame:
    """
    Optionally cap control patients at `ratio` x case patients, applied
    ONLY to the training split and ONLY at the patient level, using a seeded
    RNG over subject_ids.

    Never call this on val or test: the evaluation prevalence must reflect
    the cohort, not a convenience ratio. This is the corrected version of the
    archived pipeline's `ctrl_rows.sample(n=len(anx_kept)*4)`, which sampled
    NOTES (after the text had already been filtered) across all splits.
    """
    if ratio is None or ratio <= 0:
        return patient_table

    rng = np.random.default_rng(seed)
    part = patient_table[patient_table[split_col] == only_split]
    other = patient_table[patient_table[split_col] != only_split]

    cases = part[part["label"] == 1]
    controls = part[part["label"] == 0]
    keep_n = min(len(controls), int(round(len(cases) * ratio)))
    if keep_n >= len(controls):
        return patient_table

    keep_idx = rng.choice(controls.index.to_numpy(), size=keep_n, replace=False)
    controls_kept = controls.loc[np.sort(keep_idx)]
    return pd.concat([other, cases, controls_kept], ignore_index=False).sort_index()
