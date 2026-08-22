"""
test_patient_split.py — run with: pytest -q tests/

These tests do not need MIMIC. They build synthetic tables so the invariants
can be checked on any machine, including a reviewer's.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcwpn import cohort as C
from tcwpn import splits as S


# =============================================================================
# COHORT ASSIGNMENT
# =============================================================================
def _diag(rows):
    return pd.DataFrame(
        rows, columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    )


def test_anxiety_case_detected_icd10_and_icd9():
    d = _diag([
        (1, 100, 1, "F41.1", 10),
        (2, 200, 2, "300.02", 9),
        (3, 300, 1, "F32.9", 10),
    ])
    out = C.build_patient_cohort(d).set_index("subject_id")
    assert out.loc[1, "arm"] == "case"
    assert out.loc[2, "arm"] == "case"
    assert out.loc[3, "arm"] == "psych_control"


def test_psych_control_never_has_an_anxiety_code():
    d = _diag([
        (1, 100, 1, "F32.9", 10),
        (1, 101, 4, "F41.9", 10),      # same patient also has anxiety
        (2, 200, 1, "F31.0", 10),
    ])
    out = C.build_patient_cohort(d).set_index("subject_id")
    assert out.loc[1, "arm"] == "case", "any anxiety code makes a patient a case"
    assert out.loc[2, "arm"] == "psych_control"


def test_clean_control_has_no_mental_chapter_code():
    d = _diag([
        (1, 100, 1, "I10", 10),        # hypertension only
        (2, 200, 1, "F10.20", 10),     # alcohol use -> neither arm
        (3, 300, 1, "401.9", 9),       # hypertension, ICD-9
        (4, 400, 1, "305.00", 9),      # alcohol abuse, ICD-9 -> excluded
    ])
    out = C.build_patient_cohort(d).set_index("subject_id")
    assert out.loc[1, "arm"] == "clean_control"
    assert out.loc[2, "arm"] == "excluded"
    assert out.loc[3, "arm"] == "clean_control"
    assert out.loc[4, "arm"] == "excluded"


def test_seq_num_restriction_excludes_rather_than_controls():
    """A patient with only a low-priority anxiety code must NEVER become a
    control when --anxiety-seq-num is in force; they are excluded."""
    d = _diag([(1, 100, 9, "F41.1", 10), (1, 100, 1, "F32.9", 10)])
    out = C.build_patient_cohort(d, require_anxiety_seq_num=3).set_index("subject_id")
    assert out.loc[1, "arm"] == "excluded"


def test_cohort_functions_never_receive_text():
    """Guard against regressions: these signatures must stay text-free."""
    import inspect

    for fn in (C.build_patient_cohort, C.flag_diagnoses, C.anxiety_admissions):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"text", "note", "clinical_note_text", "notes"}), (
            f"{fn.__name__} takes a text argument; the cohort must be defined "
            f"from structured data only"
        )


# =============================================================================
# SPLITS
# =============================================================================
def test_split_is_deterministic_and_stable_under_growth():
    ids_small = list(range(100))
    ids_big = list(range(500))
    a = S.assign_splits(ids_small).set_index("subject_id")["split"]
    b = S.assign_splits(ids_big).set_index("subject_id")["split"]
    for i in ids_small:
        assert a[i] == b[i], "adding patients must not move existing ones"


def test_split_proportions_are_approximately_right():
    df = S.assign_splits(range(20000), train_frac=0.70, val_frac=0.15)
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02
    assert abs(frac["val"] - 0.15) < 0.02
    assert abs(frac["test"] - 0.15) < 0.02


def test_split_does_not_depend_on_label():
    """The same subject_id must land in the same split whatever its label."""
    ids = list(range(2000))
    a = S.assign_splits(ids).set_index("subject_id")["split"]
    b = S.assign_splits(list(reversed(ids))).set_index("subject_id")["split"]
    assert (a.sort_index() == b.sort_index()).all()


def test_assert_disjoint_catches_leakage():
    notes = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "split": ["train", "test", "val"],
        "label": [1, 1, 0],
        "note_id": ["a", "b", "c"],
    })
    with pytest.raises(AssertionError, match="PATIENT LEAKAGE"):
        S.assert_disjoint(notes)


def test_assert_disjoint_passes_on_clean_split():
    notes = pd.DataFrame({
        "subject_id": [1, 1, 2, 3],
        "split": ["train", "train", "val", "test"],
        "label": [1, 1, 0, 0],
        "note_id": ["a", "b", "c", "d"],
    })
    assert S.assert_disjoint(notes) == {
        "train_val": 0, "train_test": 0, "val_test": 0}


def test_train_control_downsampling_touches_only_train():
    pat = pd.DataFrame({
        "subject_id": range(300),
        "label": [1] * 50 + [0] * 250,
        "split": ["train"] * 200 + ["val"] * 50 + ["test"] * 50,
    })
    out = S.downsample_control_patients(pat, ratio=1.0, only_split="train")
    for s in ("val", "test"):
        assert (out["split"] == s).sum() == (pat["split"] == s).sum()
    tr = out[out["split"] == "train"]
    assert (tr["label"] == 0).sum() <= (tr["label"] == 1).sum()
