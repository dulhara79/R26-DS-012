"""
test_index_time.py — the index-time protocol on data whose answer is known by hand.

The scenario mirrors the failure the supervisor identified:

    patient 1 (case)
        2010 note   2012 note   2015 ANXIETY ADMISSION   2018 note
                                        ^ index
    Under 'strictly_before' the 2018 note must be gone.
    Under 'at_or_before'   the 2018 note must be gone, the 2015 discharge kept.
    Under 'none'           everything is kept (the old behaviour).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcwpn import indexing as IX  # noqa: E402


def _admissions():
    return pd.DataFrame([
        # subject, hadm, admittime, dischtime
        (1, 11, "2010-01-01", "2010-01-10"),
        (1, 12, "2012-01-01", "2012-01-10"),
        (1, 15, "2015-06-01", "2015-06-20"),   # anxiety coded here
        (1, 18, "2018-01-01", "2018-01-10"),
        (2, 21, "2011-03-01", "2011-03-08"),   # psych control, coded here
        (2, 22, "2016-03-01", "2016-03-08"),
    ], columns=["subject_id", "hadm_id", "admittime", "dischtime"])


def _diagnoses():
    return pd.DataFrame([
        (1, 11, 1, "I10", 10),
        (1, 12, 1, "E119", 10),
        (1, 15, 1, "F411", 10),     # GAD -> case, index admission
        (1, 18, 1, "I10", 10),
        (2, 21, 1, "F329", 10),     # depression -> psych control
        (2, 22, 1, "I10", 10),
    ], columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])


def _per_patient():
    return pd.DataFrame([
        (1, "case", 1),
        (2, "psych_control", 0),
    ], columns=["subject_id", "arm", "label"])


def _notes():
    return pd.DataFrame([
        # note_id, subject, hadm, charttime, label, split
        ("n1", 1, 11, "2010-01-09", 1, "train"),
        ("n2", 1, 12, "2012-01-09", 1, "train"),
        ("n3", 1, 15, "2015-06-19", 1, "train"),   # index admission's own note
        ("n4", 1, 18, "2018-01-09", 1, "train"),   # POST-INDEX -> must be dropped
        ("n5", 2, 21, "2011-03-07", 0, "train"),
        ("n6", 2, 22, "2016-03-07", 0, "train"),   # POST-INDEX -> must be dropped
    ], columns=["note_id", "subject_id", "hadm_id", "charttime", "label", "split"])


def _index_times():
    return IX.compute_index_times(_diagnoses(), _admissions(), _per_patient())


# ===========================================================================
def test_case_index_is_the_first_anxiety_admission():
    idx = _index_times().set_index("subject_id")
    assert idx.loc[1, "index_hadm_id"] == 15
    assert idx.loc[1, "index_rule"] == "first_anxiety_coded_admission"


def test_psych_control_index_is_the_first_psych_admission():
    idx = _index_times().set_index("subject_id")
    assert idx.loc[2, "index_hadm_id"] == 21
    assert idx.loc[2, "index_rule"] == "first_psych_coded_admission"


@pytest.mark.parametrize("policy,expected", [
    ("none", {"n1", "n2", "n3", "n4", "n5", "n6"}),
    ("at_or_before", {"n1", "n2", "n3", "n5"}),
    # n5 is written DURING patient 2's index admission, so a strictly-
    # prospective window excludes it and patient 2 leaves the cohort
    # entirely. See test_strictly_before_can_empty_an_arm.
    ("strictly_before", {"n1", "n2"}),
])
def test_policy_admits_the_right_notes(policy, expected):
    kept, _ = IX.apply_index_policy(_notes(), _index_times(), policy)
    assert set(kept["note_id"]) == expected


def test_post_index_notes_are_gone_under_both_real_policies():
    for policy in ("at_or_before", "strictly_before"):
        kept, _ = IX.apply_index_policy(_notes(), _index_times(), policy)
        assert "n4" not in set(kept["note_id"]), (
            f"{policy}: the 2018 note post-dates the 2015 anxiety diagnosis"
        )


def test_delta_is_measured_against_the_index_not_the_last_note():
    kept, _ = IX.apply_index_policy(_notes(), _index_times(), "strictly_before")
    row = kept.set_index("note_id").loc["n1"]
    # 2010-01-09 -> index admittime 2015-06-01 is ~1969 days
    delta = float(row["days_before_index"])
    assert 1960 < delta < 1980, delta
    # The old feature would have measured against 2012-01-09 (the last admitted
    # note), giving ~730 days. Confirm we are NOT reporting that.
    assert abs(delta - 730) > 100


def test_delta_is_never_negative():
    for policy in IX.POLICIES:
        kept, _ = IX.apply_index_policy(_notes(), _index_times(), policy)
        assert (kept["days_before_index"] >= 0).all()


def test_report_flags_differential_attrition():
    _, rep = IX.apply_index_policy(_notes(), _index_times(), "strictly_before")
    assert rep["label1_patients_after"] == 1
    assert rep["label0_patients_after"] == 0
    assert rep["differential_patient_retention_pp"] == 100.0


def test_strictly_before_can_empty_an_arm():
    """
    THE PRACTICAL WARNING THIS SUITE EXISTS TO SURFACE.

    A patient whose qualifying diagnosis is coded at their FIRST admission has
    no prior record, so a strictly-prospective window admits nothing and the
    patient is dropped. In MIMIC that describes most single-admission patients,
    and single-admission patients are the majority.

    Consequence: 'strictly_before' does not merely shrink the cohort, it
    selects for multi-admission patients, and it may do so at different rates
    in the two arms. Here it removes 100 % of controls and 0 % of cases.

    Check `differential_patient_retention_pp` in the stage report before
    trusting any result computed under this policy. If the number is large,
    the prospective arm is a different population, not a harder version of the
    same one.
    """
    _, rep = IX.apply_index_policy(_notes(), _index_times(), "strictly_before")
    assert rep["label0_patients_before"] == 1
    assert rep["label0_patients_after"] == 0
    assert abs(rep["differential_patient_retention_pp"]) > 10


def test_within_patient_features_are_recomputed_after_filtering():
    kept, _ = IX.apply_index_policy(_notes(), _index_times(), "strictly_before")
    kept = IX.recompute_within_patient_features(kept)
    p1 = kept[kept["subject_id"] == 1]
    assert p1["n_notes_patient"].unique().tolist() == [2]
    assert sorted(p1["note_index_within_patient"]) == [0, 1]


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        IX.apply_index_policy(_notes(), _index_times(), "whenever")
