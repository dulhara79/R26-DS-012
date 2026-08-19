"""
indexing.py — Clinical index-time protocol.

WHY THIS EXISTS
===============
The benchmark is clean in the machine-learning sense (no text-derived labels,
no patient leakage) but was not yet clean in the *clinical* sense. A patient is
a case if an anxiety code appears in ANY admission. Notes were then taken from
the whole record, including notes written after that diagnosis was made. A model
can therefore score well by recognising post-diagnosis documentation -- follow-up
language, medication reconciliation, "history of" phrasing -- rather than by
detecting anxiety.

That is not label leakage in the usual sense. The label is still a function of
the codes alone. It is *temporal target leakage*: the input window straddles the
event the label describes, so the task being measured is not the task being
claimed.

THE PROTOCOL
============
Every patient gets one index time t_index, fixed from structured data before any
note is read. Notes are then admitted according to a policy:

    none            all notes (the previous behaviour; kept only so the old and
                    new results can be compared in the paper)
    at_or_before    charttime <= dischtime of the index admission
                    -> CONCURRENT DETECTION: "given this admission's record,
                       is this patient's presentation an anxiety presentation?"
                       The index admission's own discharge summary is included,
                       so explicit diagnosis language is available. Legitimate,
                       but it must be named as concurrent, not predictive.
    strictly_before charttime <  admittime of the index admission
                    -> PROSPECTIVE DETECTION: "from the record so far, can we
                       anticipate the anxiety diagnosis?" Harder, and the only
                       version that supports predictive language in the abstract.

INDEX TIME BY ARM
=================
    case            first admission carrying an anxiety code
    psych_control   first admission carrying its qualifying psychiatric code
    clean_control   an admission drawn to match the ordinal position of case
                    index admissions (seeded), because clean controls have no
                    defining diagnosis to anchor on

The clean-control rule is the weakest link and must be stated in the paper. Any
anchor for a patient with no qualifying diagnosis is arbitrary; ordinal matching
at least prevents the systematic difference where cases are indexed mid-record
and controls at end-of-record, which would make record length itself predictive.
This is one more reason the psychiatric-control arm is the primary experiment.

TEMPORAL FEATURE
================
    days_before_index = (t_index - charttime) in days,  >= 0 under either policy

This replaces days_before_patient_last_note as the input to

    w_i^T = exp(-lambda * days_before_index / 365)

The old feature was measured against the patient's LAST note, which under any
index policy sits at or after t_index, so it encoded information about the
future relative to the prediction point. The new one is measured against a
timestamp fixed before the model sees anything.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cohort import flag_diagnoses

POLICIES = ("none", "at_or_before", "strictly_before")


# ===========================================================================
# 1. ADMISSION TIMELINE
# ===========================================================================
def build_admission_timeline(admissions: pd.DataFrame) -> pd.DataFrame:
    """
    admissions : subject_id, hadm_id, admittime, dischtime  (any case)

    Returns one row per admission with an `admission_ordinal` (0-based, by
    admittime within patient).
    """
    a = admissions.copy()
    a.columns = [c.lower() for c in a.columns]
    need = {"subject_id", "hadm_id", "admittime"}
    missing = need - set(a.columns)
    if missing:
        raise ValueError(f"admissions table missing {sorted(missing)}")

    a["admittime"] = pd.to_datetime(a["admittime"], errors="coerce")
    if "dischtime" in a.columns:
        a["dischtime"] = pd.to_datetime(a["dischtime"], errors="coerce")
    else:
        a["dischtime"] = pd.NaT
    # If dischtime is missing, fall back to admittime so the window is
    # conservative (shorter) rather than silently unbounded.
    a["dischtime"] = a["dischtime"].fillna(a["admittime"])

    a = a[a["admittime"].notna()].sort_values(["subject_id", "admittime"])
    a["admission_ordinal"] = a.groupby("subject_id").cumcount()
    a["n_admissions_patient"] = a.groupby("subject_id")["hadm_id"].transform("count")
    return a[["subject_id", "hadm_id", "admittime", "dischtime",
              "admission_ordinal", "n_admissions_patient"]]


# ===========================================================================
# 2. INDEX TIME PER PATIENT
# ===========================================================================
def compute_index_times(
    diagnoses: pd.DataFrame,
    admissions: pd.DataFrame,
    per_patient: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    per_patient : output of cohort.build_patient_cohort (subject_id, arm, label)

    Returns
    -------
    subject_id, index_hadm_id, index_admittime, index_dischtime,
    index_ordinal, index_rule
    """
    tl = build_admission_timeline(admissions)
    d = flag_diagnoses(diagnoses)

    anx_hadm = set(d.loc[d["is_anxiety"], "hadm_id"].dropna().unique())
    psy_hadm = set(d.loc[d["is_psych_control"], "hadm_id"].dropna().unique())

    arms = per_patient.set_index("subject_id")["arm"].to_dict()
    tl = tl[tl["subject_id"].isin(arms)].copy()
    tl["arm"] = tl["subject_id"].map(arms)

    rows = []

    # ---- cases: first anxiety-coded admission ------------------------------
    cas = tl[(tl["arm"] == "case") & tl["hadm_id"].isin(anx_hadm)]
    first_cas = cas.groupby("subject_id", as_index=False).first()
    first_cas["index_rule"] = "first_anxiety_coded_admission"
    rows.append(first_cas)

    # ---- psychiatric controls: first psych-coded admission -----------------
    psy = tl[(tl["arm"] == "psych_control") & tl["hadm_id"].isin(psy_hadm)]
    first_psy = psy.groupby("subject_id", as_index=False).first()
    first_psy["index_rule"] = "first_psych_coded_admission"
    rows.append(first_psy)

    # ---- clean controls: ordinal-matched draw ------------------------------
    clean = tl[tl["arm"] == "clean_control"]
    if len(clean):
        # Empirical distribution of the FRACTIONAL position of case index
        # admissions, so matching works across patients with different
        # admission counts.
        if len(first_cas):
            frac = (
                first_cas["admission_ordinal"]
                / np.maximum(first_cas["n_admissions_patient"] - 1, 1)
            ).clip(0, 1).to_numpy()
        else:
            frac = np.array([0.0])

        rng = np.random.default_rng(seed)
        picks = []
        for sid, grp in clean.groupby("subject_id", sort=True):
            n = len(grp)
            f = rng.choice(frac)
            ordinal = int(round(f * (n - 1)))
            picks.append(grp.iloc[ordinal])
        clean_idx = pd.DataFrame(picks).reset_index(drop=True)
        clean_idx["index_rule"] = "ordinal_matched_admission"
        rows.append(clean_idx)

    out = pd.concat([r for r in rows if len(r)], ignore_index=True)
    out = out.rename(columns={
        "hadm_id": "index_hadm_id",
        "admittime": "index_admittime",
        "dischtime": "index_dischtime",
        "admission_ordinal": "index_ordinal",
    })
    return out[["subject_id", "index_hadm_id", "index_admittime",
                "index_dischtime", "index_ordinal", "n_admissions_patient",
                "index_rule"]]


# ===========================================================================
# 3. APPLY THE POLICY TO NOTES
# ===========================================================================
def apply_index_policy(
    notes: pd.DataFrame,
    index_times: pd.DataFrame,
    policy: str = "at_or_before",
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (filtered_notes_with_days_before_index, report).

    A patient with no admissible note under the policy disappears from the
    cohort entirely. That attrition is reported per class because it is a
    differential-selection risk: if the policy removes 60 % of cases and 20 % of
    controls, the surviving cohort is no longer the cohort you defined.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")

    n = notes.copy()
    n["charttime"] = pd.to_datetime(n["charttime"], errors="coerce")
    before_patients = n.groupby("label")["subject_id"].nunique().to_dict()
    before_notes = n.groupby("label").size().to_dict()

    n = n.merge(index_times, on="subject_id", how="left")
    no_index = n["index_admittime"].isna()
    report = {
        "policy": policy,
        "notes_dropped_no_index_time": int(no_index.sum()),
        "patients_without_index_time": int(
            n.loc[no_index, "subject_id"].nunique()),
    }
    n = n[~no_index]

    if policy == "none":
        ref = pd.to_datetime(n["index_dischtime"])
        n["days_before_index"] = (ref - n["charttime"]).dt.total_seconds() / 86400.0
        kept = n
    elif policy == "at_or_before":
        ref = pd.to_datetime(n["index_dischtime"])
        n["days_before_index"] = (ref - n["charttime"]).dt.total_seconds() / 86400.0
        kept = n[n["charttime"] <= ref]
    else:  # strictly_before
        ref = pd.to_datetime(n["index_admittime"])
        n["days_before_index"] = (ref - n["charttime"]).dt.total_seconds() / 86400.0
        kept = n[n["charttime"] < ref]

    kept = kept.copy()
    kept["days_before_index"] = kept["days_before_index"].clip(lower=0.0)

    after_patients = kept.groupby("label")["subject_id"].nunique().to_dict()
    after_notes = kept.groupby("label").size().to_dict()

    for lab in (0, 1):
        b_p, a_p = before_patients.get(lab, 0), after_patients.get(lab, 0)
        b_n, a_n = before_notes.get(lab, 0), after_notes.get(lab, 0)
        report[f"label{lab}_patients_before"] = int(b_p)
        report[f"label{lab}_patients_after"] = int(a_p)
        report[f"label{lab}_patients_retained_pct"] = round(
            100 * a_p / max(b_p, 1), 2)
        report[f"label{lab}_notes_before"] = int(b_n)
        report[f"label{lab}_notes_after"] = int(a_n)
        report[f"label{lab}_notes_retained_pct"] = round(
            100 * a_n / max(b_n, 1), 2)

    report["differential_patient_retention_pp"] = round(
        report["label1_patients_retained_pct"]
        - report["label0_patients_retained_pct"], 2)

    return kept, report


def recompute_within_patient_features(notes: pd.DataFrame) -> pd.DataFrame:
    """
    After the policy removes notes, per-patient counters must be recomputed or
    they describe a record the model never sees.
    """
    n = notes.copy().sort_values(["subject_id", "charttime"])
    n["note_index_within_patient"] = n.groupby("subject_id").cumcount()
    n["n_notes_patient"] = n.groupby("subject_id")["note_id"].transform("count")
    last = n.groupby("subject_id")["charttime"].transform("max")
    n["days_before_patient_last_note"] = (
        (last - n["charttime"]).dt.total_seconds() / 86400.0)
    n["is_patient_last_note"] = n["charttime"].eq(last)
    return n
