"""
cohort.py — Clean cohort definition for TC-WPN (publication-clean benchmark).

DESIGN CONTRACT (this is the scientific core of the rebuild)
============================================================
Every function in this module operates on STRUCTURED tables only
(diagnoses_icd, patients, admissions). None of them ever receives, reads,
inspects, or filters on clinical note TEXT.

That is the invariant that fixes the previous benchmark. In the archived
pipeline, cohort membership and the test set were partly determined by
regex over the note text (`has_psychiatric_content`, `label_confidence`,
`test_hc = test[test.label_confidence >= 0.7]`). That makes the label a
function of the input, and any classifier that reads the text will appear
to succeed. Here the labels are fixed by ICD codes before any text is
loaded, and the splits are fixed by patient id before any text is loaded.

COHORT DEFINITIONS
==================
CASE       (label 1) : patient has >= 1 anxiety ICD code (F40*/F41* in ICD-10,
                       300.0x/300.2x in ICD-9) in any admission.

PSYCH_CTRL (label 0) : patient has >= 1 non-anxiety psychiatric ICD code
                       (mood, psychotic, trauma/adjustment, OCD) and ZERO
                       anxiety codes.  --> PRIMARY EXPERIMENT

CLEAN_CTRL (label 0) : patient has ZERO codes in the mental-health chapter
                       (F01-F99 / ICD-9 290-319).  --> SECONDARY EXPERIMENT

The primary experiment (case vs psych control) is the clinically meaningful
question and is far harder than case vs healthy control. Report both; lead
with the primary.

Author: Dulhara Kaushalya (IT22130648)
Branch: develop/publication-clean-benchmark
"""

from __future__ import annotations

import pandas as pd

# =============================================================================
# ICD CODE DEFINITIONS
# Codes are compared after stripping '.' and upper-casing, so 'F41.1' -> 'F411'
# and '300.02' -> '30002'.
# =============================================================================

# --- Anxiety (the positive class) --------------------------------------------
ANXIETY_ICD10_PREFIXES = ("F40", "F41")
# ICD-9: 300.0x anxiety states, 300.2x phobic disorders.
# NOTE: 300.3 (OCD) is deliberately EXCLUDED from the case definition and
# instead treated as a psychiatric control, because DSM-5 moved OCD out of
# the anxiety disorders chapter. State this in the paper.
ANXIETY_ICD9_PREFIXES = ("3000", "3002")

# --- Non-anxiety psychiatric comparators (the PRIMARY negative class) --------
PSYCH_CONTROL_ICD10_PREFIXES = (
    "F20",  # schizophrenia
    "F21",  # schizotypal
    "F22",  # delusional disorders
    "F23",  # brief psychotic
    "F25",  # schizoaffective
    "F28", "F29",  # other/unspecified psychosis
    "F30",  # manic episode
    "F31",  # bipolar
    "F32",  # depressive episode
    "F33",  # recurrent depressive
    "F34",  # persistent mood (dysthymia, cyclothymia)
    "F39",  # unspecified mood
    "F42",  # OCD
    "F43",  # reaction to severe stress / adjustment / PTSD
    "F44",  # dissociative
    "F45",  # somatoform
    "F48",  # other neurotic
)
PSYCH_CONTROL_ICD9_PREFIXES = (
    "295",   # schizophrenic disorders
    "296",   # episodic mood disorders
    "297",   # delusional disorders
    "298",   # other non-organic psychoses
    "3003",  # OCD
    "3004",  # dysthymic disorder
    "3006",  # depersonalisation
    "3007",  # hypochondriasis
    "3008",  # somatoform
    "3009",  # unspecified neurotic
    "301",   # personality disorders
    "3078",  # pain disorder / psychalgia
    "308",   # acute stress reaction
    "309",   # adjustment reaction / PTSD (309.81)
    "311",   # depressive disorder NEC
)

# --- Full mental-health chapter, used only to define the CLEAN control arm ---
MENTAL_CHAPTER_ICD10_PREFIX = "F"
MENTAL_CHAPTER_ICD9_RANGE = range(290, 320)  # 290-319 inclusive


def clean_code(series: pd.Series) -> pd.Series:
    """Normalise an ICD code column: strip dots/whitespace, upper-case."""
    return (
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.upper()
    )


def _startswith_any(codes: pd.Series, prefixes) -> pd.Series:
    if not prefixes:
        return pd.Series(False, index=codes.index)
    return codes.str.startswith(tuple(prefixes), na=False)


def _is_mental_chapter(codes: pd.Series, icd_version: pd.Series) -> pd.Series:
    """True where the code belongs to the mental/behavioural chapter."""
    is10 = icd_version.astype(str).str.strip() == "10"
    is9 = ~is10

    m10 = is10 & codes.str.startswith(MENTAL_CHAPTER_ICD10_PREFIX, na=False)

    head3 = pd.to_numeric(codes.str.slice(0, 3), errors="coerce")
    m9 = is9 & head3.between(
        MENTAL_CHAPTER_ICD9_RANGE.start, MENTAL_CHAPTER_ICD9_RANGE.stop - 1
    )
    return m10 | m9.fillna(False)


def flag_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean flag columns to a diagnoses_icd table.

    Expects columns: subject_id, hadm_id, seq_num, icd_code, icd_version.
    Returns a copy with: code_clean, is_anxiety, is_psych_control, is_mental.
    """
    required = {"subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"}
    missing = required - set(diagnoses.columns)
    if missing:
        raise ValueError(f"diagnoses table is missing columns: {sorted(missing)}")

    d = diagnoses.copy()
    d["code_clean"] = clean_code(d["icd_code"])

    is10 = d["icd_version"].astype(str).str.strip() == "10"

    d["is_anxiety"] = (
        is10 & _startswith_any(d["code_clean"], ANXIETY_ICD10_PREFIXES)
    ) | (~is10 & _startswith_any(d["code_clean"], ANXIETY_ICD9_PREFIXES))

    d["is_psych_control"] = (
        is10 & _startswith_any(d["code_clean"], PSYCH_CONTROL_ICD10_PREFIXES)
    ) | (~is10 & _startswith_any(d["code_clean"], PSYCH_CONTROL_ICD9_PREFIXES))

    d["is_mental"] = _is_mental_chapter(d["code_clean"], d["icd_version"])

    return d


def build_patient_cohort(
    diagnoses: pd.DataFrame,
    eligible_subject_ids=None,
    require_anxiety_seq_num: int | None = None,
) -> pd.DataFrame:
    """
    Assign every eligible patient to exactly one arm.

    Parameters
    ----------
    diagnoses : DataFrame
        Raw diagnoses_icd table (all patients).
    eligible_subject_ids : set | None
        Restrict to these subject_ids (e.g. the 18-50 age band). If None,
        all patients present in `diagnoses` are eligible.
    require_anxiety_seq_num : int | None
        If set (e.g. 3), a patient only counts as a CASE when an anxiety code
        appears at seq_num <= this value in at least one admission, i.e. the
        anxiety diagnosis was reasonably prominent for that admission.
        If None, any-position anxiety codes qualify. `None` is the more
        inclusive and more defensible default; report whichever you use.

    Returns
    -------
    DataFrame with one row per patient:
        subject_id, arm, label, n_anxiety_codes, n_psych_codes,
        n_mental_codes, n_admissions
        where arm in {'case', 'psych_control', 'clean_control', 'excluded'}
        and label in {1, 0, 0, -1} respectively.

    Patients that have neither anxiety nor psych codes but DO have some other
    mental-health chapter code (e.g. substance use F10-F19, dementia F00-F09,
    developmental F80-F89) are marked 'excluded': they are neither a clean
    control nor a comparable psychiatric comparator. Excluding them explicitly
    is more honest than silently folding them into either arm.
    """
    d = flag_diagnoses(diagnoses)

    if eligible_subject_ids is not None:
        d = d[d["subject_id"].isin(set(eligible_subject_ids))]

    if require_anxiety_seq_num is not None:
        anx_flag = d["is_anxiety"] & (
            pd.to_numeric(d["seq_num"], errors="coerce") <= require_anxiety_seq_num
        )
    else:
        anx_flag = d["is_anxiety"]
    d = d.assign(_anx_qualifying=anx_flag)

    per_patient = d.groupby("subject_id").agg(
        n_anxiety_codes=("_anx_qualifying", "sum"),
        n_any_anxiety_codes=("is_anxiety", "sum"),
        n_psych_codes=("is_psych_control", "sum"),
        n_mental_codes=("is_mental", "sum"),
        n_admissions=("hadm_id", "nunique"),
    ).reset_index()

    def assign(row):
        if row["n_anxiety_codes"] > 0:
            return "case"
        # A patient with ANY anxiety code but not a "qualifying" one is
        # ambiguous under a seq_num restriction -> exclude, never a control.
        if row["n_any_anxiety_codes"] > 0:
            return "excluded"
        if row["n_psych_codes"] > 0:
            return "psych_control"
        if row["n_mental_codes"] == 0:
            return "clean_control"
        return "excluded"

    per_patient["arm"] = per_patient.apply(assign, axis=1)
    per_patient["label"] = per_patient["arm"].map(
        {"case": 1, "psych_control": 0, "clean_control": 0, "excluded": -1}
    )
    return per_patient


def anxiety_admissions(diagnoses: pd.DataFrame) -> set:
    """
    hadm_ids in which an anxiety code was recorded. Used ONLY as reporting
    metadata (`anx_coded_this_adm`) so the paper can describe how many notes
    come from an admission where anxiety was actually coded. It must never be
    used to filter the validation or test set.
    """
    d = flag_diagnoses(diagnoses)
    return set(d.loc[d["is_anxiety"], "hadm_id"].dropna().unique())


def eligible_by_age(patients: pd.DataFrame, age_min=18, age_max=50) -> set:
    """
    MIMIC-IV: patients.csv.gz has `anchor_age` directly.
    Returns the set of subject_ids inside the young-adult band, matching the
    parent project's population (young adults with anxiety disorders).
    """
    if "anchor_age" not in patients.columns:
        raise ValueError(
            "patients table has no `anchor_age`; for MIMIC-III use "
            "eligible_by_age_mimic3() instead."
        )
    p = patients[
        (patients["anchor_age"] >= age_min) & (patients["anchor_age"] <= age_max)
    ]
    return set(p["subject_id"].unique())


def eligible_by_age_mimic3(patients3: pd.DataFrame, admissions3: pd.DataFrame,
                           age_min=18, age_max=50) -> set:
    """
    MIMIC-III has no anchor_age; age is derived as (admittime - dob).

    IMPORTANT: MIMIC-III shifts the DOB of patients over 89 to exactly 300
    years before their first admission. Those patients therefore compute to
    age ~300 and are naturally excluded by an upper bound of 50, but the
    resulting ages for everyone else are still only approximate. Age at FIRST
    admission is used so a patient gets one stable age.
    """
    p = patients3.rename(columns=str.lower).copy()
    a = admissions3.rename(columns=str.lower).copy()
    p["dob"] = pd.to_datetime(p["dob"], errors="coerce")
    a["admittime"] = pd.to_datetime(a["admittime"], errors="coerce")

    first_adm = a.groupby("subject_id")["admittime"].min().reset_index()
    merged = first_adm.merge(p[["subject_id", "dob"]], on="subject_id", how="left")
    merged["age"] = (merged["admittime"] - merged["dob"]).dt.days / 365.25
    keep = merged[(merged["age"] >= age_min) & (merged["age"] <= age_max)]
    return set(keep["subject_id"].unique())
