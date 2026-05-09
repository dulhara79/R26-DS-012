"""
extraction_v3.py
TC-WPN Research Pipeline — Publication-Grade Data Extraction Helpers
Author: Dulhara Kaushalya

KEY CHANGES FROM extraction_v2.py:
1. has_psychiatric_content() gate added.
   Discharge summaries dominated by non-psychiatric admissions (e.g.,
   cirrhosis, ascites) carry an anxiety ICD code but zero anxiety signal
   in the note text. This gate requires ≥2 psychiatric keywords before
   a note is included — eliminates the single biggest source of label noise.

2. filter_primary_anxiety_admissions() added.
   Filters diagnoses so that anxiety is the PRIMARY diagnosis (seq_num=1)
   OR one of the top-3 diagnoses (seq_num<=3). Notes from admissions where
   anxiety is diagnosis #47 carry almost no anxiety content.

3. compute_section_quality() improved.
   Now also rewards: 'axis i', 'gad-7', 'phq', 'anxiety', 'panic', 'worry'
   Penalizes very short notes (< 200 chars) more aggressively.

4. assign_anxiety_confidence() — no logic changes, but now only called
   AFTER the psychiatric content gate, so false-negative family_history
   misclassifications are less likely.

5. penalize_control_noise() — unchanged, works correctly from v2.

6. verify_and_clean_notes() — unchanged, template filter remains removed.
"""

import pandas as pd
import re
import numpy as np


# =============================================================================
# SAFE CSV LOADER
# =============================================================================
def load_csv_safe(path, usecols=None):
    try:
        df = pd.read_csv(path, usecols=usecols, compression="gzip", low_memory=False)
        print(f"  ✓ Loaded: {path.name} ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"  ❌ Failed to load {path}: {e}")
        return None


# =============================================================================
# ICD CODE DEFINITIONS
# =============================================================================
ANXIETY_ICD10_PREFIXES = [
    "F40",  # Phobic anxiety disorders
    "F41",  # Other anxiety disorders (GAD, Panic, Mixed)
]

ANXIETY_ICD9_PREFIXES = [
    "3000",  # Anxiety states
    "3002",  # Phobic disorders
]

EXCLUSION_MENTAL_PREFIXES = [
    "F20",  "F25",  "F31",  "F32",  "F33",  "F43",
    "296",  "295",  "311",
]


# =============================================================================
# PSYCHIATRIC CONTENT GATE  (v3 — NEW)
#
# WHY: Discharge summaries for admissions with a secondary anxiety ICD code
# (e.g., admitted for cirrhosis, has F41.1 as diagnosis #12) contain the
# word "anxiety" at most once in passing. These notes are labeled anxiety=1
# by the ICD filter but have essentially zero anxiety signal. Including them
# trains the model on label noise and tanks AUROC.
#
# GATE: require ≥2 distinct psychiatric keywords to be present in the note.
# This is a CONTENT filter, not a section filter — it works even on notes
# that don't use standard section headers.
# =============================================================================
PSYCH_KEYWORDS = [
    "anxiety", "anxious", "panic",
    "generalized anxiety", "gad",
    "worry", "worries",
    "psychiatric", "psychiatry",
    "mental status", "mental health",
    "phobia", "phobic",
    "mood", "affect",
    "depression", "depressed",
    "ssri", "snri", "benzodiazepine", "buspirone",
    "psychotherapy", "cbt", "cognitive behavioral",
    "axis i", "dsm",
    "phq", "gad-7",
]


def has_psychiatric_content(text, min_keywords=2):
    """
    Returns True if the note contains at least min_keywords distinct
    psychiatric keywords. Requires the text to already be lowercased.

    Use min_keywords=2 for anxiety cases (strict — ensures the note
    actually discusses anxiety, not just mentions it once).
    Use min_keywords=1 for control notes (they should not mention
    psychiatric content at all if clean, but 1 is fine for filtering).
    """
    if not isinstance(text, str) or len(text) < 100:
        return False
    count = sum(1 for kw in PSYCH_KEYWORDS if kw in text)
    return count >= min_keywords


# =============================================================================
# PRIMARY DIAGNOSIS FILTER  (v3 — NEW)
#
# WHY: An anxiety ICD code at seq_num=47 means the clinician added it as
# an afterthought on a 50-diagnosis list for a complex ICU patient.
# The note will be about the ICU admission, not anxiety.
# Filtering to seq_num <= 3 ensures anxiety was a primary reason for the visit.
# =============================================================================
def filter_primary_anxiety_admissions(diagnoses, max_seq_num=3):
    """
    Filters the diagnoses table to only hadm_ids where anxiety appears
    as one of the top max_seq_num diagnoses.

    This is applied BEFORE identify_anxiety_patients so the resulting
    anxiety cases come from admissions where anxiety was the main concern.
    """
    diagnoses = diagnoses.copy()
    diagnoses["code_clean"] = (
        diagnoses["icd_code"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.upper()
    )

    is_anxiety = diagnoses["code_clean"].apply(
        lambda x: any(
            x.startswith(p) for p in ANXIETY_ICD10_PREFIXES + ANXIETY_ICD9_PREFIXES
        )
    )
    is_primary = diagnoses["seq_num"] <= max_seq_num

    primary_anxiety_hadm = set(
        diagnoses[is_anxiety & is_primary]["hadm_id"].unique()
    )
    print(f"  Primary anxiety admissions (seq_num ≤ {max_seq_num}): "
          f"{len(primary_anxiety_hadm):,}")
    return primary_anxiety_hadm


# =============================================================================
# IDENTIFY ANXIETY PATIENTS
# =============================================================================
def identify_anxiety_patients(diagnoses):
    diagnoses = diagnoses.copy()
    diagnoses["code_clean"] = (
        diagnoses["icd_code"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.upper()
    )
    anxiety_mask = diagnoses["code_clean"].apply(
        lambda x: any(
            x.startswith(p)
            for p in ANXIETY_ICD10_PREFIXES + ANXIETY_ICD9_PREFIXES
        )
    )
    anxiety = diagnoses[anxiety_mask].copy()
    anxiety["has_anxiety"] = 1
    return anxiety[["subject_id", "hadm_id", "has_anxiety"]].drop_duplicates()


# =============================================================================
# IDENTIFY PSYCH-CLEAN CONTROLS
# =============================================================================
def identify_control_patients(diagnoses, anxiety_cases):
    diagnoses = diagnoses.copy()
    diagnoses["code_clean"] = (
        diagnoses["icd_code"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.upper()
    )
    anxiety_subjects = set(anxiety_cases["subject_id"])
    controls = diagnoses[~diagnoses["subject_id"].isin(anxiety_subjects)].copy()
    psych_exclusion = controls["code_clean"].apply(
        lambda x: any(x.startswith(p) for p in EXCLUSION_MENTAL_PREFIXES)
    )
    controls = controls[~psych_exclusion].copy()
    controls["has_anxiety"] = 0
    return controls[["subject_id", "hadm_id", "has_anxiety"]].drop_duplicates()


# =============================================================================
# NOTE CLEANING
# =============================================================================
def clean_note_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    header_patterns = [
        r"admission date\s*:", r"discharge date\s*:", r"date of birth\s*:",
        r"service\s*:", r"sex\s*:", r"attending\s*:", r"allergies\s*:",
    ]
    for pat in header_patterns:
        text = re.sub(pat, " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,!?;:\-\(\) ]", "", text)
    return text.strip()


# =============================================================================
# SECTION QUALITY SCORING  (v3 — expanded keyword list)
# =============================================================================
def compute_section_quality(text):
    """
    Scores a clinical note by presence of high-value psychiatric sections
    and content keywords. Base score 0.5; each match adds 0.1, capped at 1.0.
    v3: added anxiety-specific content keywords in addition to section headers.
    """
    if not isinstance(text, str) or len(text) < 200:
        return 0.3  # v3: penalize very short notes more aggressively

    score = 0.5

    high_value_sections = [
        "history of present illness",
        "past psychiatric history",
        "assessment",
        "mental status examination",
        "psychiatric",
        "axis i",
        "mood and affect",
    ]

    # v3: additional content quality signals
    anxiety_content_signals = [
        "anxiety",
        "panic",
        "gad-7",
        "phq",
        "worry",
        "generalized anxiety disorder",
    ]

    for sec in high_value_sections:
        if sec in text:
            score += 0.1

    # v3: bonus for direct anxiety content (max +0.2)
    anxiety_hits = sum(1 for kw in anxiety_content_signals if kw in text)
    score += min(0.2, anxiety_hits * 0.05)

    return min(score, 1.0)


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================
def compute_temporal_features(df):
    """
    note_age_days = days BEFORE the patient's most recent note
    (0 = most recent, higher = older)
    """
    df = df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.sort_values(["subject_id", "charttime"])

    df["visit_number"] = df.groupby("subject_id").cumcount() + 1
    df["total_visits"] = df.groupby("subject_id")["note_id"].transform("count")

    first_visit = df.groupby("subject_id")["charttime"].transform("min")
    last_visit  = df.groupby("subject_id")["charttime"].shift(1)

    df["days_since_first_visit"] = (df["charttime"] - first_visit).dt.days
    df["days_since_last_visit"]  = (df["charttime"] - last_visit).dt.days

    patient_max = df.groupby("subject_id")["charttime"].transform("max")
    df["note_age_days"]   = (patient_max - df["charttime"]).dt.days
    df["is_most_recent"]  = df["charttime"] == patient_max

    return df[[
        "note_id", "subject_id", "charttime", "visit_number", "total_visits",
        "days_since_first_visit", "days_since_last_visit",
        "note_age_days", "is_most_recent",
    ]]


# =============================================================================
# VERIFY AND FILTER NOTES
# =============================================================================
def verify_and_clean_notes(df):
    df = df.copy()
    df["clinical_note_text"] = df["clinical_note_text"].fillna("").astype(str)
    df = df[df["clinical_note_text"].str.len() > 100]
    duplicate_cols = ["subject_id", "charttime", "clinical_note_text"]
    df = df.drop_duplicates(subset=duplicate_cols)
    return df


# =============================================================================
# LABEL CONFIDENCE ENGINE
# =============================================================================
def assign_anxiety_confidence(text):
    if not isinstance(text, str):
        return 0.5, "unspecified"
    text = text.lower()

    if re.search(
        r"\b(family history of anxiety|fh:\s*anxiety|fh anxiety|"
        r"mother has anxiety|father has anxiety|"
        r"family hx of anxiety|family h/o anxiety)\b", text,
    ):
        return 0.4, "family_history"

    if re.search(
        r"\b(anxious about|anxiety about|nervous about|"
        r"anxious regarding|anxiety regarding|"
        r"anxious to (?:go home|leave|return|be discharged)|"
        r"pre-?operative anxiety|procedural anxiety)\b", text,
    ):
        return 0.45, "situational"

    if re.search(
        r"\b(no anxiety|denies anxiety|without anxiety|"
        r"negative for anxiety|no panic|denies panic|"
        r"no evidence of anxiety)\b", text,
    ):
        return 0.4, "negated"

    if re.search(
        r"\b(generalized anxiety disorder|panic disorder|"
        r"anxiety disorder|severe anxiety|gad\b)\b", text,
    ):
        return 1.0, "active"

    if re.search(
        r"\b(stable anxiety|controlled anxiety|treated anxiety|"
        r"anxiety controlled|anxiety stable|anxiety managed)\b", text,
    ):
        return 0.7, "stable"

    if re.search(
        r"\b(history of anxiety|hx anxiety|h/o anxiety|"
        r"past anxiety|prior anxiety|previous anxiety)\b", text,
    ):
        return 0.65, "past"

    if re.search(r"\b(anxiety|panic|anxious)\b", text):
        return 0.8, "active"

    return 0.5, "unspecified"


# =============================================================================
# CONTROL PENALTY ENGINE
# =============================================================================
def penalize_control_noise(text):
    if not isinstance(text, str):
        return 1.0
    text = text.lower()

    if re.search(
        r"\b(anxiety disorder|generalized anxiety disorder|panic disorder|"
        r"diagnosis of anxiety|diagnosed with anxiety|"
        r"anxiety disorder confirmed|gad diagnosis)\b", text,
    ):
        return 0.2

    if re.search(
        r"(assessment|impression|diagnosis|axis i|psychiatric evaluation)"
        r"[^.]{0,80}\b(anxiety|panic)\b", text,
    ):
        return 0.6

    return 1.0
