"""
extraction_v2.py
TC-WPN Research Pipeline — Publication-Grade Data Extraction Helpers
Author: Dulhara Kaushalya

FIXES IN THIS VERSION:
- verify_and_clean_notes: removed broken template_heavy filter that was
  killing >95% of valid notes. Template noise is already handled by
  clean_note_text() which strips the header patterns before NLP runs.
- penalize_control_noise: only penalises STRONG diagnostic language,
  not incidental anxiety mentions (fixes 100% contamination bug).
- assign_anxiety_confidence: adds family_history + situational filters
  to reduce false positives.
- compute_section_quality: corrected to run on lowercase text.
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
# ICD CODE DEFINITIONS (Publication-Grade)
# =============================================================================
ANXIETY_ICD10_PREFIXES = [
    "F40",  # Phobic anxiety disorders
    "F41",  # Other anxiety disorders (GAD, Panic, Mixed)
]

ANXIETY_ICD9_PREFIXES = [
    "3000",  # Anxiety states
    "3002",  # Phobic disorders
]

# Exclude these from controls — ensures psych-clean negative class
EXCLUSION_MENTAL_PREFIXES = [
    "F20",  # Schizophrenia
    "F25",  # Schizoaffective
    "F31",  # Bipolar
    "F32",  # Depressive episode
    "F33",  # Recurrent depression
    "F43",  # PTSD / stress reactions
    "296",  # ICD-9 mood disorders
    "295",  # ICD-9 schizophrenia
    "311",  # ICD-9 depression NOS
]


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
            x.startswith(prefix)
            for prefix in ANXIETY_ICD10_PREFIXES + ANXIETY_ICD9_PREFIXES
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

    # Remove any patient who ever had an anxiety diagnosis
    anxiety_subjects = set(anxiety_cases["subject_id"])
    controls = diagnoses[~diagnoses["subject_id"].isin(anxiety_subjects)].copy()

    # Remove patients with other major psychiatric diagnoses
    psych_exclusion = controls["code_clean"].apply(
        lambda x: any(x.startswith(prefix) for prefix in EXCLUSION_MENTAL_PREFIXES)
    )
    controls = controls[~psych_exclusion].copy()

    controls["has_anxiety"] = 0

    return controls[["subject_id", "hadm_id", "has_anxiety"]].drop_duplicates()


# =============================================================================
# NOTE CLEANING
# =============================================================================
def clean_note_text(text):
    """
    Cleans MIMIC clinical note text.
    Removes PHI placeholders and common header boilerplate.
    Preserves clinical content including section headers.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove PHI placeholders like [**Name**], [**Date**]
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)

    # Remove specific header fields only (with colon — exact header match)
    header_patterns = [
        r"admission date\s*:",
        r"discharge date\s*:",
        r"date of birth\s*:",
        r"service\s*:",
        r"sex\s*:",
        r"attending\s*:",
        r"allergies\s*:",
    ]
    for pat in header_patterns:
        text = re.sub(pat, " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Keep alphanumeric + medical punctuation
    text = re.sub(r"[^a-z0-9.,!?;:\-\(\) ]", "", text)

    return text.strip()


# =============================================================================
# SECTION QUALITY SCORING
# FIX: run on already-lowercased cleaned text
# =============================================================================
def compute_section_quality(text):
    """
    Scores a clinical note by presence of high-value psychiatric sections.
    Base score 0.5; each high-value section adds 0.1, capped at 1.0.
    """
    if not isinstance(text, str) or len(text) < 50:
        return 0.5

    # text is already lowercased by clean_note_text
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

    for sec in high_value_sections:
        if sec in text:
            score += 0.1

    return min(score, 1.0)


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================
def compute_temporal_features(df):
    """
    Computes patient-relative temporal metadata.
    note_age_days = days BEFORE the patient's most recent note
    (0 = most recent, higher = older) — correct for recency weighting.
    """
    df = df.copy()

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.sort_values(["subject_id", "charttime"])

    df["visit_number"] = df.groupby("subject_id").cumcount() + 1
    df["total_visits"] = df.groupby("subject_id")["note_id"].transform("count")

    first_visit = df.groupby("subject_id")["charttime"].transform("min")
    last_visit = df.groupby("subject_id")["charttime"].shift(1)

    df["days_since_first_visit"] = (df["charttime"] - first_visit).dt.days
    df["days_since_last_visit"] = (df["charttime"] - last_visit).dt.days

    # Patient-relative: days before most recent note (not global minimum)
    patient_max = df.groupby("subject_id")["charttime"].transform("max")
    df["note_age_days"] = (patient_max - df["charttime"]).dt.days
    df["is_most_recent"] = df["charttime"] == patient_max

    return df[
        [
            "note_id",
            "subject_id",
            "charttime",
            "visit_number",
            "total_visits",
            "days_since_first_visit",
            "days_since_last_visit",
            "note_age_days",
            "is_most_recent",
        ]
    ]


# =============================================================================
# VERIFY AND FILTER NOTES
# FIX: Removed template_heavy filter — it was incorrectly dropping valid notes.
# Template noise is already handled upstream by clean_note_text().
# =============================================================================
def verify_and_clean_notes(df):
    """
    Removes genuinely invalid notes:
    - Empty or too-short notes (< 100 chars after cleaning)
    - Exact duplicates (same patient, same time, same text)
    Does NOT remove notes based on content keywords — that caused 95%+ data loss.
    """
    df = df.copy()

    df["clinical_note_text"] = df["clinical_note_text"].fillna("").astype(str)

    # Remove very short notes — not clinically informative
    df = df[df["clinical_note_text"].str.len() > 100]

    # Remove exact duplicates
    duplicate_cols = ["subject_id", "charttime", "clinical_note_text"]
    df = df.drop_duplicates(subset=duplicate_cols)

    return df


# =============================================================================
# LABEL CONFIDENCE ENGINE (ANXIETY)
# FIX: Added family_history and situational disqualifiers
# =============================================================================
def assign_anxiety_confidence(text):
    """
    Assigns a confidence score [0.4–1.0] and context label to an anxiety note.

    Priority order:
    1. Disqualifiers (family history, situational, negation) → low confidence
    2. Strong diagnostic language → 1.0
    3. Stable/past mentions → 0.65–0.7
    4. General anxiety mention → 0.8
    5. No signal → 0.5
    """
    if not isinstance(text, str):
        return 0.5, "unspecified"

    text = text.lower()

    # ------------------------------------------------------------------
    # DISQUALIFIERS — check first
    # ------------------------------------------------------------------

    # Family history — this is not the patient's own condition
    if re.search(
        r"\b(family history of anxiety|fh:\s*anxiety|fh anxiety|"
        r"mother has anxiety|father has anxiety|"
        r"family hx of anxiety|family h/o anxiety)\b",
        text,
    ):
        return 0.4, "family_history"

    # Situational / procedural anxiety — not a disorder
    if re.search(
        r"\b(anxious about|anxiety about|nervous about|"
        r"anxious regarding|anxiety regarding|"
        r"anxious to (?:go home|leave|return|be discharged)|"
        r"pre-?operative anxiety|procedural anxiety)\b",
        text,
    ):
        return 0.45, "situational"

    # Negation
    if re.search(
        r"\b(no anxiety|denies anxiety|without anxiety|"
        r"negative for anxiety|no panic|denies panic|"
        r"no evidence of anxiety)\b",
        text,
    ):
        return 0.4, "negated"

    # ------------------------------------------------------------------
    # POSITIVE SIGNAL
    # ------------------------------------------------------------------

    # Strong: named disorder
    if re.search(
        r"\b(generalized anxiety disorder|panic disorder|"
        r"anxiety disorder|severe anxiety|gad\b)\b",
        text,
    ):
        return 1.0, "active"

    # Stable / managed
    if re.search(
        r"\b(stable anxiety|controlled anxiety|treated anxiety|"
        r"anxiety controlled|anxiety stable|anxiety managed)\b",
        text,
    ):
        return 0.7, "stable"

    # Past / historical
    if re.search(
        r"\b(history of anxiety|hx anxiety|h/o anxiety|"
        r"past anxiety|prior anxiety|previous anxiety)\b",
        text,
    ):
        return 0.65, "past"

    # General mention — likely current but unspecified
    if re.search(r"\b(anxiety|panic|anxious)\b", text):
        return 0.8, "active"

    return 0.5, "unspecified"


# =============================================================================
# CONTROL PENALTY ENGINE
# FIX: Only penalises STRONG diagnostic language, not incidental mentions.
# Previous version penalised any mention of "anxiety" → 100% contamination.
# =============================================================================
def penalize_control_noise(text):
    """
    Penalises control notes that contain anxiety-related clinical language.
    Returns a weight in [0.2, 1.0]:
    - 1.0  = clean control, no anxiety signal
    - 0.6  = anxiety mentioned in assessment/diagnosis context (use with caution)
    - 0.2  = strong anxiety diagnostic language (likely mislabelled control)

    IMPORTANT: Simple mentions of "anxious" or "anxiety" in body text
    are NOT penalised — these are ubiquitous in discharge notes and do
    not indicate an anxiety disorder in the control patient.
    """
    if not isinstance(text, str):
        return 1.0

    text = text.lower()

    # Strong diagnostic language — this control is likely contaminated
    if re.search(
        r"\b(anxiety disorder|generalized anxiety disorder|panic disorder|"
        r"diagnosis of anxiety|diagnosed with anxiety|"
        r"anxiety disorder confirmed|gad diagnosis)\b",
        text,
    ):
        return 0.2

    # Anxiety mentioned specifically in assessment/impression/diagnosis section
    if re.search(
        r"(assessment|impression|diagnosis|axis i|psychiatric evaluation)"
        r"[^.]{0,80}\b(anxiety|panic)\b",
        text,
    ):
        return 0.6

    # Everything else — incidental "anxious", "anxiety about X", etc.
    # Do NOT penalise. These are normal in discharge notes.
    return 1.0
