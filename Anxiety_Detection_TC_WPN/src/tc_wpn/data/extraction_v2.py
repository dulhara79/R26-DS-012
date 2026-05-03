# updated extraction_v2
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
    "F41",  # Other anxiety disorders
]

ANXIETY_ICD9_PREFIXES = [
    "3000",  # Anxiety states
    "3002",  # Phobic disorders
]

EXCLUSION_MENTAL_PREFIXES = [
    "F20",
    "F25",
    "F31",
    "F32",
    "F33",
    "F43",
    "296",
    "295",
    "311",
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
# IDENTIFY CLEAN CONTROLS
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
        lambda x: any(x.startswith(prefix) for prefix in EXCLUSION_MENTAL_PREFIXES)
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

    # Remove PHI placeholders
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)

    # Remove common discharge template noise
    template_patterns = [
        r"admission date:",
        r"discharge date:",
        r"date of birth:",
        r"service:",
        r"sex:",
    ]

    for pat in template_patterns:
        text = re.sub(pat, " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Keep useful punctuation
    text = re.sub(r"[^a-z0-9.,!?;:\- ]", "", text)

    return text.strip()


# =============================================================================
# SECTION QUALITY SCORING
# =============================================================================
def compute_section_quality(text):
    if not isinstance(text, str):
        return 0.5

    score = 0.5

    high_value_sections = [
        "history of present illness",
        "past psychiatric history",
        "assessment",
        "mental status examination",
        "psychiatric",
    ]

    for sec in high_value_sections:
        if sec in text:
            score += 0.1

    return min(score, 1.0)


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================
def compute_temporal_features(df):
    df = df.copy()

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.sort_values(["subject_id", "charttime"])

    df["visit_number"] = df.groupby("subject_id").cumcount() + 1
    df["total_visits"] = df.groupby("subject_id")["note_id"].transform("count")

    first_visit = df.groupby("subject_id")["charttime"].transform("min")
    last_visit = df.groupby("subject_id")["charttime"].shift(1)

    df["days_since_first_visit"] = (df["charttime"] - first_visit).dt.days
    df["days_since_last_visit"] = (df["charttime"] - last_visit).dt.days

    # FIXED: patient-relative note age
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
# =============================================================================
def verify_and_clean_notes(df):
    df = df.copy()

    df["clinical_note_text"] = df["clinical_note_text"].fillna("").astype(str)

    df = df[df["clinical_note_text"].str.len() > 100]

    template_heavy = (
        df["clinical_note_text"]
        .str.lower()
        .str.contains(r"discharge date|admission date|date of birth", na=False)
    )

    df = df[~template_heavy]

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

    strong = re.search(
        r"\b(generalized anxiety disorder|panic disorder|severe anxiety|anxiety disorder)\b",
        text,
    )

    moderate = re.search(
        r"\b(anxiety|panic|anxious)\b",
        text,
    )

    negated = re.search(
        r"\b(no anxiety|denies anxiety|without anxiety|negative for anxiety)\b",
        text,
    )

    past = re.search(
        r"\b(history of anxiety|hx anxiety|past anxiety)\b",
        text,
    )

    stable = re.search(
        r"\b(stable anxiety|controlled anxiety|treated anxiety)\b",
        text,
    )

    if negated:
        return 0.4, "negated"

    if strong:
        return 1.0, "active"

    if moderate:
        if stable:
            return 0.7, "stable"
        if past:
            return 0.65, "past"
        return 0.8, "active"

    return 0.5, "unspecified"


# =============================================================================
# CONTROL PENALTY ENGINE
# =============================================================================
def penalize_control_noise(text):
    if not isinstance(text, str):
        return 1.0

    text = text.lower()

    if re.search(r"\b(anxiety disorder|panic disorder)\b", text):
        return 0.2

    if re.search(r"\b(anxiety|panic|anxious)\b", text):
        return 0.5

    return 1.0
