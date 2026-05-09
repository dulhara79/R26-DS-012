"""
extraction_v4.py
TC-WPN Research Pipeline — Publication-Grade Data Extraction Helpers
Author: Dulhara Kaushalya

KEY CHANGES FROM extraction_v3.py:
1. MIMIC-III NOTEEVENTS support added.
   - Filters by CATEGORY to keep only high-signal note types:
     Psychiatry, Nursing/other, Physician, Social Work, Discharge summary
   - MIMIC-III uses ICD-9 codes only (no ICD-10)
   - Subject IDs do NOT overlap with MIMIC-IV — safe to combine

2. Prescription-based confidence boosting added.
   - Patients with anxiety ICD code AND anxiety medication get
     label_confidence = 1.0 (clinician-confirmed via prescription)
   - Works for both MIMIC-III (PRESCRIPTIONS.csv.gz) and
     MIMIC-IV (prescriptions.csv.gz)

3. MIMIC-III-Ext-Notes negation integration.
   - labels.csv from Ext-Notes provides ground-truth negation annotations
   - Used to validate and calibrate regex-based negation detection
   - Also used to build a negation keyword list from confirmed negated concepts

4. OMR-based GAD-7/PHQ gold label extraction (MIMIC-IV only).
   - omr.csv.gz contains structured outpatient measurements
   - GAD-7 >= 10 or PHQ >= 10 → label_confidence = 1.0 (gold standard)

5. has_psychiatric_content() threshold lowered from 2 to 1 for
   MIMIC-III psychiatry/social work notes — these are inherently
   psychiatric so a lower keyword threshold is appropriate.

6. Note source priority weighting added to training_weight:
   Psychiatry notes → ×1.3 bonus
   Social Work notes → ×1.2 bonus
   Physician notes  → ×1.1 bonus
   Nursing notes    → ×1.0 (baseline)
   Discharge summary → ×0.9 (lowest — most diluted)
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
# ICD CODE DEFINITIONS — MIMIC-IV (ICD-10)
# =============================================================================
ANXIETY_ICD10_PREFIXES = ["F40", "F41"]

EXCLUSION_MENTAL_PREFIXES_ICD10 = [
    "F20",
    "F25",
    "F31",
    "F32",
    "F33",
    "F43",
]

# =============================================================================
# ICD CODE DEFINITIONS — MIMIC-III (ICD-9)
# =============================================================================
ANXIETY_ICD9_PREFIXES = ["3000", "3002"]

EXCLUSION_MENTAL_PREFIXES_ICD9 = [
    "296",  # Mood disorders (bipolar, MDD)
    "295",  # Schizophrenic disorders
    "311",  # Depressive disorder NOS
    "3090",  # Adjustment disorder
    "3091",  # Prolonged depressive reaction
    "309",  # Adjustment reactions
]

# =============================================================================
# MIMIC-III NOTE CATEGORIES — high psychiatric signal
# Reference: MIMIC-III NOTEEVENTS.csv CATEGORY column values
# =============================================================================
MIMIC3_HIGH_SIGNAL_CATEGORIES = [
    "Psychiatry",  # Direct psychiatric assessment — highest signal
    "Social Work",  # Contains GAD-7, PHQ, psychosocial assessments
    "Physician",  # Progress notes with MSE sections
    "Nursing/other",  # Daily longitudinal notes — ideal for temporal modeling
    "Nursing",  # Some versions use this spelling
    "Discharge summary",  # Lower signal but high volume
]

# =============================================================================
# ANXIETY MEDICATIONS — for prescription-based confidence boosting
# Reference: MIMIC-IV prescriptions.csv.gz and MIMIC-III PRESCRIPTIONS.csv.gz
# drug column contains drug names (lowercase matching safe)
# =============================================================================
ANXIETY_MEDICATIONS = [
    # SSRIs
    "sertraline",
    "escitalopram",
    "fluoxetine",
    "paroxetine",
    "citalopram",
    "fluvoxamine",
    # SNRIs
    "venlafaxine",
    "duloxetine",
    "desvenlafaxine",
    # Azapirones
    "buspirone",
    # Benzodiazepines (short-term anxiety)
    "lorazepam",
    "clonazepam",
    "alprazolam",
    "diazepam",
    "oxazepam",
    "temazepam",
    # Other
    "hydroxyzine",  # antihistamine used for anxiety
    "pregabalin",  # GAD treatment
]

# =============================================================================
# SOURCE TYPE PRIORITY WEIGHTS
# Higher = more likely to contain dense psychiatric signal
# =============================================================================
SOURCE_PRIORITY_WEIGHTS = {
    "psychiatry": 1.3,
    "social_work": 1.2,
    "physician": 1.1,
    "nursing": 1.0,
    "nursing/other": 1.0,
    "discharge": 0.9,
    "discharge_detail": 0.85,
    "radiology": 0.7,
    "unknown": 1.0,
}


# =============================================================================
# PSYCHIATRIC CONTENT GATE
# =============================================================================
PSYCH_KEYWORDS = [
    "anxiety",
    "anxious",
    "panic",
    "generalized anxiety",
    "gad",
    "worry",
    "worries",
    "psychiatric",
    "psychiatry",
    "mental status",
    "mental health",
    "phobia",
    "phobic",
    "mood",
    "affect",
    "depression",
    "depressed",
    "ssri",
    "snri",
    "benzodiazepine",
    "buspirone",
    "psychotherapy",
    "cbt",
    "cognitive behavioral",
    "axis i",
    "dsm",
    "phq",
    "gad-7",
]


def has_psychiatric_content(text, min_keywords=2):
    if not isinstance(text, str) or len(text) < 100:
        return False
    count = sum(1 for kw in PSYCH_KEYWORDS if kw in text)
    return count >= min_keywords


# =============================================================================
# PRIMARY DIAGNOSIS FILTER
# =============================================================================
def filter_primary_anxiety_admissions(diagnoses, max_seq_num=3):
    diagnoses = diagnoses.copy()
    diagnoses["code_clean"] = (
        diagnoses["icd_code"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.upper()
    )
    is_anxiety = diagnoses["code_clean"].apply(
        lambda x: isinstance(x, str)
        and any(x.startswith(p) for p in ANXIETY_ICD10_PREFIXES + ANXIETY_ICD9_PREFIXES)
    )
    is_primary = diagnoses["seq_num"] <= max_seq_num
    primary_hadm = set(diagnoses[is_anxiety & is_primary]["hadm_id"].unique())
    print(
        f"  Primary anxiety admissions (seq_num ≤ {max_seq_num}): {len(primary_hadm):,}"
    )
    return primary_hadm


# =============================================================================
# IDENTIFY ANXIETY PATIENTS — MIMIC-IV (ICD-10) and MIMIC-III (ICD-9)
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
        lambda x: isinstance(x, str)
        and any(x.startswith(p) for p in ANXIETY_ICD10_PREFIXES + ANXIETY_ICD9_PREFIXES)
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

    all_exclusion = EXCLUSION_MENTAL_PREFIXES_ICD10 + EXCLUSION_MENTAL_PREFIXES_ICD9
    psych_exclusion = controls["code_clean"].apply(
        lambda x: isinstance(x, str) and any(x.startswith(p) for p in all_exclusion)
    )
    controls = controls[~psych_exclusion].copy()
    controls["has_anxiety"] = 0
    return controls[["subject_id", "hadm_id", "has_anxiety"]].drop_duplicates()


# =============================================================================
# PRESCRIPTION-BASED CONFIDENCE BOOSTING  (NEW v4)
#
# Joins prescriptions table to identify patients with anxiety ICD code
# AND anxiety medication prescription. These are clinician-confirmed
# anxiety cases → label_confidence = 1.0
#
# Works for both MIMIC-IV (prescriptions.csv.gz) and
# MIMIC-III (PRESCRIPTIONS.csv.gz).
# Column name is 'drug' in both versions.
# =============================================================================
def load_prescription_confirmed_subjects(prescriptions_path):
    """
    Returns a set of subject_ids who were prescribed an anxiety medication.
    These subjects' notes get label_confidence boosted to 1.0 if they
    also have an anxiety ICD diagnosis.

    prescriptions_path: Path to prescriptions.csv.gz (MIMIC-IV) or
                        PRESCRIPTIONS.csv.gz (MIMIC-III)
    """
    print(f"\nLoading prescriptions from {prescriptions_path.name}...")
    try:
        # MIMIC-IV uses lowercase columns (subject_id, drug).
        # MIMIC-III uses UPPERCASE (SUBJECT_ID, DRUG).
        # Use a lambda usecols filter to handle both versions automatically.
        df = pd.read_csv(
            prescriptions_path,
            usecols=lambda c: c.upper() in ("SUBJECT_ID", "DRUG"),
            compression="gzip",
            low_memory=False,
        )
        print(f"  Loaded {len(df):,} prescription records")

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        if "subject_id" not in df.columns or "drug" not in df.columns:
            print(f"  ⚠ Could not find subject_id/drug columns — skipping boost")
            return set()

        # Fill NaN drug names before string operations
        df["drug_lower"] = df["drug"].fillna("").astype(str).str.lower().str.strip()

        # Match any row where drug name contains an anxiety medication keyword
        med_mask = df["drug_lower"].apply(
            lambda d: bool(d) and any(med in d for med in ANXIETY_MEDICATIONS)
        )
        confirmed = set(df[med_mask]["subject_id"].unique())
        print(f"  ✓ Prescription-confirmed anxiety subjects: {len(confirmed):,}")
        return confirmed

    except Exception as e:
        print(f"  ⚠ Could not load prescriptions: {e} — skipping boost")
        return set()


# =============================================================================
# OMR-BASED GOLD LABEL EXTRACTION — MIMIC-IV only  (NEW v4)
#
# omr.csv.gz contains outpatient measurement records.
# Some rows contain GAD-7 and PHQ scores as structured text.
# GAD-7 >= 10 or PHQ-9 >= 10 = moderate-to-severe anxiety/depression.
# These subject_ids get label_confidence = 1.0
#
# Column: result_name (e.g. "GAD-7 Total Score"), result_value (numeric string)
# =============================================================================
def load_omr_gold_subjects(omr_path):
    """
    Returns a set of subject_ids with clinically significant GAD-7 or PHQ
    scores from MIMIC-IV OMR (Outpatient Medical Records).
    """
    print(f"\nLoading OMR scores from {omr_path.name}...")
    try:
        df = pd.read_csv(
            omr_path,
            usecols=["subject_id", "result_name", "result_value"],
            compression="gzip",
            low_memory=False,
        )
        print(f"  Loaded {len(df):,} OMR records")

        df["result_name_lower"] = df["result_name"].astype(str).str.lower()
        df["result_value_num"] = pd.to_numeric(df["result_value"], errors="coerce")

        # GAD-7 >= 10 (moderate-to-severe anxiety)
        gad_mask = df["result_name_lower"].str.contains("gad", na=False) & (
            df["result_value_num"] >= 10
        )
        # PHQ-9 >= 10 (moderate depression — often comorbid)
        phq_mask = df["result_name_lower"].str.contains("phq", na=False) & (
            df["result_value_num"] >= 10
        )

        gold_subjects = set(df[gad_mask | phq_mask]["subject_id"].unique())
        print(
            f"  ✓ OMR gold-label subjects (GAD-7≥10 or PHQ≥10): {len(gold_subjects):,}"
        )
        return gold_subjects

    except Exception as e:
        print(f"  ⚠ Could not load OMR: {e} — skipping gold labels")
        return set()


# =============================================================================
# MIMIC-III-EXT-NOTES NEGATION INTEGRATION  (NEW v4)
#
# labels.csv from MIMIC-III-Ext-Notes contains clinician-annotated
# negation status for 2,288 clinical concepts.
# We use this to:
# 1. Extract confirmed negation trigger words → expand our regex patterns
# 2. Validate our regex-based negation detection
#
# File columns: row_id, trigger_word, concept, semtypes,
#               start, end, detection, encounter, negation
# =============================================================================
def load_ext_notes_negation_patterns(ext_notes_labels_path):
    """
    Extracts negation trigger words from MIMIC-III-Ext-Notes labels.csv
    where negation == 'yes'.
    Returns a set of lowercase trigger words for use in negation detection.
    """
    print(f"\nLoading Ext-Notes negation patterns...")
    try:
        df = pd.read_csv(ext_notes_labels_path, low_memory=False)
        negated = df[df["negation"].astype(str).str.lower() == "yes"]
        triggers = set(negated["trigger_word"].astype(str).str.lower().unique())
        print(f"  ✓ Negation trigger words from Ext-Notes: {len(triggers)}")
        return triggers
    except Exception as e:
        print(
            f"  ⚠ Could not load Ext-Notes labels: {e} — using default negation patterns"
        )
        return set()


# =============================================================================
# NOTE CLEANING
# =============================================================================
def clean_note_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # MIMIC-IV PHI placeholders: [**Name**]
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    # MIMIC-III PHI placeholders: [**Name**] same format
    header_patterns = [
        r"admission date\s*:",
        r"discharge date\s*:",
        r"date of birth\s*:",
        r"service\s*:",
        r"sex\s*:",
        r"attending\s*:",
        r"allergies\s*:",
        r"unit no\s*:",
        r"name\s*:",
    ]
    for pat in header_patterns:
        text = re.sub(pat, " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,!?;:\-\(\) ]", "", text)
    return text.strip()


# =============================================================================
# SECTION QUALITY SCORING
# =============================================================================
def compute_section_quality(text, source_type="unknown"):
    if not isinstance(text, str) or len(text) < 200:
        return 0.3

    score = 0.5

    high_value_sections = [
        "history of present illness",
        "past psychiatric history",
        "assessment",
        "mental status examination",
        "psychiatric",
        "axis i",
        "mood and affect",
        "social work",  # MIMIC-III Social Work notes
        "mental health",
    ]
    anxiety_content_signals = [
        "anxiety",
        "panic",
        "gad-7",
        "phq",
        "worry",
        "generalized anxiety disorder",
        "anxious",
    ]

    for sec in high_value_sections:
        if sec in text:
            score += 0.1

    anxiety_hits = sum(1 for kw in anxiety_content_signals if kw in text)
    score += min(0.2, anxiety_hits * 0.05)

    # Apply source type bonus
    src_key = source_type.lower().replace(" ", "_")
    src_bonus = SOURCE_PRIORITY_WEIGHTS.get(src_key, 1.0)
    score = score * src_bonus

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
    df = df.drop_duplicates(subset=["subject_id", "charttime", "clinical_note_text"])
    return df


# =============================================================================
# LABEL CONFIDENCE ENGINE — ANXIETY
# =============================================================================
def assign_anxiety_confidence(
    text, subject_id=None, prescription_confirmed_subjects=None, omr_gold_subjects=None
):
    """
    v4: checks prescription_confirmed_subjects and omr_gold_subjects
    before text-based heuristics. These are the highest-confidence signals.
    """
    # Gold standard overrides — highest priority
    if omr_gold_subjects and subject_id and subject_id in omr_gold_subjects:
        return 1.0, "omr_gold"

    if (
        prescription_confirmed_subjects
        and subject_id
        and subject_id in prescription_confirmed_subjects
    ):
        return 1.0, "prescription_confirmed"

    # Text-based heuristics (unchanged from v3)
    if not isinstance(text, str):
        return 0.5, "unspecified"
    text = text.lower()

    if re.search(
        r"\b(family history of anxiety|fh:\s*anxiety|fh anxiety|"
        r"mother has anxiety|father has anxiety|"
        r"family hx of anxiety|family h/o anxiety)\b",
        text,
    ):
        return 0.4, "family_history"

    if re.search(
        r"\b(anxious about|anxiety about|nervous about|"
        r"anxious regarding|anxiety regarding|"
        r"anxious to (?:go home|leave|return|be discharged)|"
        r"pre-?operative anxiety|procedural anxiety)\b",
        text,
    ):
        return 0.45, "situational"

    if re.search(
        r"\b(no anxiety|denies anxiety|without anxiety|"
        r"negative for anxiety|no panic|denies panic|"
        r"no evidence of anxiety)\b",
        text,
    ):
        return 0.4, "negated"

    if re.search(
        r"\b(generalized anxiety disorder|panic disorder|"
        r"anxiety disorder|severe anxiety|gad\b)\b",
        text,
    ):
        return 1.0, "active"

    if re.search(
        r"\b(stable anxiety|controlled anxiety|treated anxiety|"
        r"anxiety controlled|anxiety stable|anxiety managed)\b",
        text,
    ):
        return 0.7, "stable"

    if re.search(
        r"\b(history of anxiety|hx anxiety|h/o anxiety|"
        r"past anxiety|prior anxiety|previous anxiety)\b",
        text,
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
        r"anxiety disorder confirmed|gad diagnosis)\b",
        text,
    ):
        return 0.2

    if re.search(
        r"(assessment|impression|diagnosis|axis i|psychiatric evaluation)"
        r"[^.]{0,80}\b(anxiety|panic)\b",
        text,
    ):
        return 0.6

    return 1.0
