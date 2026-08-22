import pandas as pd
import re


# =========================================================
# SAFE CSV LOADER
# =========================================================
def load_csv_safe(path, usecols=None):
    try:
        df = pd.read_csv(path, usecols=usecols, compression="gzip")
        print(f"  ✓ Loaded: {path.name} ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"  ❌ Failed to load {path}: {e}")
        return None


# =========================================================
# IDENTIFY ANXIETY PATIENTS
# =========================================================
def identify_anxiety_patients(diagnoses):

    anxiety_codes = set(
        [
            "F410",
            "F411",
            "F412",
            "F413",
            "F418",
            "F419",
            "F4000",
            "F4001",
            "F4010",
            "F4011",
            "F408",
            "F409",
            "30000",
            "30001",
            "30002",
            "30009",
            "30020",
            "30021",
            "30023",
        ]
    )

    diagnoses = diagnoses.copy()

    diagnoses["code_clean"] = (
        diagnoses["icd_code"].astype(str).str.replace(".", "", regex=False).str.strip()
    )

    anxiety = diagnoses[diagnoses["code_clean"].isin(anxiety_codes)].copy()
    anxiety["has_anxiety"] = 1

    return anxiety[["subject_id", "hadm_id", "has_anxiety"]].drop_duplicates()


# =========================================================
# IDENTIFY CONTROL PATIENTS (STRONGER VERSION)
# =========================================================
def identify_control_patients(diagnoses, anxiety_cases):

    anxiety_subjects = set(anxiety_cases["subject_id"])

    # 🔥 IMPORTANT: remove ANY patient who EVER had anxiety
    controls = diagnoses[~diagnoses["subject_id"].isin(anxiety_subjects)].copy()

    controls["has_anxiety"] = 0

    return controls[["subject_id", "hadm_id", "has_anxiety"]].drop_duplicates()


# =========================================================
# CLEAN NOTE TEXT (IMPROVED)
# =========================================================
def clean_note_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove PHI placeholders
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)

    # normalize
    text = re.sub(r"\s+", " ", text)

    # keep medical punctuation
    text = re.sub(r"[^a-z0-9.,!? ]", "", text)

    return text.strip()


# =========================================================
# TEMPORAL FEATURES
# =========================================================
def compute_temporal_features(df):

    df = df.copy()

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.sort_values(["subject_id", "charttime"])

    df["visit_number"] = df.groupby("subject_id").cumcount() + 1
    df["total_visits"] = df.groupby("subject_id")["note_id"].transform("count")

    df["days_since_first_visit"] = (
        df["charttime"] - df.groupby("subject_id")["charttime"].transform("min")
    ).dt.days

    df["days_since_last_visit"] = df.groupby("subject_id")["charttime"].shift(1)
    df["days_since_last_visit"] = (
        df["charttime"] - df["days_since_last_visit"]
    ).dt.days

    df["note_age_days"] = (df["charttime"] - df["charttime"].min()).dt.days

    df["is_most_recent"] = (
        df.groupby("subject_id")["charttime"].transform("max") == df["charttime"]
    )

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


# =========================================================
# VERIFY NOTES (STRONGER FILTER)
# =========================================================
def verify_and_clean_notes(df):

    df = df.copy()

    # remove short notes
    df = df[df["clinical_note_text"].str.len() > 50]

    # remove template-heavy notes (very important)
    # df = df[~df["clinical_note_text"].str.contains("discharge date", na=False)]

    return df
