"""GLOBEM cohort loading utilities."""
from pathlib import Path
import pandas as pd

from config import COHORTS, GLOBEM_ROOT, LABEL_FILE


def load_cohort(cohort: str, root: Path = GLOBEM_ROOT, usecols=None):
    feature_path = Path(root) / cohort / "FeatureData" / "rapids.csv"
    label_path = Path(root) / cohort / "SurveyData" / LABEL_FILE

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    feats = pd.read_csv(feature_path, usecols=usecols, low_memory=False)
    labels = pd.read_csv(label_path, low_memory=False)

    for df in (feats, labels):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    feats["cohort"] = cohort
    labels["cohort"] = cohort
    feats["uid"] = cohort + "_" + feats["pid"].astype(str)
    labels["uid"] = cohort + "_" + labels["pid"].astype(str)

    return feats, labels


def load_all_cohorts(cohorts=COHORTS, root: Path = GLOBEM_ROOT, usecols=None):
    feature_frames, label_frames = [], []
    for cohort in cohorts:
        f, l = load_cohort(cohort, root=root, usecols=usecols)
        feature_frames.append(f)
        label_frames.append(l)
    return (
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(label_frames, ignore_index=True),
    )
