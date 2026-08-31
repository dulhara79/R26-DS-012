from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from .fusion import probability_to_fusion_score

GENDERS = {"female", "male"}
EDUCATION = {"bachelor's degree", "associate degree", "master's degree", "doctorate degree"}
SMOKING = {
    "never smokes",
    "occasional smoker (cumulative smoking <10 packs)",
    "current smoker (cumulative smoking >10 packs)",
    "former smoker (cumulative smoking >10 packs), but not in the past year",
}
DRINKING = {
    "never drinks",
    "drinks occasionally (less than once a week)",
    "drank in the past (more than once a week), but not in the past year",
    "current regular drinker (more than once a week)",
}


def validate_patient_input(age, gender, education, smoking, drinking) -> pd.DataFrame:
    age = float(age)
    if not 15 <= age <= 65:
        raise ValueError("age must be within the model training range 15-65")
    if gender not in GENDERS:
        raise ValueError(f"Unknown gender category: {gender}")
    if education not in EDUCATION:
        raise ValueError(f"Unknown education category: {education}")
    if smoking not in SMOKING:
        raise ValueError(f"Unknown smoking category: {smoking}")
    if drinking not in DRINKING:
        raise ValueError(f"Unknown drinking category: {drinking}")
    return pd.DataFrame([{
        "age": age,
        "gender": gender,
        "edu": education,
        "smoke": smoking,
        "drink": drinking,
    }])


def load_bundle(path: str | Path):
    return joblib.load(path)


def predict_patient(bundle, age, gender, education, smoking, drinking) -> dict:
    row = validate_patient_input(age, gender, education, smoking, drinking)
    probability = float(bundle["model"].predict_proba(row)[:, 1][0])
    fusion_score = probability_to_fusion_score(probability, float(bundle["reference_prevalence"]))
    return {
        "demographic_probability": probability,
        "demographic_probability_percent": probability * 100.0,
        "demographic_fusion_score": fusion_score,
        "reference_prevalence": float(bundle["reference_prevalence"]),
        "target": "GAD-7 >= 10",
        "model": "calibrated_elastic_net_logistic_regression",
    }
