from __future__ import annotations

import pandas as pd

GAD_QUESTIONS = [f"question{i}" for i in range(1, 8)]
FEATURE_COLUMNS = ["age", "gender", "edu", "smoke", "drink"]


def validate_gad7(gad7: pd.DataFrame) -> pd.DataFrame:
    required = ["export_id", "score", *GAD_QUESTIONS]
    missing = [c for c in required if c not in gad7.columns]
    if missing:
        raise ValueError(f"Missing GAD-7 columns: {missing}")
    if gad7["export_id"].isna().any() or gad7["export_id"].duplicated().any():
        raise ValueError("GAD-7 export_id values must be non-null and unique")
    for col in GAD_QUESTIONS:
        bad = ~gad7[col].isin([0, 1, 2, 3])
        if bad.any():
            raise ValueError(f"Invalid GAD-7 item values in {col}")
    out = gad7.copy()
    out["gad7_recalculated"] = out[GAD_QUESTIONS].sum(axis=1)
    if not (out["gad7_recalculated"] == out["score"]).all():
        raise ValueError("GAD-7 score mismatch detected")
    out["anxiety_positive"] = (out["score"] >= 10).astype(int)
    return out


def build_master_dataset(
    demographic: pd.DataFrame,
    gad7: pd.DataFrame,
    min_age: float = 15,
    max_age: float = 65,
) -> pd.DataFrame:
    required_demo = ["export_id", *FEATURE_COLUMNS]
    missing = [c for c in required_demo if c not in demographic.columns]
    if missing:
        raise ValueError(f"Missing demographic columns: {missing}")
    if demographic["export_id"].isna().any() or demographic["export_id"].duplicated().any():
        raise ValueError("Demographic export_id values must be non-null and unique")
    validated = validate_gad7(gad7)
    target = validated[["export_id", "score", "anxiety_positive"]].rename(columns={"score": "gad7_score"})
    merged = demographic[required_demo].merge(target, on="export_id", how="inner", validate="one_to_one")
    merged = merged[merged["age"].between(min_age, max_age, inclusive="both")].copy()
    return merged[["export_id", *FEATURE_COLUMNS, "gad7_score", "anxiety_positive"]].reset_index(drop=True)
