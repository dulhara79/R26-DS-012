import pandas as pd
import pytest

from anxiety_risk.data import validate_gad7, build_master_dataset


def test_validate_gad7_reconstructs_score_and_target():
    gad = pd.DataFrame({
        "export_id": [1, 2],
        "score": [10, 3],
        "question1": [1, 0],
        "question2": [2, 0],
        "question3": [1, 0],
        "question4": [1, 0],
        "question5": [2, 1],
        "question6": [1, 2],
        "question7": [2, 0],
    })
    out = validate_gad7(gad)
    assert out["gad7_recalculated"].tolist() == [10, 3]
    assert out["anxiety_positive"].tolist() == [1, 0]


def test_validate_gad7_rejects_score_mismatch():
    gad = pd.DataFrame({
        "export_id": [1],
        "score": [11],
        "question1": [1], "question2": [2], "question3": [1],
        "question4": [1], "question5": [2], "question6": [1], "question7": [2],
    })
    with pytest.raises(ValueError, match="GAD-7 score mismatch"):
        validate_gad7(gad)


def test_build_master_dataset_uses_only_demographic_features_and_target():
    demo = pd.DataFrame({
        "export_id": [1, 2, 3],
        "gender": ["female", "male", "female"],
        "age": [20.0, 22.0, 99.0],
        "edu": ["bachelor's degree", "master's degree", "bachelor's degree"],
        "smoke": ["never smokes", "never smokes", "never smokes"],
        "drink": ["never drinks", "never drinks", "never drinks"],
    })
    gad = pd.DataFrame({
        "export_id": [1, 2, 3], "score": [0, 10, 4],
        "question1": [0,2,0], "question2": [0,2,0], "question3": [0,2,0],
        "question4": [0,1,1], "question5": [0,1,1], "question6": [0,1,1], "question7": [0,1,1],
    })
    out = build_master_dataset(demo, gad, min_age=15, max_age=65)
    assert list(out.columns) == [
        "export_id", "age", "gender", "edu", "smoke", "drink",
        "gad7_score", "anxiety_positive"
    ]
    assert out["export_id"].tolist() == [1, 2]
    assert out["anxiety_positive"].tolist() == [0, 1]
