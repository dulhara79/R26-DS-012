import numpy as np

from anxiety_risk.predict import validate_patient_input


def test_validate_patient_input_accepts_known_categories():
    row = validate_patient_input(
        age=22,
        gender="female",
        education="bachelor's degree",
        smoking="never smokes",
        drinking="never drinks",
    )
    assert row.shape == (1, 5)
    assert row.iloc[0]["age"] == 22


def test_validate_patient_input_rejects_age_outside_training_range():
    try:
        validate_patient_input(
            age=80,
            gender="female",
            education="bachelor's degree",
            smoking="never smokes",
            drinking="never drinks",
        )
    except ValueError as exc:
        assert "age" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for out-of-range age")


def test_predict_patient_returns_probability_and_fusion_score_without_binary_threshold():
    import pandas as pd
    from anxiety_risk.model import build_elasticnet_pipeline
    from anxiety_risk.predict import predict_patient

    X = pd.DataFrame({
        "age": [18,19,20,21,22,23,24,25,26,27,28,29],
        "gender": ["female","male"] * 6,
        "edu": ["bachelor's degree"] * 12,
        "smoke": ["never smokes"] * 10 + ["current smoker (cumulative smoking >10 packs)"] * 2,
        "drink": ["never drinks"] * 9 + ["current regular drinker (more than once a week)"] * 3,
    })
    y = [0,0,0,0,0,0,0,0,1,0,1,1]
    model = build_elasticnet_pipeline(C=1.0, l1_ratio=0.75, class_weight=None).fit(X, y)
    bundle = {"model": model, "reference_prevalence": 0.25}
    result = predict_patient(
        bundle,
        age=22,
        gender="female",
        education="bachelor's degree",
        smoking="never smokes",
        drinking="never drinks",
    )
    assert 0 < result["demographic_probability"] < 1
    assert 0 < result["demographic_fusion_score"] < 1
    assert "screen_positive" not in result
