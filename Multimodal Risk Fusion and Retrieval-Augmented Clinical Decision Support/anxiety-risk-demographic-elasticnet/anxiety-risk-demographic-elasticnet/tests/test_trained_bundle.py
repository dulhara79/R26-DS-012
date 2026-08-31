from pathlib import Path

from anxiety_risk.predict import load_bundle, predict_patient


def test_trained_bundle_predicts_probability_and_fusion_score():
    path = Path(__file__).resolve().parents[1] / "models/demographic_elasticnet_calibrated.joblib"
    bundle = load_bundle(path)
    result = predict_patient(
        bundle,
        age=22,
        gender="female",
        education="bachelor's degree",
        smoking="current smoker (cumulative smoking >10 packs)",
        drinking="current regular drinker (more than once a week)",
    )
    assert 0 < result["demographic_probability"] < 1
    assert result["demographic_fusion_score"] > 0.5
    assert bundle["isi_used"] is False
