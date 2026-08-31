import numpy as np
import pandas as pd

from anxiety_risk.model import build_elasticnet_pipeline


def test_pipeline_returns_probabilities_for_demographic_inputs():
    X = pd.DataFrame({
        "age": [18,19,20,21,22,23,24,25,26,27],
        "gender": ["female","male"] * 5,
        "edu": ["bachelor's degree"] * 10,
        "smoke": ["never smokes"] * 8 + ["current smoker (cumulative smoking >10 packs)"] * 2,
        "drink": ["never drinks"] * 7 + ["current regular drinker (more than once a week)"] * 3,
    })
    y = np.array([0,0,0,0,0,0,0,1,0,1])
    model = build_elasticnet_pipeline(C=0.5, l1_ratio=0.5, class_weight="balanced")
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (10,)
    assert np.all((probs >= 0) & (probs <= 1))
