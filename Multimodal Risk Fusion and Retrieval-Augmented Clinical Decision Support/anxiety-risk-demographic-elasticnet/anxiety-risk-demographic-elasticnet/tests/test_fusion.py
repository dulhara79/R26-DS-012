import math
import pytest

from anxiety_risk.fusion import probability_to_fusion_score, fusion_score_to_probability


def test_population_baseline_maps_to_half():
    prevalence = 0.0223
    assert probability_to_fusion_score(prevalence, prevalence) == pytest.approx(0.5, abs=1e-12)


def test_above_baseline_maps_above_half_and_below_maps_below_half():
    prevalence = 0.0223
    assert probability_to_fusion_score(0.08, prevalence) > 0.5
    assert probability_to_fusion_score(0.008, prevalence) < 0.5


def test_fusion_transform_is_reversible():
    prevalence = 0.0223
    p = 0.061
    score = probability_to_fusion_score(p, prevalence)
    assert fusion_score_to_probability(score, prevalence) == pytest.approx(p, rel=1e-10)


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        probability_to_fusion_score(1.2, 0.02)
