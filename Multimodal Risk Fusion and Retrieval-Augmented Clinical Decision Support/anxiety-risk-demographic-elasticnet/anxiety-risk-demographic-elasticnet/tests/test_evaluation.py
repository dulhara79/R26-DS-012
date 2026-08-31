import numpy as np
import pytest

from anxiety_risk.evaluation import choose_threshold_for_sensitivity


def test_threshold_meets_target_sensitivity_and_prefers_specificity():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.7, 0.5, 0.2, 0.1])
    result = choose_threshold_for_sensitivity(y, p, target_sensitivity=1.0)
    assert result["threshold"] == pytest.approx(0.6)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(0.75)
