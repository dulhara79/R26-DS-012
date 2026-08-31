from __future__ import annotations

import math

_EPS = 1e-12


def _validate_unit_interval(value: float, name: str) -> None:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")


def _logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def probability_to_fusion_score(probability: float, reference_prevalence: float) -> float:
    """Map calibrated probability to baseline-adjusted 0-1 fusion evidence.

    A probability equal to the training prevalence maps to 0.5.
    Above-baseline risk maps above 0.5; below-baseline risk maps below 0.5.
    This value is an evidence index, not an anxiety probability.
    """
    _validate_unit_interval(probability, "probability")
    _validate_unit_interval(reference_prevalence, "reference_prevalence")
    evidence = _logit(probability) - _logit(reference_prevalence)
    return _sigmoid(evidence)


def fusion_score_to_probability(fusion_score: float, reference_prevalence: float) -> float:
    _validate_unit_interval(fusion_score, "fusion_score")
    _validate_unit_interval(reference_prevalence, "reference_prevalence")
    return _sigmoid(_logit(fusion_score) + _logit(reference_prevalence))
