from __future__ import annotations

import pytest

from care_anxrag.statistics import (
    mcnemar_exact,
    paired_bootstrap_difference,
)


def test_paired_bootstrap_difference_is_deterministic() -> None:
    left = [0.2, 0.4, 0.6, 0.8]
    right = [0.3, 0.5, 0.7, 0.9]

    first = paired_bootstrap_difference(left, right, samples=2000, seed=42)
    second = paired_bootstrap_difference(left, right, samples=2000, seed=42)

    assert first == second
    assert first["n"] == 4
    assert first["difference"] == pytest.approx(0.1)
    assert first["ci95_low"] == pytest.approx(0.1)
    assert first["ci95_high"] == pytest.approx(0.1)


def test_paired_bootstrap_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])


def test_mcnemar_exact_reports_discordant_pairs() -> None:
    result = mcnemar_exact(
        [True, True, False, False, False],
        [True, False, True, True, False],
    )

    assert result["n"] == 5
    assert result["left_only_correct"] == 1
    assert result["right_only_correct"] == 2
    assert result["discordant_pairs"] == 3
    assert result["p_value"] == 1.0


def test_mcnemar_exact_zero_discordance() -> None:
    result = mcnemar_exact([True, False], [True, False])

    assert result["discordant_pairs"] == 0
    assert result["p_value"] == 1.0
