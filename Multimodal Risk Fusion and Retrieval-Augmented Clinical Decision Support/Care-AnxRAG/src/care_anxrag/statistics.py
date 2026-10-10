from __future__ import annotations

import math
import random
from typing import Iterable


def paired_bootstrap_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    samples: int = 10000,
    seed: int = 20260927,
) -> dict[str, float | int]:
    """Estimate the paired mean difference (right minus left) with a bootstrap CI."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("Paired samples must have the same length")
    if not left_values:
        raise ValueError("Paired samples must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")

    differences = [
        right_value - left_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ]
    observed = sum(differences) / len(differences)

    rng = random.Random(seed)
    bootstrap: list[float] = []
    count = len(differences)
    for _ in range(samples):
        bootstrap.append(
            sum(differences[rng.randrange(count)] for _ in range(count)) / count
        )
    bootstrap.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(bootstrap) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return bootstrap[lower]
        fraction = position - lower
        return bootstrap[lower] * (1.0 - fraction) + bootstrap[upper] * fraction

    return {
        "n": count,
        "left_mean": sum(left_values) / count,
        "right_mean": sum(right_values) / count,
        "difference": observed,
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def mcnemar_exact(
    left_correct: Iterable[bool],
    right_correct: Iterable[bool],
) -> dict[str, float | int]:
    """Two-sided exact McNemar test for paired binary correctness outcomes."""
    left = [bool(value) for value in left_correct]
    right = [bool(value) for value in right_correct]
    if len(left) != len(right):
        raise ValueError("Paired outcomes must have the same length")
    if not left:
        raise ValueError("Paired outcomes must not be empty")

    left_only = sum(
        left_value and not right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    right_only = sum(
        right_value and not left_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    discordant = left_only + right_only

    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(left_only, right_only)
        cumulative = sum(
            math.comb(discordant, k)
            for k in range(tail + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * cumulative)

    return {
        "n": len(left),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }
