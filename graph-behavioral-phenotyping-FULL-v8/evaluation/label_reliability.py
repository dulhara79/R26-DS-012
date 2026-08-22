"""Reference calculation for imperfect binary-label reliability."""


def label_reliability_reference(se, sp, prevalence):
    p1 = prevalence * se / (
        prevalence * se + (1 - prevalence) * (1 - sp)
    )
    p0 = prevalence * (1 - se) / (
        prevalence * (1 - se) + (1 - prevalence) * sp
    )
    return float(
        0.5 * (p1 * p0 + (1 - p1) * (1 - p0))
        + p1 * (1 - p0)
    )


def fraction_above_chance(observed_auroc, reference_auroc):
    return float((observed_auroc - 0.5) / (reference_auroc - 0.5))
