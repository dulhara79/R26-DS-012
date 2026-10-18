"""Resting-baseline validation shared by calibration and inference."""

import numpy as np


MINIMUM_BASELINE_WINDOWS = 10
# Temperature is retained in the saved ten-field baseline but is not a
# requirement for calibration variation or a model-scoring feature.
NON_TEMPERATURE_INDICES = (0, 1, 2, 3, 4, 5, 8, 9)
MINIMUM_BASELINE_STDS = np.array(
    [1.0, 15.0, 2.0, 2.0, 0.5, 0.10, 0.05, 0.01, 0.01, 0.005],
    dtype=np.float32,
)


def baseline_statistics(windows):
    """Estimate minute-level variation from real resting minutes, not padding."""
    values = np.asarray(windows, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(MINIMUM_BASELINE_STDS)
        or len(values) < MINIMUM_BASELINE_WINDOWS
    ):
        raise ValueError(
            "Recalibrate while resting: at least ten complete one-minute "
            "baseline windows of ten features are required."
        )
    if not np.isfinite(values).all():
        raise ValueError("Baseline windows must contain only finite numbers.")
    measured_std = values.std(axis=0)
    active = list(NON_TEMPERATURE_INDICES)
    if np.all(measured_std[active] <= MINIMUM_BASELINE_STDS[active]):
        raise ValueError(
            "The resting baseline has insufficient measured variation. "
            "Check that fresh chest-strap measurements are arriving, let "
            "the readings settle, and repeat the ten-minute calibration."
        )
    return values.mean(axis=0), np.maximum(measured_std, MINIMUM_BASELINE_STDS)


def baseline_issue(stds, window_count=None):
    """Reject the collapsed legacy baseline instead of emitting false certainty."""
    values = np.asarray(stds, dtype=np.float64)
    if (
        values.shape != MINIMUM_BASELINE_STDS.shape
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        return "The saved resting baseline is invalid. Please repeat calibration."
    if window_count is not None and window_count < MINIMUM_BASELINE_WINDOWS:
        return "The saved baseline is too short. Please repeat the ten-minute calibration."
    active = list(NON_TEMPERATURE_INDICES)
    if np.all(
        values[active] <= MINIMUM_BASELINE_STDS[active] * (1.0 + 1e-6)
    ):
        return (
            "The saved resting baseline could not measure your normal variation. "
            "Please repeat the ten-minute calibration after the readings settle."
        )
    return None
