"""Resting-baseline validation shared by calibration and inference."""

import numpy as np


FEATURE_SCHEMA_VERSION = 2
MINIMUM_BASELINE_WINDOWS = 10
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
    # A steadily changing HR during setup is not a stationary resting reference.
    # This is a calibration-quality rule, not an anxiety/medical threshold.
    first, second = np.array_split(values[:, 0], 2)
    within = np.sqrt((first.var() + second.var()) / 2)
    difference = abs(first.mean() - second.mean())
    if difference > 5.0 and difference > 2.0 * max(within, 1.0):
        raise ValueError(
            "Heart rate was still changing during setup. Let the readings "
            "settle in the same resting position, then repeat calibration."
        )
    measured_std = values.std(axis=0)
    if np.all(measured_std <= MINIMUM_BASELINE_STDS):
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
    if np.all(values <= MINIMUM_BASELINE_STDS * (1.0 + 1e-6)):
        return (
            "The saved resting baseline could not measure your normal variation. "
            "Please repeat the ten-minute calibration after the readings settle."
        )
    return None
