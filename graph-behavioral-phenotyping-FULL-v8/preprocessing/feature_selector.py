"""RAPIDS feature-base selection used by the final v8 representation."""
from __future__ import annotations
import pandas as pd

from config import (
    MAX_BASES,
    MAX_BASES_PER_SENSOR,
    MIN_BASE_COVERAGE,
    USE_SEGMENTS,
)


def parse_feature_column(column: str):
    parts = column.split(":")
    sensor = parts[0].replace("f_", "") if parts else "unknown"
    segment = parts[-1] if len(parts) >= 3 else "unknown"
    return sensor, segment


def strip_segment(column: str):
    return ":".join(column.split(":")[:-1])


def select_feature_bases(
    feature_csv,
    max_bases=MAX_BASES,
    max_per_sensor=MAX_BASES_PER_SENSOR,
    min_coverage=MIN_BASE_COVERAGE,
    segments=USE_SEGMENTS,
    probe_rows=4000,
):
    header = pd.read_csv(feature_csv, nrows=0).columns.tolist()
    feature_cols = [c for c in header if c.startswith("f_")]
    segment_of = {c: parse_feature_column(c)[1] for c in feature_cols}

    by_segment = {
        seg: {strip_segment(c) for c in feature_cols if segment_of[c] == seg}
        for seg in segments
    }
    common_bases = set.intersection(*by_segment.values())

    candidate_cols = [
        c for c in feature_cols
        if segment_of[c] in segments and strip_segment(c) in common_bases
    ]

    sample = pd.read_csv(feature_csv, nrows=probe_rows, usecols=candidate_cols)
    coverage = sample.notna().mean()

    base_numeric = {}
    base_coverage = {}
    for c in candidate_cols:
        base = strip_segment(c)
        numeric = pd.api.types.is_numeric_dtype(sample[c])
        base_numeric[base] = base_numeric.get(base, True) and numeric
        base_coverage.setdefault(base, []).append(float(coverage[c]))

    ranked = []
    for base, values in base_coverage.items():
        mean_cov = sum(values) / len(values)
        if mean_cov >= min_coverage and base_numeric.get(base, False):
            ranked.append((base, mean_cov))
    ranked.sort(key=lambda x: -x[1])

    selected = []
    per_sensor = {}
    for base, _ in ranked:
        sensor = base.split(":")[0].replace("f_", "")
        if per_sensor.get(sensor, 0) >= max_per_sensor:
            continue
        selected.append(base)
        per_sensor[sensor] = per_sensor.get(sensor, 0) + 1
        if len(selected) >= max_bases:
            break

    return sorted(selected), per_sensor
