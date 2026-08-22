from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

try:
    from .processor import (
        BASELINE_DAYS,
        MIN_BASELINE_USABLE_DAYS,
        RECENT_WINDOW_DAYS,
        Component2Processor as BaseComponent2Processor,
        _as_float,
        _as_int,
        _parse_timestamp,
    )
except ImportError:
    from processor import (
        BASELINE_DAYS,
        MIN_BASELINE_USABLE_DAYS,
        RECENT_WINDOW_DAYS,
        Component2Processor as BaseComponent2Processor,
        _as_float,
        _as_int,
        _parse_timestamp,
    )


def _row_has_observed_data(row: dict[str, Any]) -> bool:
    """Return True only when a completed daily row contains observed data.

    The daily processor deliberately creates a row for each completed calendar
    day. Therefore row existence alone is not evidence that the phone actually
    contributed data on that day.
    """
    coverage_keys = (
        "location_coverage",
        "screen_coverage",
        "movement_coverage",
    )
    if any(_as_float(row.get(key)) > 0.0 for key in coverage_keys):
        return True

    count_keys = (
        "unlock_count",
        "incoming_calls",
        "outgoing_calls",
        "missed_calls",
        "rejected_calls",
        "sms_sent",
        "sms_received",
    )
    if any(_as_int(row.get(key)) > 0 for key in count_keys):
        return True

    duration_keys = (
        "screen_minutes",
        "night_screen_minutes",
        "distance_km",
        "social_media_minutes",
        "entertainment_minutes",
        "education_minutes",
    )
    return any(_as_float(row.get(key)) > 0.0 for key in duration_keys)


class Component2Processor(BaseComponent2Processor):
    """Component 2 processor with explicit completed-day baseline accounting."""

    def build_observation(
        self,
        participant: dict[str, Any],
        feature_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        observation = super().build_observation(participant, feature_rows)

        enrolled_date = _parse_timestamp(participant["enrolled_at"]).astimezone(self.tz).date()
        today = datetime.now(self.tz).date()
        completed_through = today - timedelta(days=1)
        baseline_end = enrolled_date + timedelta(days=BASELINE_DAYS - 1)

        rows = sorted(feature_rows, key=lambda row: str(row.get("feature_date") or ""))
        baseline_rows = [
            row
            for row in rows
            if enrolled_date
            <= date.fromisoformat(str(row["feature_date"]))
            <= baseline_end
        ]
        baseline_usable_rows = [
            row for row in baseline_rows if row.get("usable_day") is True
        ]
        baseline_days_with_features = sum(
            1 for row in baseline_rows if _row_has_observed_data(row)
        )

        if completed_through < enrolled_date:
            baseline_calendar_days_elapsed = 0
        else:
            last_completed_baseline_day = min(completed_through, baseline_end)
            baseline_calendar_days_elapsed = min(
                BASELINE_DAYS,
                (last_completed_baseline_day - enrolled_date).days + 1,
            )

        post_baseline_usable = [
            row
            for row in rows
            if date.fromisoformat(str(row["feature_date"])) > baseline_end
            and row.get("usable_day") is True
        ]
        recent_usable = post_baseline_usable[-RECENT_WINDOW_DAYS:]

        # The 28 baseline calendar days must be completed before reporting can
        # start. The current partial day is never counted toward readiness.
        baseline_ready = (
            baseline_calendar_days_elapsed >= BASELINE_DAYS
            and len(baseline_usable_rows) >= MIN_BASELINE_USABLE_DAYS
        )
        reportable = baseline_ready and len(recent_usable) >= 3

        observation["baseline_ready"] = baseline_ready
        observation["reportable"] = reportable
        if not reportable:
            observation["observations"] = {}

        quality = observation.setdefault("data_quality", {})
        quality["baseline_calendar_days_elapsed"] = baseline_calendar_days_elapsed
        quality["baseline_days_with_features"] = baseline_days_with_features
        # Preserve the existing mobile field name, but make it mean what the UI
        # needs: completed baseline days that actually contain observed data.
        quality["baseline_days_available"] = baseline_days_with_features
        quality["baseline_usable_days"] = len(baseline_usable_rows)
        quality["baseline_days_required"] = BASELINE_DAYS
        quality["baseline_min_usable_days"] = MIN_BASELINE_USABLE_DAYS
        quality["recent_usable_days"] = len(recent_usable)

        return observation
