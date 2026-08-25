"""Deterministic PP2 demo check for Component 2 change detection.

Generates 60 synthetic DAILY FEATURE rows in memory:
  - days 1-28: variable personal baseline
  - days 29-44: baseline-like behaviour
  - days 45-60: sustained upward screen-activity drift

Nothing is uploaded and no research data are modified.

Run:
    cd apps/anxiety_mobile_app/component2_backend
    python test_pp2_demo_change_detection.py
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from processor import Component2Processor, EWMA_THRESHOLD


def make_rows(start_day):
    rows = []
    for i in range(60):
        day = start_day + timedelta(days=i)
        baseline_like_screen = 120.0 + ((i % 7) - 3) * 4.0
        screen_minutes = (
            baseline_like_screen
            if i < 44
            else 330.0 + ((i % 5) - 2) * 6.0
        )
        rows.append(
            {
                "feature_date": day.isoformat(),
                "usable_day": True,
                "screen_minutes": screen_minutes,
                "distance_km": 4.0 + ((i % 5) - 2) * 0.08,
                "high_motion_fraction": 0.11 + ((i % 3) - 1) * 0.003,
                "social_media_minutes": 48.0 + ((i % 4) - 1.5) * 2.0,
                "location_coverage": 0.75,
                "screen_coverage": 0.80,
                "movement_coverage": 0.70,
            }
        )
    return rows


def main():
    processor = Component2Processor(db=None)
    today = datetime.now(processor.tz).date()
    start_day = today - timedelta(days=59)
    enrolled_at = datetime.combine(
        start_day,
        time.min,
        tzinfo=processor.tz,
    ).astimezone(timezone.utc)

    participant = {
        "participant_code": "PP2_SYNTHETIC_DEMO",
        "auth_user_id": "00000000-0000-0000-0000-000000000000",
        "enrolled_at": enrolled_at.isoformat(),
        "active": True,
    }

    output = processor.build_observation(participant, make_rows(start_day))
    change = output.get("change_detection") or {}

    print("baseline_ready:", output["baseline_ready"])
    print("reportable:", output["reportable"])
    print("change_detection:", change)

    assert output["baseline_ready"] is True
    assert output["reportable"] is True
    assert change.get("detected") is True
    assert change.get("feature") == "screen activity"
    assert abs(float(change["ewma_z"])) >= EWMA_THRESHOLD

    print("PASS: 60-day synthetic drift triggers the Day-57+ EWMA detector.")


if __name__ == "__main__":
    main()
