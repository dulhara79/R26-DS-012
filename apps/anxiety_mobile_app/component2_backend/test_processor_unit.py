"""Unit tests for the Component 2 daily processor (no network, no database).

Run from component2_backend/:
    python -m pytest test_processor_unit.py -v
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from processor import Component2Processor, EWMA_THRESHOLD

P = Component2Processor(db=None)
TZ = P.tz
DAY = date(2026, 8, 10)


def ev(hh, mm, event_type, **value):
    local = datetime.combine(DAY, time(hh, mm), tzinfo=TZ)
    return {
        "event_time": local.astimezone(timezone.utc).isoformat(),
        "event_type": event_type,
        "value_json": value,
    }


def agg(events):
    return P.aggregate_day("uid", "P_TEST", DAY, events)


# ---------------------------------------------------------------- screen
def test_screen_session_minutes_and_unlocks():
    row = agg([
        ev(10, 0, "Screen_Event", state="Screen_Unlocked"),
        ev(10, 30, "Screen_Event", state="Screen_Off"),
        ev(14, 0, "Screen_Event", state="Screen_On"),
        ev(14, 15, "Screen_Event", state="Screen_Off"),
    ])
    assert row["screen_minutes"] == pytest.approx(45.0)
    assert row["unlock_count"] == 1


def test_night_screen_counts_only_00_to_06():
    row = agg([
        ev(5, 30, "Screen_Event", state="Screen_On"),
        ev(6, 30, "Screen_Event", state="Screen_Off"),
    ])
    assert row["screen_minutes"] == pytest.approx(60.0)
    assert row["night_screen_minutes"] == pytest.approx(30.0)


def test_unterminated_session_is_capped():
    row = agg([ev(23, 50, "Screen_Event", state="Screen_Unlocked")])
    # capped at day end (10 min), never the 30-minute open-session cap here
    assert row["screen_minutes"] == pytest.approx(10.0)


# ---------------------------------------------------------------- location
def test_distance_uses_haversine_and_drops_implausible_jumps():
    row = agg([
        ev(8, 0, "Location_Grid_100m", lat=6.900, lng=79.850),
        ev(8, 15, "Location_Grid_100m", lat=6.910, lng=79.850),   # ~1.11 km
        ev(8, 30, "Location_Grid_100m", lat=7.900, lng=79.850),   # ~111 km jump: dropped
    ])
    assert row["distance_km"] == pytest.approx(1.112, abs=0.01)


def test_home_minutes_and_places():
    pts = [ev(h, 0, "Location_Grid_100m", lat=6.900, lng=79.850) for h in range(0, 6)]
    pts += [ev(12, 0, "Location_Grid_100m", lat=6.950, lng=79.860)]
    row = agg(pts)
    assert row["home_minutes"] == pytest.approx(6 * 15.0)
    assert row["significant_places"] == 2
    assert 0.0 < row["location_entropy"] < 1.0


# ---------------------------------------------------------------- app usage
def test_app_usage_window_is_scaled_to_window_length():
    row = agg([
        ev(9, 0, "App_Usage_Category_15m", window_minutes=15,
           categories_sec={"Social_Media": 900, "Entertainment": 300}),
    ])
    # 1,200 s reported inside a 900 s window -> scaled so total == 15 min
    total = row["social_media_minutes"] + row["entertainment_minutes"]
    assert total == pytest.approx(15.0, abs=0.01)
    assert row["social_media_minutes"] == pytest.approx(11.25, abs=0.01)


# ---------------------------------------------------------------- coverage
def test_usable_day_requires_two_modalities_and_density():
    dense = [ev(h, m, "Location_Grid_100m", lat=6.9, lng=79.85)
             for h in range(24) for m in (0, 15, 30)]          # 72/96 = 0.75
    dense += [ev(h, m, "Movement_Window_5m", mean_magnitude=9.8, std_magnitude=0.2,
                 high_motion_fraction=0.1, sample_count=100)
              for h in range(24) for m in (0, 30)]                 # 48/288 = 0.17
    assert agg(dense)["usable_day"] is True

    sparse = [ev(h, 0, "Service_Heartbeat") for h in range(5)]
    assert agg(sparse)["usable_day"] is False


def test_communication_is_counts_only():
    row = agg([ev(1, 0, "Call_Stats_Daily", incoming=3, outgoing=2, missed=1, rejected=0),
               ev(1, 0, "SMS_Activity_Daily", sent=4, received=6)])
    assert (row["incoming_calls"], row["outgoing_calls"], row["missed_calls"]) == (3, 2, 1)
    assert (row["sms_sent"], row["sms_received"]) == (4, 6)


# ---------------------------------------------------------------- observations
def participant(days_ago):
    start = datetime.now(TZ).date() - timedelta(days=days_ago)
    enrolled = datetime.combine(start, time.min, tzinfo=TZ).astimezone(timezone.utc)
    return start, {"participant_code": "P_TEST", "auth_user_id": "uid",
                   "enrolled_at": enrolled.isoformat(), "active": True}


def rows(start, n, screen=lambda i: 120.0 + (i % 5)):
    return [{"feature_date": (start + timedelta(days=i)).isoformat(), "usable_day": True,
             "screen_minutes": screen(i), "distance_km": 4.0 + (i % 3) * 0.1,
             "high_motion_fraction": 0.1, "social_media_minutes": 40.0 + (i % 4),
             "location_coverage": 0.8, "screen_coverage": 0.8, "movement_coverage": 0.8}
            for i in range(n)]


def test_cold_start_withholds_everything():
    start, part = participant(9)
    out = P.build_observation(part, rows(start, 10))
    assert out["baseline_ready"] is False and out["reportable"] is False
    assert out["observations"] == {}
    assert out["change_detection"] is None
    assert out["model_output"] is None
    assert out["model_status"] == "withheld_pending_validation"


def test_baseline_ready_but_too_few_recent_days_is_not_reportable():
    start, part = participant(29)
    out = P.build_observation(part, rows(start, 30))   # only 2 post-baseline days
    assert out["baseline_ready"] is True
    assert out["reportable"] is False


def test_reportable_observation_has_direction_labels():
    start, part = participant(39)
    out = P.build_observation(part, rows(start, 40))
    assert out["reportable"] is True
    item = out["observations"]["screen_activity"]
    assert item["direction"] in {"above", "below", "stable"}
    assert item["unit"] == "hours/day"


def test_stable_behaviour_does_not_trigger_change_detection():
    start, part = participant(59)
    out = P.build_observation(part, rows(start, 60))
    assert out["change_detection"] == {"detected": False}


def test_sustained_drift_triggers_change_detection():
    start, part = participant(59)
    out = P.build_observation(part, rows(start, 60, screen=lambda i: 120.0 + (i % 5) if i < 44 else 300.0))
    cd = out["change_detection"]
    assert cd["detected"] is True and cd["feature"] == "screen activity"
    assert abs(cd["ewma_z"]) >= EWMA_THRESHOLD


def test_no_model_score_is_ever_emitted():
    for days in (5, 30, 60):
        start, part = participant(days - 1)
        out = P.build_observation(part, rows(start, days))
        assert out["model_output"] is None
        assert "score" not in out and "behavioral_score" not in out
