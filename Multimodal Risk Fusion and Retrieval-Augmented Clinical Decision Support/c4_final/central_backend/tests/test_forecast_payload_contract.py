"""Handbook section 8: the two independently documented C1 forecast shapes."""

from forecast import _ten_minute_forecast


def test_explicit_five_ten_minute_forecast():
    assert _ten_minute_forecast({
        "risk_forecast": [52, 84],
        "forecast_horizons_minutes": [5, 10],
    }) == .84


def test_legacy_ten_step_forecast():
    assert _ten_minute_forecast({"risk_forecast": [.5] * 9 + [.81]}) == .81


def test_fails_closed_for_missing_or_mismatched_horizon():
    assert _ten_minute_forecast({"risk_forecast": [.52, .84]}) is None
    assert _ten_minute_forecast({
        "risk_forecast": [.52, .84], "forecast_horizons_minutes": [10, 5],
    }) is None
    assert _ten_minute_forecast({
        "risk_forecast": [.52, .84], "forecast_horizons_minutes": [5],
    }) is None


def test_fails_closed_for_invalid_steps_even_with_valid_endpoint():
    assert _ten_minute_forecast({"risk_forecast": [.5] * 3 + [None] + [.5] * 5 + [.85]}) is None
    assert _ten_minute_forecast({"risk_forecast": [.5] * 9 + [True]}) is None
    assert _ten_minute_forecast({"risk_forecast": [.5] * 9 + [float("nan")]}) is None
