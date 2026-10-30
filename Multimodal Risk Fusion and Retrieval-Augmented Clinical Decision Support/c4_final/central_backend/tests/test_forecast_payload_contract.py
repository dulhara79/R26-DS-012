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


def test_c1_percentages_below_one_percent_are_not_treated_as_probabilities():
    # C1 says "risk_forecast": "forecast * 100" even below 1 percent.
    assert abs(_ten_minute_forecast({
        "risk_forecast": [.3, .8],
        "forecast": [.003, .008],
        "forecast_horizons_minutes": [5, 10],
    }) - .008) < 1e-12


def test_c1_native_forecast_must_agree_with_percent_display():
    assert _ten_minute_forecast({
        "risk_forecast": [30, 80],
        "forecast": [.30, .08],
        "forecast_horizons_minutes": [5, 10],
    }) is None
    assert _ten_minute_forecast({
        "risk_forecast": [30, 80],
        "forecast": [.30, .80],
        "forecast_horizons_minutes": [5, 10],
    }) == .80


def test_c1_forecast_rejects_out_of_range_or_non_finite_percent():
    for invalid in ([30, 120], [30, float("nan")], [30, -1], [30, True]):
        assert _ten_minute_forecast({
            "risk_forecast": invalid,
            "forecast_horizons_minutes": [5, 10],
        }) is None
