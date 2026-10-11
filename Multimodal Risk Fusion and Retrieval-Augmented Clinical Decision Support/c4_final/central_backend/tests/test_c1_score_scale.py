"""C1 raw-score contract: the display index is NOT a fusion probability."""
import math
import pytest
import modality_clients as mc


class FakeResponse:
    status_code = 200
    def __init__(self, body):
        self.body = body
    def json(self):
        return self.body


class FakeClient:
    def __init__(self, body):
        self.body = body
        self.urls = []
    def get(self, url, **kwargs):
        self.urls.append(url)
        return FakeResponse(self.body)


def test_modern_c1_output_keeps_0_to_1_scale(monkeypatch):
    monkeypatch.setattr(mc, "C1_BASE", "https://c1.example")
    client = FakeClient({
        "status": "success", "current_c1_score": 0.375,
        "score": 0.375, "current_risk_index": 37.5,
        "latest_reading_at": "2026-10-10T22:10:43+00:00",
    })
    result = mc.call_c1("synthetic-user", client=client)
    assert result.status == "ok"
    assert math.isclose(result.raw_score, 0.375)
    assert client.urls == ["https://c1.example/predict/synthetic-user"]
    assert "normalized C1 score" in result.note


def test_legacy_c1_display_index_is_divided_by_100(monkeypatch):
    monkeypatch.setattr(mc, "C1_BASE", "https://c1.example")
    result = mc.call_c1("synthetic-user", client=FakeClient({
        "status": "success", "current_risk_index": 37.5,
    }))
    assert result.status == "ok"
    assert math.isclose(result.raw_score, 0.375)
    assert "current_risk_index / 100" in result.note


@pytest.mark.parametrize("bad", [
    {"current_c1_score": 5.2, "current_risk_index": 52.0},
    {"current_c1_score": 0.3, "current_risk_index": 70.0},
    {"current_c1_score": float("nan"), "current_risk_index": 10.0},
    {"current_risk_index": 180.0},
    {"current_risk_index": -1.0},
    {"current_risk_index": float("inf")},
    {"current_c1_score": True},
    {},
])
def test_invalid_or_disagreeing_c1_scores_fail_closed(monkeypatch, bad):
    monkeypatch.setattr(mc, "C1_BASE", "https://c1.example")
    result = mc.call_c1("synthetic-user", client=FakeClient({
        "status": "success", **bad,
    }))
    assert result.status == "error"
    assert result.raw_score is None
    assert "scale validation failed" in result.note


@pytest.mark.parametrize("remote_status,expected", [
    ("not_calibrated", "warming_up"),
    ("buffering", "warming_up"),
    ("stale", "poor_signal"),
])
def test_non_ready_c1_stays_missing(monkeypatch, remote_status, expected):
    monkeypatch.setattr(mc, "C1_BASE", "https://c1.example")
    result = mc.call_c1("synthetic-user", client=FakeClient({
        "status": remote_status,
        "message": "Physiological data not yet sufficient",
        "current_risk_index": 75.0,
    }))
    assert result.status == expected
    assert result.raw_score is None
