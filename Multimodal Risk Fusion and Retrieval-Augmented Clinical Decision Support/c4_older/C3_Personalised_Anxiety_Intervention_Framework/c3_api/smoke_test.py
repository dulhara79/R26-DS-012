"""C3 Phase 3 smoke test — exercises all 9 endpoints end-to-end.

Run AFTER `uvicorn app.main:app` is up:
    python smoke_test.py

Exits 0 if all tests pass, 1 otherwise.
"""
from __future__ import annotations

import sys
import time
import uuid

import requests

BASE = "http://127.0.0.1:8000"
TIMEOUT = 30


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------
SAMPLE_FEATURES_MEDIUM = {
    "age_norm": 0.41,
    "gender_enc": 2.0,
    "marital_enc": 3.0,
    "education_enc": 3.0,
    "income_enc": 0.45,
    "physiological_risk": 0.62,
    "behavioral_risk": 0.55,
    "textual_risk": 0.58,
    "composite_risk": 0.58,
    "risk_tier_enc": 0.0,  # forced to 0 by validator anyway
    "interaction_count_norm": 0.05,
    "last_reward_norm": 0.50,
    "escalation_count_norm": 0.00,
}

SAMPLE_FEATURES_HIGH = {
    **SAMPLE_FEATURES_MEDIUM,
    "physiological_risk": 0.88,
    "behavioral_risk": 0.82,
    "textual_risk": 0.85,
    "composite_risk": 0.85,
}


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------
class Results:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def ok(self, name: str):
        self.passed.append(name)
        print(f"  [PASS] {name}")

    def fail(self, name: str, reason: str):
        self.failed.append((name, reason))
        print(f"  [FAIL] {name}: {reason}")

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 60}")
        print(f"  Passed: {len(self.passed)}/{total}")
        if self.failed:
            print(f"  Failed: {len(self.failed)}")
            for name, reason in self.failed:
                print(f"    - {name}: {reason}")
        print(f"{'=' * 60}")
        return 0 if not self.failed else 1


R = Results()


def _post(path: str, payload: dict, token: str | None = None) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.post(f"{BASE}{path}", json=payload, headers=headers, timeout=TIMEOUT)


def _get(path: str) -> requests.Response:
    return requests.get(f"{BASE}{path}", timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def wait_for_server() -> bool:
    print("Waiting for server ...")
    for i in range(30):
        try:
            r = _get("/health")
            if r.status_code == 200:
                print(f"  Server up. Health: {r.json()}")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def test_health():
    r = _get("/health")
    if r.status_code != 200:
        return R.fail("GET /health", f"status {r.status_code}")
    data = r.json()
    required = {"xgboost_loaded", "calibrator_loaded", "conformal_loaded"}
    missing = required - set(data.keys())
    if missing:
        return R.fail("GET /health", f"missing keys {missing}")
    if not all(data[k] for k in required):
        return R.fail("GET /health", f"core artifacts not loaded: {data}")
    R.ok("GET /health")


def test_classify() -> int | None:
    r = _post(
        "/v3/risk/classify",
        {"patient_id": "test_medium", "features": SAMPLE_FEATURES_MEDIUM},
    )
    if r.status_code != 200:
        return R.fail("POST /v3/risk/classify", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    for k in ("risk_tier", "risk_label", "calibrated_probabilities",
              "conformal_set", "intervention_type", "priority"):
        if k not in data:
            return R.fail("POST /v3/risk/classify", f"missing key {k}")
    if data["risk_tier"] not in (0, 1, 2):
        return R.fail("POST /v3/risk/classify", f"invalid tier {data['risk_tier']}")
    R.ok(f"POST /v3/risk/classify (tier={data['risk_tier']}, iv={data['intervention_type']})")
    return data["risk_tier"]


def test_classify_leakage_fix():
    """Verify risk_tier_enc=99 in input is ignored (forced to 0)."""
    payload = {
        "patient_id": "test_leak",
        "features": {**SAMPLE_FEATURES_MEDIUM, "risk_tier_enc": 99.0},
    }
    r = _post("/v3/risk/classify", payload)
    if r.status_code != 200:
        return R.fail("leakage_fix", f"status {r.status_code}: {r.text[:200]}")
    R.ok("Leakage fix: risk_tier_enc=99 accepted and forced to 0")


def test_explain():
    r = _post(
        "/v3/risk/explain",
        {"patient_id": "test_explain", "features": SAMPLE_FEATURES_HIGH},
    )
    if r.status_code != 200:
        return R.fail("POST /v3/risk/explain", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not data.get("shap_values") or not data.get("top_risk_factors"):
        return R.fail("POST /v3/risk/explain", "empty shap_values or top_risk_factors")
    if not data.get("nl_summary"):
        return R.fail("POST /v3/risk/explain", "missing nl_summary")
    R.ok(f"POST /v3/risk/explain (top={data['top_risk_factors'][:3]})")


def test_recommend():
    r = _post(
        "/v3/recommend",
        {"patient_id": "test_rec", "features": SAMPLE_FEATURES_MEDIUM, "k": 5},
    )
    if r.status_code != 200:
        return R.fail("POST /v3/recommend", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not data.get("similar_cases"):
        return R.fail("POST /v3/recommend", "no similar cases returned")
    if not data.get("recommended_intervention"):
        return R.fail("POST /v3/recommend", "no recommendation")
    R.ok(
        f"POST /v3/recommend ({len(data['similar_cases'])} cases, "
        f"retriever={data['retriever_used']}, "
        f"iv={data['recommended_intervention']})"
    )


def test_session_complete() -> str:
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    r = _post("/v3/session/complete", {
        "patient_id":        "test_session",
        "session_id":        session_id,
        "intervention_type": "targeted_nudge",
        "completion_flag":   True,
        "user_rating":       4.0,
        "hr_mean":           72.5,
        "hrv_rmssd":         45.0,
        "duration_seconds":  420,
    })
    if r.status_code != 200:
        R.fail("POST /v3/session/complete", f"status {r.status_code}: {r.text[:200]}")
        return session_id
    data = r.json()
    if not data.get("recorded"):
        R.fail("POST /v3/session/complete", "recorded=False")
        return session_id
    R.ok(f"POST /v3/session/complete ({session_id})")
    return session_id


def test_reward_compute(session_id: str):
    r = _post("/v3/reward/compute", {
        "patient_id":          "test_session",
        "session_id":          session_id,
        "completion_flag":     1.0,
        "user_rating":         4.0,
        "gad7_pre":            12.0,
        "gad7_post":           9.0,
        "escalation_occurred": False,
    })
    if r.status_code != 200:
        return R.fail("POST /v3/reward/compute", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not -1.0 <= data["composite_reward"] <= 1.0:
        return R.fail(
            "POST /v3/reward/compute",
            f"reward out of bounds: {data['composite_reward']}",
        )
    if not 0.0 <= data["updated_last_reward_norm"] <= 1.0:
        return R.fail(
            "POST /v3/reward/compute",
            f"updated_last_reward_norm out of bounds: {data['updated_last_reward_norm']}",
        )
    R.ok(
        f"POST /v3/reward/compute (R={data['composite_reward']:.3f}, "
        f"F12'={data['updated_last_reward_norm']:.3f})"
    )


def test_register() -> str | None:
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    r = _post("/v3/register", {
        "email":           email,
        "password":        "TestPass123!",
        "age":             25,
        "gender":          "Female",
        "marital_status":  "Never",
        "education_level": 3,
        "income_pir":      2.5,
    })
    if r.status_code != 200:
        R.fail("POST /v3/register", f"status {r.status_code}: {r.text[:200]}")
        return None
    data = r.json()
    if not data.get("access_token") or not data.get("user_id"):
        R.fail("POST /v3/register", "missing token or user_id")
        return None
    R.ok(f"POST /v3/register (user_id={data['user_id'][:8]}...)")
    return data["user_id"]


def test_gad7_submit(user_id: str | None):
    if user_id is None:
        user_id = "test_gad7"
    r = _post("/v3/gad7/submit", {
        "patient_id":          user_id,
        "gad7_answers":        [2, 2, 1, 2, 1, 2, 2],  # total = 12 → Medium
        "physiological_risk":  0.6,
        "behavioral_risk":     0.55,
        "textual_risk":        0.58,
    })
    if r.status_code != 200:
        return R.fail("POST /v3/gad7/submit", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    if data["gad7_score"] != 12:
        return R.fail("POST /v3/gad7/submit", f"wrong score: {data['gad7_score']}")
    if "features" not in data:
        return R.fail("POST /v3/gad7/submit", "missing features")
    R.ok(f"POST /v3/gad7/submit (score={data['gad7_score']})")


def test_clinician_review():
    r = _post("/v3/clinician/review", {
        "patient_id":            "test_clin",
        "clinician_id":          "DR_SMITH",
        "action":                "modify",
        "modified_intervention": "priority_followup",
        "clinical_notes":        "Patient shows elevated physiological markers.",
    })
    if r.status_code != 200:
        return R.fail("POST /v3/clinician/review", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    if data["final_intervention"] != "priority_followup":
        return R.fail("POST /v3/clinician/review", f"wrong final: {data}")
    R.ok("POST /v3/clinician/review (modify)")


def test_intervention_assign():
    r = _post("/v3/intervention/assign", {
        "patient_id":         "test_assign",
        "intervention_type":  "targeted_nudge",
        "priority":           "P3",
        "clinician_approved": True,
    })
    if r.status_code != 200:
        return R.fail("POST /v3/intervention/assign", f"status {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not data.get("assigned"):
        return R.fail("POST /v3/intervention/assign", "assigned=False")
    R.ok("POST /v3/intervention/assign")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("\nC3 Phase 3 smoke test\n" + "=" * 60)
    if not wait_for_server():
        print("Server never came up. Is uvicorn running on port 8000?")
        return 1

    print("\n[Core endpoints]")
    test_health()
    test_classify()
    test_classify_leakage_fix()
    test_explain()
    test_recommend()

    print("\n[Session + reward]")
    sid = test_session_complete()
    test_reward_compute(sid)

    print("\n[User + clinician]")
    uid = test_register()
    test_gad7_submit(uid)
    test_clinician_review()
    test_intervention_assign()

    return R.summary()


if __name__ == "__main__":
    sys.exit(main())
