"""Focused regression checks for the temporary investor demo simulation."""

from __future__ import annotations

import os
import tempfile

_tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdb.name}"
os.environ["MRN_PEPPER"] = "investor-demo-test-pepper"
os.environ["FUSION_MODE"] = "inprocess"
os.environ["FUSION_SERVICE_DIR"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fusion_service")
)
os.environ["INVESTOR_DEMO_SIMULATION"] = "1"

import modality_clients as mc  # noqa: E402

_calls = {"c3": 0}


def stub_c1(subject_id, window=None, client=None):
    return mc.ComponentResult(
        raw_score=89.67,
        status="ok",
        confidence=0.74,
        coverage=0.81,
        model_version="c1-demo",
    )


def stub_c3(*args, **kwargs):
    _calls["c3"] += 1
    raise AssertionError("external C3 must not be called in investor demo mode")


def stub_c4(subject_id, demographics, client=None):
    return mc.ComponentResult(
        raw_score=0.55,
        status="ok",
        confidence=0.61,
        coverage=1.0,
        model_version="c4-demo",
    )


# Install deterministic external-service doubles BEFORE importing main_demo.
# main_demo captures the original C3 function so disabling the flag can prove
# that the normal C3 path is restored.
mc.call_c1 = stub_c1
mc.call_c3 = stub_c3
mc.call_c4 = stub_c4

import main as core  # noqa: E402
import main_demo as demo  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

core.mc.call_c1 = stub_c1
core.mc.call_c4 = stub_c4
client = TestClient(demo.app)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}")


# Create one Aura-style subject.
participant_id = "P_1234567890ABCDEF"
r = client.post("/v1/subjects/self", json={"app_user_id": participant_id})
check("self enrol", r.status_code == 200, r.text)
subject_id = r.json()["subject_id"]

# Physiological ingest must immediately create a fusion row equal to C1 / 100.
r = client.post("/v1/ingest/physiological", json={
    "app_user_id": participant_id,
    "device_user_id": participant_id,
})
check("physio ingest", r.status_code == 200, r.text)
payload = r.json()
check("physio raw stays 0-100", payload["score"] == 89.67, str(payload))
check("physio fusion triggers immediately", payload.get("fusion_triggered") is True, str(payload))
check("physio fusion exact", payload.get("fusion", {}).get("composite") == 0.8967, str(payload))

pat = client.get(f"/v1/patients/{subject_id}/risk").json()
doc = client.get(f"/v1/doctor/patients/{subject_id}/timeline").json()
check("patient and doctor share composite", pat["composite"] == doc["composite"] == 0.8967, f"pat={pat} doc={doc}")
check("high demo band", pat["band"] == doc["band"] == "RED", f"pat={pat} doc={doc}")

# Clinical note demo must bypass C3 and emit deterministic low/high values.
low = client.post("/v1/clinical-notes", json={
    "subject_id": subject_id,
    "note_text": "Patient has low anxiety, is stable, and remains on the lower end.",
}).json()
check("low note score", low["score"] == 0.2033, str(low))
check("low note detail", low.get("component_detail", {}).get("calibrated_probability") == 0.2033, str(low))

high = client.post("/v1/clinical-notes", json={
    "subject_id": subject_id,
    "note_text": "Severe high anxiety with escalating panic symptoms.",
}).json()
check("high note score", high["score"] == 0.8967, str(high))
check("high note detail", high.get("component_detail", {}).get("calibrated_probability") == 0.8967, str(high))
check("external C3 bypassed", _calls["c3"] == 0, str(_calls))

neutral = client.post("/v1/clinical-notes", json={
    "subject_id": subject_id,
    "note_text": "Patient attended follow-up and discussed daily routine.",
}).json()
check("neutral note score", neutral["score"] == 0.4967, str(neutral))

# Every note-triggered fusion still mirrors physiology rather than allowing the
# simulated clinical score to change the investor-demo final score.
pat_after_notes = client.get(f"/v1/patients/{subject_id}/risk").json()
doc_after_notes = client.get(f"/v1/doctor/patients/{subject_id}/timeline").json()
check(
    "clinical note does not change demo fusion source",
    pat_after_notes["composite"] == doc_after_notes["composite"] == 0.8967,
    f"pat={pat_after_notes} doc={doc_after_notes}",
)

# Turning the flag off must restore the normal code path.
os.environ["INVESTOR_DEMO_SIMULATION"] = "0"
try:
    client.post("/v1/clinical-notes", json={
        "subject_id": subject_id,
        "note_text": "High anxiety should now use the real C3 path.",
        "support_set": [{
            "id": "x",
            "text": "example",
            "label": "anxiety",
            "note_date": "2026-01-01",
        }],
    })
except AssertionError:
    pass
check("flag off restores C3 call", _calls["c3"] == 1, str(_calls))

print("Investor demo regression checks passed.")
