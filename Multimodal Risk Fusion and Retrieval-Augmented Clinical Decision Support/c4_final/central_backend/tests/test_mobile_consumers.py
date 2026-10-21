"""Pin Aura and ClinAnx consumer routes/fixtures to the backend API contract.

Set AURA_CHECKOUT and CLINANX_CHECKOUT to the exact checked-out revisions in
contracts/mobile_revisions.json. No live service or mobile emulator is used.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clinician_api import AssessmentWire, EventWire
from main import app


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = json.loads((ROOT / "contracts/mobile_revisions.json").read_text())
SCHEMA = TestClient(app).get("/openapi.json").json()


def _checkout(kind, variable):
    raw = os.getenv(variable)
    if not raw:
        pytest.skip(f"{variable} is provided by the cross-repo CI job")
    checkout = Path(raw)
    assert checkout.is_dir()
    sha = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                  text=True).strip()
    assert sha == REVISIONS[kind]["sha"], f"{kind} checkout drifted from the pinned manifest"
    return checkout


def _path(method, path):
    assert path in SCHEMA["paths"], path
    assert method in SCHEMA["paths"][path], f"{method.upper()} {path} missing"


def test_patient_client_uses_frozen_paths_and_server_projection():
    aura = _checkout("patient", "AURA_CHECKOUT")
    source = (aura / "lib/services/api_service.dart").read_text()
    routes = (
        ("post", "/v1/subjects/self", "/v1/subjects/self"),
        ("post", "/v1/patients/me/assignment-invites", "/v1/patients/me/assignment-invites"),
        ("post", "/v1/ingest/contextual", "/v1/ingest/contextual"),
        ("post", "/v1/ingest/physiological", "/v1/ingest/physiological"),
        ("get", "/v1/patients/{subject_id}/risk",
         "/v1/patients/${Uri.encodeComponent(subjectId)}/risk"),
        ("get", "/v1/patients/me/attention-events", "/v1/patients/me/attention-events"),
    )
    for method, backend_path, client_path in routes:
        _path(method, backend_path)
        assert client_path in source, f"Aura no longer calls {backend_path}"
    assert "/v1/subjects/attach" not in source


def test_clinanx_client_paths_and_lifecycle_fixtures():
    clinanx = _checkout("clinician", "CLINANX_CHECKOUT")
    repositories = "\n".join(path.read_text() for path in
                             (clinanx / "lib/data/repositories").glob("*.dart"))
    routes = (
        ("get", "/v1/clinicians/me/patients", "/v1/clinicians/me/patients"),
        ("get", "/v1/clinicians/me/dashboard", "/v1/clinicians/me/dashboard"),
        ("get", "/v1/patients/{subject_id}/assessment/latest", "/assessment/latest"),
        ("get", "/v1/patients/{subject_id}/assessments", "/assessments"),
        ("get", "/v1/attention-events", "/v1/attention-events"),
        ("post", "/v1/clinicians/me/assignments", "/v1/clinicians/me/assignments"),
        ("post", "/v1/attention-events/{event_id}/acknowledge", "/acknowledge"),
        ("post", "/v1/attention-events/{event_id}/resolve", "/resolve"),
    )
    for method, backend_path, client_path in routes:
        _path(method, backend_path)
        assert client_path in repositories, f"ClinAnx no longer calls {backend_path}"

    fixtures = clinanx / "test/fixtures/contracts"
    for name in ("assessment_complete", "assessment_partial_stale_c1",
                 "assessment_unavailable_c3"):
        sample = json.loads((fixtures / f"{name}.json").read_text())
        # The client intentionally accepts missing optional modality fields;
        # the backend response always supplies them. Normalize those omissions
        # before checking the wire types against the backend model.
        for modality in sample["modalities"]:
            for key in ("confidence", "coverage", "captured_at", "contribution"):
                modality.setdefault(key, None)
        parsed = AssessmentWire.model_validate(sample)
        assert parsed.subject_id and parsed.fusion_result_id is not None
        assert all(not modality.included_in_fusion for modality in parsed.modalities
                   if modality.component_id == "c2_behavioral")
        assert all(not modality.available for modality in parsed.modalities
                   if modality.status == "stale")
        if parsed.assessment_status == "unavailable":
            assert parsed.current_assessment is None or parsed.current_assessment.score is None
        if parsed.forecast:
            assert parsed.forecast.scope == "physiological"

    assessment = json.loads((fixtures / "assessment_complete.json").read_text())
    statuses = ("OPEN", "ACKNOWLEDGED", "RESOLVED")
    for name, status in zip(("attention_event_open", "attention_event_acknowledged",
                             "attention_event_resolved"), statuses):
        sample = json.loads((fixtures / f"{name}.json").read_text())
        sample.setdefault("resolution_note", None)
        event = EventWire.model_validate(sample)
        assert event.status == status
        assert event.fusion_result_id == assessment["fusion_result_id"]
        assert event.subject_id == assessment["subject_id"]
        if status != "OPEN":
            assert event.acknowledged_by and event.acknowledged_at
        if status == "RESOLVED":
            assert event.resolved_by and event.resolved_at
