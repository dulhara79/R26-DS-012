"""Exercise the deployment verifier against a stateful synthetic wire server."""

import datetime as dt
import json

import httpx

from scripts.staging_verifier import _base_url, main, verify, VerificationError


NOW = dt.datetime.now(dt.timezone.utc).isoformat()
SUBJECT = "synthetic-subject"
EVENT = "evt_synthetic"


def server(*, denied=True, mismatch=False, unsafe_model=False):
    state = {"status": "OPEN", "writes": 0}

    def handler(request):
        route = request.url.path
        auth = request.headers.get("Authorization")
        if route == "/ready":
            return httpx.Response(200, json={"status": "ready", "database": "ready",
                "schema_revision": "0002_assignment_invites_and_forecast_link",
                "auth": {"clinician": True, "patient": True}})
        if route == "/openapi.json":
            from scripts.staging_verifier import REQUIRED_ROUTES
            paths = {}
            for method, path in REQUIRED_ROUTES:
                paths.setdefault(path, {})[method] = {}
            return httpx.Response(200, json={"paths": paths})
        if auth == "Bearer invalid-synthetic-token":
            return httpx.Response(401, json={"detail": "invalid token"})
        if auth == "Bearer unassigned-token-secret":
            if route == "/v1/me":
                return httpx.Response(200, json={"clinician_id": "DR002"})
            return httpx.Response(403 if denied else 200, json={"detail": "denied"})
        if route == "/v1/me":
            return httpx.Response(200, json={"clinician_id": "DR001"})
        if route == f"/v1/patients/{SUBJECT}/assessment/latest":
            return httpx.Response(200, json={"subject_id": SUBJECT, "fusion_result_id": 123,
                "current_assessment": {"score": .58, "tier": "Medium", "band": "AMBER"},
                "forecast": {"scope": "physiological", "valid_until": NOW,
                             "score": .84, "tier": "High", "horizon_minutes": 10},
                "assessment_status": "complete", "modalities": [],
                "model_version": "private patient note" if unsafe_model else "ragf-v0.4"})
        if route == f"/v1/patients/{SUBJECT}/risk":
            return httpx.Response(200, json={"subject_id": SUBJECT,
                "fusion_result_id": 124 if mismatch else 123, "composite": .58,
                "tier": "Medium", "band": "AMBER", "assessment_status": "complete",
                "forecast": {"scope": "physiological", "valid_until": NOW,
                             "score": .84, "tier": "High", "horizon_minutes": 10}})
        if route == "/v1/clinicians/me/dashboard":
            return httpx.Response(200, json={"patients": [{"subject_id": SUBJECT,
                "fusion_result_id": 123}]})
        event = {"id": EVENT, "subject_id": SUBJECT, "fusion_result_id": 123,
                 "forecast_result_id": "fcst_synthetic", "policy_version": "escalation-v1",
                 "status": state["status"], "created_at": NOW,
                 "acknowledged_by": state.get("acknowledged_by"),
                 "acknowledged_at": state.get("acknowledged_at"),
                 "resolved_by": state.get("resolved_by"),
                 "resolved_at": state.get("resolved_at")}
        if route == f"/v1/attention-events/{EVENT}" and request.method == "GET":
            return httpx.Response(200, json={"event": event})
        if route == "/v1/attention-events":
            return httpx.Response(200, json={"events": [event]})
        if route == "/v1/patients/me/attention-events":
            return httpx.Response(200, json={"events": [{"id": EVENT, "status": state["status"]}]})
        if route.endswith("/acknowledge"):
            if state["status"] != "OPEN":
                return httpx.Response(409, json={"detail": "changed"})
            state.update(status="ACKNOWLEDGED", acknowledged_by="DR001", acknowledged_at=NOW)
        elif route.endswith("/resolve"):
            if state["status"] != "ACKNOWLEDGED":
                return httpx.Response(409, json={"detail": "changed"})
            state.update(status="RESOLVED", resolved_by="DR001", resolved_at=NOW)
        else:
            raise AssertionError(f"unexpected route: {route}")
        state["writes"] += 1
        event.update(status=state["status"], acknowledged_by=state.get("acknowledged_by"),
                     acknowledged_at=state.get("acknowledged_at"),
                     resolved_by=state.get("resolved_by"), resolved_at=state.get("resolved_at"))
        return httpx.Response(200, json={"event": event})

    return httpx.MockTransport(handler), state


def run(transport, *, transition=False):
    with httpx.Client(base_url="https://staging.example.test", transport=transport) as client:
        return verify(client, subject_id=SUBJECT, event_id=EVENT,
                      patient_token="patient-token-secret", clinician_token="clinician-token-secret",
                      unassigned_token="unassigned-token-secret",
                      transition_event=transition)


def test_read_only_never_mutates_and_redacts_ids_and_tokens():
    transport, state = server()
    result = run(transport)
    assert result["result"] == "passed_read_only"
    assert state["writes"] == 0
    assert all(check["status"] == "passed" for check in result["checks"])
    evidence = json.dumps(result)
    for private in (SUBJECT, EVENT, "patient-token-secret", "clinician-token-secret",
                    "unassigned-token-secret"):
        assert private not in evidence


def test_explicit_transition_checks_actor_persistence_and_conflicts():
    transport, state = server()
    result = run(transport, transition=True)
    assert result["result"] == "passed_transition"
    assert state["writes"] == 2 and state["status"] == "RESOLVED"
    assert len(result["checks"]) == 10


def test_auth_denial_or_mismatched_patient_identity_blocks_writes():
    for opts, failed in (({"denied": False}, "authentication_and_assignment_denial"),
                         ({"mismatch": True}, "shared_assessment_identity"),
                         ({"unsafe_model": True}, "shared_assessment_identity")):
        transport, state = server(**opts)
        result = run(transport, transition=True)
        assert result["result"] == "failed"
        assert result["checks"][-1]["name"] == failed
        assert state["writes"] == 0
        assert "private patient note" not in json.dumps(result)


def test_https_origin_and_cli_credential_fail_closed(tmp_path, monkeypatch, capsys):
    for url in ("http://example.test", "https://user:pass@example.test",
                "https://example.test/path", "https://example.test?token=secret"):
        try:
            _base_url(url)
        except VerificationError:
            pass
        else:
            raise AssertionError(f"accepted unsafe URL: {url}")
    for name in ("PATIENT_ACCESS_TOKEN", "CLINICIAN_ACCESS_TOKEN",
                 "UNASSIGNED_CLINICIAN_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "evidence.json"
    code = main(["--base-url", "https://example.test", "--subject-id", SUBJECT,
                 "--event-id", EVENT, "--output", str(out), "--transition-event"])
    assert code == 2 and not out.exists()
    assert "token" not in capsys.readouterr().out
