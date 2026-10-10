"""Verify a deployed P0 contract with synthetic patient and clinician sessions.

The default run is read-only. --transition-event deliberately ACKs and RESOLVEs
one OPEN synthetic event; never use it with a real participant's event.
The evidence contains checks and IDs hashed with a per-run random salt, not
tokens, response bodies, clinical text, or raw subject/event identifiers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
from urllib.parse import quote, urlsplit

import httpx


REQUIRED_ROUTES = {
    ("get", "/ready"), ("get", "/v1/me"),
    ("get", "/v1/clinicians/me/dashboard"),
    ("get", "/v1/patients/{subject_id}/assessment/latest"),
    ("get", "/v1/patients/{subject_id}/risk"),
    ("get", "/v1/attention-events"),
    ("get", "/v1/attention-events/{event_id}"),
    ("post", "/v1/attention-events/{event_id}/acknowledge"),
    ("post", "/v1/attention-events/{event_id}/resolve"),
    ("get", "/v1/patients/me/attention-events"),
}


class VerificationError(Exception):
    pass


def _require(value: bool, reason: str) -> None:
    if not value:
        raise VerificationError(reason)


def _json(response: httpx.Response, expected: int = 200) -> dict:
    _require(response.status_code == expected, f"expected HTTP {expected}, got {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise VerificationError("response is not JSON") from exc
    _require(isinstance(body, dict), "response is not a JSON object")
    return body


def _base_url(value: str) -> str:
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError as exc:
        raise VerificationError("invalid HTTPS origin") from exc
    _require(url.scheme == "https" and bool(url.hostname) and not url.username
             and not url.password and not url.path.strip("/") and not url.query
             and not url.fragment and (port is None or port > 0),
             "base URL must be an HTTPS origin without credentials, path or query")
    return value.rstrip("/")


def _timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _version(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) is not None


def verify(client: httpx.Client, *, subject_id: str, event_id: str,
           patient_token: str, clinician_token: str, unassigned_token: str,
           transition_event: bool = False) -> dict:
    """Return redacted evidence; a failed check stops before further writes."""
    _require(all((subject_id, event_id, patient_token, clinician_token, unassigned_token)),
             "synthetic identifiers and three tokens are required")
    _require(all("/" not in x and "?" not in x and "#" not in x for x in (subject_id, event_id)),
             "invalid synthetic identifier")
    salt = secrets.token_bytes(16)
    digest = lambda value: hashlib.sha256(salt + value.encode()).hexdigest()[:16]
    evidence = {
        "schema": "r26-staging-verification-v1",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "transition" if transition_event else "read_only",
        "subject_ref": digest(subject_id), "event_ref": digest(event_id),
        "checks": [], "result": "failed",
    }
    patient = {"Authorization": f"Bearer {patient_token}"}
    clinician = {"Authorization": f"Bearer {clinician_token}"}
    unassigned = {"Authorization": f"Bearer {unassigned_token}"}
    sid, eid = quote(subject_id, safe=""), quote(event_id, safe="")

    def check(name, action):
        try:
            result = action()
            evidence["checks"].append({"name": name, "status": "passed"})
            return result
        except (VerificationError, httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError) as exc:
            # No exception text: httpx exceptions can contain request URLs and
            # future response validators might include clinical data.
            evidence["checks"].append({"name": name, "status": "failed",
                                       "reason": (str(exc) if isinstance(exc, VerificationError)
                                                  else type(exc).__name__)})
            return None

    def run(name, action):
        value = check(name, action)
        if evidence["checks"][-1]["status"] == "failed":
            return False, None
        return True, value

    def readiness():
        body = _json(client.get("/ready"))
        _require(body.get("status") == "ready" and body.get("database") == "ready",
                 "backend/database is not ready")
        _require(body.get("schema_revision") == "0002_assignment_invites_and_forecast_link",
                 "unexpected schema revision")
        _require(body.get("auth") == {"clinician": True, "patient": True},
                 "auth configuration is not ready")
        evidence["schema_revision"] = body["schema_revision"]

    ok, _ = run("readiness", readiness)
    if not ok: return evidence

    def openapi():
        paths = _json(client.get("/openapi.json")).get("paths", {})
        _require(isinstance(paths, dict), "OpenAPI paths missing")
        missing = sorted(f"{method.upper()} {route}" for method, route in REQUIRED_ROUTES
                         if method not in paths.get(route, {}))
        _require(not missing, "frozen OpenAPI routes missing: " + ", ".join(missing))

    ok, _ = run("openapi_contract", openapi)
    if not ok: return evidence

    def identity():
        principal = _json(client.get("/v1/me", headers=clinician))
        other = _json(client.get("/v1/me", headers=unassigned))
        actor = principal.get("clinician_id")
        _require(bool(actor) and bool(other.get("clinician_id"))
                 and actor != other["clinician_id"], "clinician principals are not distinct")
        return actor

    ok, actor = run("clinician_identity", identity)
    if not ok: return evidence

    def access():
        _json(client.get("/v1/me", headers={"Authorization": "Bearer invalid-synthetic-token"}), 401)
        _json(client.get(f"/v1/patients/{sid}/assessment/latest", headers=unassigned), 403)
        _json(client.get(f"/v1/attention-events/{eid}", headers=unassigned), 403)
        _json(client.get("/v1/attention-events", params={"subject_id": subject_id}, headers=unassigned), 403)

    ok, _ = run("authentication_and_assignment_denial", access)
    if not ok: return evidence

    def assessment():
        a = _json(client.get(f"/v1/patients/{sid}/assessment/latest", headers=clinician))
        p = _json(client.get(f"/v1/patients/{sid}/risk", headers=patient))
        d = _json(client.get("/v1/clinicians/me/dashboard", headers=clinician))
        _require(a.get("subject_id") == p.get("subject_id") == subject_id,
                 "assessment subject differs")
        fid = a.get("fusion_result_id")
        _require(isinstance(fid, int) and fid > 0 and fid == p.get("fusion_result_id"),
                 "patient and clinician fusion identities differ")
        summaries = [row for row in d.get("patients", []) if row.get("subject_id") == subject_id]
        _require(len(summaries) == 1 and summaries[0].get("fusion_result_id") == fid,
                 "dashboard fusion identity differs")
        current = a.get("current_assessment")
        _require(isinstance(current, dict) and current.get("score") == p.get("composite")
                 and current.get("tier") == p.get("tier")
                 and current.get("band") == p.get("band"), "current assessment differs")
        _require(a.get("assessment_status") == p.get("assessment_status")
                 and a.get("assessment_status") in {"complete", "partial", "unavailable"},
                 "assessment status differs")
        forecast = a.get("forecast")
        _require(forecast is None or (forecast.get("scope") == "physiological"
                 and _timestamp(forecast.get("valid_until"))),
                 "forecast scope/validity is invalid")
        patient_forecast = p.get("forecast")
        _require((forecast is None) == (patient_forecast is None),
                 "patient and clinician forecast availability differs")
        if forecast is not None:
            _require(all(forecast.get(key) == patient_forecast.get(key)
                         for key in ("scope", "horizon_minutes", "score", "tier", "valid_until")),
                     "patient and clinician forecast differs")
        _require(isinstance(a.get("modalities"), list), "modality provenance missing")
        _require(_version(a.get("model_version")), "model version missing or invalid")
        evidence["model_version"] = a.get("model_version")
        return fid

    ok, fusion_id = run("shared_assessment_identity", assessment)
    if not ok: return evidence

    def event_reads():
        event = _json(client.get(f"/v1/attention-events/{eid}", headers=clinician)).get("event", {})
        listed = _json(client.get("/v1/attention-events", params={"subject_id": subject_id},
                                  headers=clinician)).get("events", [])
        projected = _json(client.get("/v1/patients/me/attention-events", headers=patient)).get("events", [])
        _require(event.get("id") == event_id and event.get("subject_id") == subject_id
                 and event.get("fusion_result_id") == fusion_id
                 and bool(event.get("forecast_result_id")) and bool(event.get("policy_version")),
                 "event source links missing or inconsistent")
        _require(_timestamp(event.get("created_at")), "event creation timestamp missing")
        _require(_version(event.get("policy_version")), "policy version missing or invalid")
        _require(sum(row.get("id") == event_id for row in listed) == 1,
                 "event list is missing or duplicated")
        matches = [row for row in projected if row.get("id") == event_id]
        _require(len(matches) == 1 and matches[0].get("status") == event.get("status")
                 and "subject_id" not in matches[0] and "reason" not in matches[0],
                 "patient event projection is missing or exposes clinician fields")
        evidence["policy_version"] = event["policy_version"]
        return event

    ok, event = run("persistent_event_projection", event_reads)
    if not ok or not transition_event:
        if ok: evidence["result"] = "passed_read_only"
        return evidence

    def mutate(path, expected_status, timestamp_field, actor_field):
        changed = _json(client.post(f"/v1/attention-events/{eid}/{path}", headers=clinician, json={})).get("event", {})
        _require(changed.get("status") == expected_status
                 and changed.get(actor_field) == actor and _timestamp(changed.get(timestamp_field)),
                 "server transition actor, timestamp or status is invalid")
        # A fresh HTTP request must retrieve the durable canonical state.
        persisted = _json(client.get(f"/v1/attention-events/{eid}", headers=clinician)).get("event", {})
        _require(persisted == changed, "event read does not match transition response")
        return changed

    def acknowledge():
        _require(event.get("status") == "OPEN", "transition requires an OPEN synthetic event")
        return mutate("acknowledge", "ACKNOWLEDGED", "acknowledged_at", "acknowledged_by")

    ok, _ = run("acknowledge_and_read_back", acknowledge)
    if not ok: return evidence
    ok, _ = run("duplicate_ack_conflict", lambda: _json(
        client.post(f"/v1/attention-events/{eid}/acknowledge", headers=clinician, json={}), 409))
    if not ok: return evidence
    ok, _ = run("resolve_and_read_back", lambda: mutate(
        "resolve", "RESOLVED", "resolved_at", "resolved_by"))
    if not ok: return evidence
    ok, _ = run("duplicate_resolve_conflict", lambda: _json(
        client.post(f"/v1/attention-events/{eid}/resolve", headers=clinician, json={}), 409))
    if ok: evidence["result"] = "passed_transition"
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="HTTPS backend origin")
    parser.add_argument("--subject-id", required=True, help="synthetic subject")
    parser.add_argument("--event-id", required=True, help="synthetic event")
    parser.add_argument("--output", required=True, type=Path, help="new redacted JSON evidence file")
    parser.add_argument("--transition-event", action="store_true",
                        help="ACK and RESOLVE the OPEN synthetic event")
    args = parser.parse_args(argv)
    try:
        url = _base_url(args.base_url)
        tokens = [os.environ.get(name, "") for name in (
            "PATIENT_ACCESS_TOKEN", "CLINICIAN_ACCESS_TOKEN", "UNASSIGNED_CLINICIAN_ACCESS_TOKEN")]
        _require(all(tokens), "three synthetic bearer tokens must be supplied as environment variables")
        _require(not args.output.exists(), "output already exists")
        with httpx.Client(base_url=url, timeout=15, follow_redirects=False) as client:
            evidence = verify(client, subject_id=args.subject_id, event_id=args.event_id,
                              patient_token=tokens[0], clinician_token=tokens[1],
                              unassigned_token=tokens[2], transition_event=args.transition_event)
        evidence["backend_host"] = urlsplit(url).hostname
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(evidence, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"{evidence['result']}: {args.output}")
        return 0 if evidence["result"].startswith("passed_") else 1
    except (VerificationError, OSError) as exc:
        # Never echo arguments, tokens, URLs, or raw response bodies.
        print(str(exc) if isinstance(exc, VerificationError) else type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
