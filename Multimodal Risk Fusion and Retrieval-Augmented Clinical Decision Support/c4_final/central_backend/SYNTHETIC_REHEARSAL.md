# R26-DS-012 synthetic episode rehearsal

Run this before the deployed patient/clinician acceptance session. Use synthetic
subjects and separate credentials; do not copy clinical data or tokens into
the report.

## Local API contract gate

From the repository root, install `central_backend/requirements.txt`,
`fusion_service/requirements.txt`, and `pytest`. Then run:

```sh
cd central_backend
python -m pytest -q tests/test_synthetic_episode.py
python -m pytest -q tests
python test_backend.py
python -m unittest test_rag_client_contract.py -v
```

`test_synthetic_episode.py` calls public backend routes through TestClient with
deterministic C1, C3 and C4 responses. It verifies patient self-enrolment and
JWT, a one-use patient-issued clinician invite, assignment scoping, separate
physiological forecast, shared `fusion_result_id`, one confirmed episode and
AttentionEvent, patient-safe event projection, deduplication, and atomic
OPEN → ACKNOWLEDGED → RESOLVED with server actor and persistent UTC timestamps.
It advances only the synthetic C1 source/forecast clock by 30 seconds; no
production policy threshold or wait is changed.

The `mobile-consumers` CI job checks out the exact Aura and ClinAnx commits in
[`contracts/mobile_revisions.json`](contracts/mobile_revisions.json), compares
their current route use to `/openapi.json`, and validates the clinician fixture
types and stale/unavailable/event semantics against the backend models. When
either app changes a consumed route or fixture, update its pinned SHA only
after the compatibility job passes. This job checks wire compatibility; the
apps' own Flutter CI remains responsible for widget and build behavior.

## Staging acceptance with real services and both apps

1. Record the backend and C1/C3/C4/RAG deployment revisions, model/policy
   versions, `/ready` result, schema revision, and both app commit SHAs. Verify
   the deployed backend URL uses HTTPS and required secrets are configured.
2. In Aura, enrol a fresh synthetic `P_…` identity, sign in, issue a one-use
   clinician invite and give only the code to the test clinician. In ClinAnx,
   redeem the code once. Verify a second redemption returns 409 and an
   unassigned clinician receives 403 for subject and event detail.
3. Ingest valid C1 readings from two distinct source-stamped windows 20–120
   seconds apart, C4 contextual data and a C3 clinical note. Confirm C2 is
   excluded. Check both apps display the same backend `fusion_result_id`,
   current tier and separate physiological forecast. Record the forecast ID,
   `valid_until`, policy version and modality inclusion/freshness status.
4. Confirm exactly one OPEN AttentionEvent for that episode in both apps after
   polling/restart. ACK and RESOLVE in ClinAnx with an empty request body.
   Verify actor/timestamps and final state after backend restart, and 409 on
   repeated or out-of-order transitions. A repeated C1 window must not alert.
5. Exercise missing/stale C1, C3/C4 error, expired JWT, guessed unassigned IDs,
   backend offline, network recovery, duplicate polling, and two clinician
   transitions racing on one event. Record responses and redacted screenshots.
6. On the staging PostgreSQL database, rehearse an upgrade from the last
   deployed revision, a backup/restore, and post-restore event state. Install
   both signed Android builds and complete the real-device smoke checklist.

The `postgres-contract` CI job verifies these database properties with a
disposable PostgreSQL 16 database and synthetic records. Its dump/restore
does not replace step 6 using the real staging schema and rollback image.

Use the deployed API verifier in [STAGING_VERIFICATION.md](STAGING_VERIFICATION.md)
to capture redacted readiness, access, assessment identity and event lifecycle
observations from the synthetic staging subject.

The local gate proves only the in-process API contract. Mark the integrated
release accepted only when steps 1–6 have actual deployment/device evidence,
artifact SHA-256 hashes, and a rollback image recorded in the release sheet.
