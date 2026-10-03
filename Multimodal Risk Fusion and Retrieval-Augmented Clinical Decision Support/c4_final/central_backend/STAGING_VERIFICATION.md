# R26-DS-012 staging verification

## Purpose

This runbook is the operator procedure for collecting deployment evidence. The
verifier checks the deployed API contract; its automated tests use synthetic
responses and do not establish that the real components or phones work.

The Integration Handbook requires a backend-authoritative flow: one subject identity, one FusionResult, separate forecast, server-owned AttentionEvent lifecycle, and consistent patient/clinician views.

## Evidence required

Record:

- backend revision
- C1/C3/C4/RAG revisions
- model and escalation policy versions
- `/ready` response
- schema revision
- patient app SHA
- ClinAnx SHA

## Synthetic staging flow

1. Create a fresh synthetic participant.
2. Register patient session.
3. Create clinician assignment invite.
4. Redeem invite once.
5. Verify second redemption fails.
6. Produce C1/C3/C4 inputs.
7. Confirm C2 remains excluded.
8. Confirm patient and clinician receive the same `fusion_result_id`.
9. Confirm one AttentionEvent episode.
10. ACK and RESOLVE from ClinAnx.
11. Restart services and verify persistence.

## Failure evidence

Capture:

- stale C1
- unavailable C3/C4
- invalid JWT
- unassigned access
- network recovery
- duplicate event attempts
- concurrent event transitions

Do not store patient identifiers or tokens in evidence files.

## Run the deployed API verifier

Prepare a **synthetic** subject with a current fusion result and an OPEN
AttentionEvent linked to that result. Obtain three short-lived bearer tokens:
its patient, an assigned clinician, and a different unassigned clinician. Put
them in `PATIENT_ACCESS_TOKEN`, `CLINICIAN_ACCESS_TOKEN`, and
`UNASSIGNED_CLINICIAN_ACCESS_TOKEN` in the operator's private environment.
Never put tokens on the command line or in a repository file.

From `central_backend`, run the read-only check:

```sh
python scripts/staging_verifier.py \
  --base-url https://YOUR-STAGING-BACKEND \
  --subject-id YOUR-SYNTHETIC-SUBJECT \
  --event-id YOUR-SYNTHETIC-OPEN-EVENT \
  --output /private/evidence/staging-read-only.json
```

It checks `/ready`, the frozen OpenAPI routes, clinician identity, invalid
token 401, unassigned subject/event 403, patient/clinician/dashboard fusion
identity and current tier, and unique server event projections. It requires
HTTPS and does not follow redirects. The report records check outcomes, schema,
model/policy versions, backend host, and randomized hashes of synthetic IDs.
It contains no bearer token, raw subject/event ID, note, or response body.
Use a fresh output path for every run. A `passed_read_only` result proves these
API observations only, at the recorded time.

To exercise the server lifecycle, use a **new synthetic OPEN event** and output
path with the explicit `--transition-event` flag. This ACKs and RESOLVEs that
event, verifies server actor/timestamps and canonical read-back, and checks
duplicate writes return 409. It fails before mutation when readiness,
authorization, assessment identity, or event links differ. If ACK succeeds but
a later step fails, inspect the canonical event before retrying. The verifier
does not roll back state. Do not run this option for a participant event.

The endpoint check does not replace the app screenshots, actual C1/C3/C4
requests and stored modality readings, RAG abstention/timeout tests, backend
restart, staging PostgreSQL upgrade/backup/restore, signed APK hashes, or
physical-device/failure-matrix evidence. Record those separately with the exact
backend/component/mobile revisions and the previous image used for rollback.

## Acceptance boundary

CI proves the code contract. This run proves the deployed environment. Release acceptance requires both.
