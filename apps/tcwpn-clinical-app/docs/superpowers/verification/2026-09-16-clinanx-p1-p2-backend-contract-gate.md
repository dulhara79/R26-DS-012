# ClinAnx P1/P2 Backend Contract Verification Gate

**Date:** 2026-09-16
**Client repository:** `dulhara79/tcwpn_mobile_app`
**Implementation branch:** `integration/clinanx-p1-p2-server-dashboard-impl`
**Backend repository inspected:** `UVINDUSEN/component4final`

## Verdict

**LIVE WIRING BLOCKED. FIXTURE/CLIENT-FOUNDATION WORK MAY CONTINUE.**

The backend state currently visible in `UVINDUSEN/component4final` does not yet implement the clinician-principal, assignment-scoped dashboard, persisted forecast, or persistent AttentionEvent contracts required by the approved ClinAnx P1/P2 design.

Do not add guessed target routes to Flutter. Continue only work that is independent of those missing server contracts (typed failure semantics, P0 contract cleanup, repository/state abstractions, fixture-backed dashboard rendering, identity cleanup, and regression coverage).

## Backend refs inspected

- `main` -> `a1ceef8daba268dac24d81aedd76c89e7c9ccc6a` (2026-08-31)
- `INTEGRATION` -> `3f7dd86979c3151c9a9966a0ff1f2cf46a7a47e0` (2026-08-30)
- `fix/c3-doctor-response` -> `4b97d41b5e8ef9bc3be3ab4a7eff6a95be35b282` (2026-08-31)
- `demo/investor-risk-simulation` -> `1e0327acf8de689e2b53f97f84d4a898097968b9`

The only open PR is the opt-in investor demo simulation. It is not an auth/assignment/attention-event implementation.

## What exists today

### Authentication

`central_backend/main.py` uses a single shared `BACKEND_API_TOKEN` and `_auth()` checks only exact equality with `Authorization: Bearer <shared token>`.

There is no authenticated clinician principal derived from a JWT/session, no `/v1/me` clinician identity route, and no role/assignment context attached to the request.

### Clinician egress

The backend exposes clinician-facing routes such as:

- `GET /v1/doctor/patients/{subject_id}/timeline`
- `POST /v1/doctor/patients/{subject_id}/evidence`
- `GET /v1/doctor/patients/{subject_id}/explanation`

These routes call `_auth()` and `_require_subject()`, but no clinician-to-subject assignment check is present.

### Current fusion

`FusionResult` is persisted and the doctor timeline returns the current fusion row plus modality status/history. It includes `fusion_result_id`, `assessment_status`, `missing_modalities`, weights/contributions, freshness and current composite/tier/band.

### Patient egress

`GET /v1/patients/{subject_id}/risk` is present and reads the same latest `FusionResult`, but it is not assignment-aware and currently has no bearer-auth dependency in the route signature.

### C1 forecast transport

C1 returns `risk_forecast` in its component response, but the central database has no separate persisted `ForecastResult` model. A C1 forecast appearing inside component detail is not equivalent to the target server-persisted forecast contract required by ClinAnx.

## Required target semantics not found

The current repository does not contain evidence of the following target contracts:

- clinician JWT/principal verification;
- `GET /v1/me` or equivalent current-clinician route;
- clinician-to-subject assignment persistence;
- assignment enforcement on patient reads;
- assignment-scoped dashboard/patient roster;
- `GET /v1/clinicians/me/dashboard` or verified equivalent;
- `GET /v1/clinicians/me/patients` or verified equivalent;
- canonical `GET /v1/patients/{subject_id}/assessment/latest` aggregate;
- persisted `ForecastResult` with explicit scope and identity;
- persistent `AttentionEvent` storage;
- assignment-scoped attention-event list/detail;
- OPEN -> ACKNOWLEDGED -> RESOLVED server lifecycle;
- actor/timestamp persistence for event transitions;
- 403 permission semantics separate from 401 identity failure;
- event concurrency/conflict reconciliation.

`central_backend/db_models.py` currently defines `Subject`, `SubjectAlias`, `PairingCode`, `ModalityReading`, `FusionResult`, `Verdict`, `AuditLog`, and support-bank storage. No clinician-assignment, forecast-result, or attention-event table/model is present.

## Safe continuation ruling

Client implementation may continue with the following, because none of these requires inventing a live server route:

1. split 401 / 403 / 409 client failure semantics;
2. separate `confidence` from `uncertainty` in P0 contracts;
3. add repository interfaces and fixture/in-memory adapters;
4. add typed async/dashboard state;
5. build the clinician worklist UI against frozen P0 fixtures;
6. remove the deprecated `attach()` compatibility shim in favor of `resolveAppUserId()`;
7. add regression tests proving no local fusion/risk/event invention occurs.

Live `CentralBackendGateway` methods for assignment-scoped patients/events MUST remain unimplemented until the backend owner supplies the actual route/OpenAPI contract.

## Message for Uvindu

Hi Uvindu,

I am implementing the ClinAnx P1/P2 server-backed dashboard against the agreed integration architecture. I checked the current `component4final` branches (`main`, `INTEGRATION`, `fix/c3-doctor-response`, and the demo branch), and the clinician-principal / assignment / persistent AttentionEvent target contracts are not present yet.

Before I wire live ClinAnx endpoints, could you please provide the updated Central Backend implementation/OpenAPI for these semantics:

1. Clinician bearer/JWT verification with a server-derived clinician identity (`/v1/me` or equivalent).
2. `401` for invalid/expired identity and `403` for an authenticated clinician who is not assigned/authorized.
3. A persisted clinician-to-subject assignment model and server-side assignment enforcement on clinician patient reads.
4. An assignment-scoped dashboard/patient-roster operation. The target docs suggest `/v1/clinicians/me/dashboard` and `/v1/clinicians/me/patients`, but I will use your actual final paths.
5. A canonical latest-assessment response containing `subject_id`, `fusion_result_id`, current assessment, separate forecast, assessment status, modality availability/freshness/inclusion, timestamps and model version.
6. A persisted `ForecastResult` with explicit `scope`. Unless the research method changed, the current near-term forecast should remain physiological/C1-led rather than being labelled multimodal.
7. Persistent `AttentionEvent` creation linked to `subject_id`, `fusion_result_id` and optional `forecast_result_id`, including dedupe/episode handling.
8. Assignment-scoped event list/detail plus server-owned `OPEN -> ACKNOWLEDGED -> RESOLVED` transitions, recording actor and timestamps atomically.
9. Conflict semantics (for example 409) when another clinician has already changed event state.

Please keep the existing `POST /v1/clinical-notes -> C3 -> fusion` path and persisted `FusionResult` as the authoritative assessment source. ClinAnx will not compute fusion locally or call TC-WPN directly for authoritative inference.

Could you send either the updated OpenAPI JSON or the exact endpoint paths plus representative JSON responses for dashboard/patients/latest-assessment/attention-events and auth errors? Once those are frozen I can wire the live adapters without guessing.

Thanks.
