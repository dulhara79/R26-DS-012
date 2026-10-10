# Patient Authority and Shared Attention Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining P0 integration by giving Aura a patient-scoped session and making Aura and ClinAnx consume one server-owned AttentionEvent.

**Architecture:** The Central Backend issues proof-bound patient JWTs and exposes self-only risk, ingest, and event projections. Aura stores its proof/token securely and polls server events using the canonical event ID. ClinAnx keeps its current lifecycle contract while optionally supplying a resolution note.

**Tech Stack:** FastAPI, SQLAlchemy, PyJWT, pytest, Flutter/Dart, flutter_secure_storage, flutter_local_notifications.

**Spec:** `docs/superpowers/specs/2026-09-28-patient-authority-shared-events-design.md`

## Global Constraints

- Preserve all frozen clinician endpoint paths and JSON keys.
- Preserve `POST /v1/clinical-notes -> C3 -> fusion`.
- Keep C2 excluded and forecast scope physiological.
- Missing/stale/unavailable data must never become zero or Low.
- Empty `{}` resolve requests remain valid.
- No service credential may be compiled into either mobile app.

## Review Focus

- A guessed `app_user_id` without the installation secret must not mint a patient session.
- A valid patient token must not read or ingest for another subject.
- Expired/malformed/wrong-audience patient JWTs must return `401`, not `500`.
- Poll retries and app restarts must not create duplicate notifications for one `event_id`.
- Empty and maximum-length resolution notes must behave predictably and atomically.

---

### Task 1: Central Backend patient principal

**Files:**
- Modify: `central_backend/db_models.py`
- Create: `central_backend/patient_auth.py`
- Modify: `central_backend/main.py`
- Create: `central_backend/tests/test_patient_contract.py`

**Interfaces:**
- Produces: `PatientPrincipal`, `issue_patient_token(subject_id)`, `require_patient`, `PatientCredential`.
- `POST /v1/subjects/self` consumes `app_user_id` and `installation_secret` and returns `subject_id`, `access_token`, `token_type`, `expires_at`.

- [ ] Write failing tests for first enrolment, idempotent renewal, wrong proof, required claims and malformed/expired JWTs.
- [ ] Run `pytest tests/test_patient_contract.py -q` and confirm failures describe the missing patient principal.
- [ ] Add credential hashing, persistence, token issue/verification and response models.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit the backend patient-principal slice.

### Task 2: Self-only patient routes and event projection

**Files:**
- Modify: `central_backend/main.py`
- Modify: `central_backend/clinician_api.py`
- Modify: `central_backend/tests/test_patient_contract.py`
- Modify: `central_backend/tests/test_clinician_contract.py`

**Interfaces:**
- Consumes: `PatientPrincipal` and `require_patient` from Task 1.
- Produces: authenticated `/v1/patients/{subject_id}/risk`, patient-authorized ingest behavior, and `GET /v1/patients/me/attention-events`.

- [ ] Write failing tests for missing auth, self-only risk/ingest, cross-patient denial, safe event fields, and shared event ID.
- [ ] Run the focused tests and verify RED.
- [ ] Add patient/service authorization dependencies and subject-binding checks without changing clinician routes.
- [ ] Add the privacy-minimized patient event projection.
- [ ] Run patient, clinician, legacy and CARE contract suites.
- [ ] Commit the self-only route slice.

### Task 3: Atomic optional resolution note

**Files:**
- Modify: `central_backend/clinician_api.py`
- Modify: `central_backend/tests/test_clinician_contract.py`

**Interfaces:**
- Produces: resolve body `{note?: string}` and `resolution_note` in canonical event responses.

- [ ] Write failing tests for `{}`, note persistence, whitespace normalization, excessive length and conflict preservation.
- [ ] Run the focused lifecycle tests and verify RED.
- [ ] Implement the backward-compatible request/response change atomically.
- [ ] Run the backend contract and regression suites.
- [ ] Commit the resolution-note slice.

### Task 4: Aura secure patient session

**Files:**
- Modify: `pubspec.yaml`, `pubspec.lock`
- Create: `lib/services/patient_session_service.dart`
- Modify: `lib/services/api_service.dart`
- Modify: `lib/services/fusion_risk_service.dart`
- Modify: `lib/pages/login_page.dart`
- Modify: `lib/services/participant_identity_service.dart`
- Test: `test/patient_session_contract_test.dart`

**Interfaces:**
- Produces: `PatientSession`, secure proof/token persistence, `ensureSession(participantId)`, and authenticated central-backend headers.

- [ ] Write failing model/storage/request tests, including no `BACKEND_TOKEN` literal in active Dart.
- [ ] Run focused Flutter tests and verify RED.
- [ ] Add secure storage dependency and patient-session service.
- [ ] Replace shared backend headers and update enrolment/risk/ingest call sites.
- [ ] Run focused and full Flutter tests plus analysis.
- [ ] Commit the Aura session slice.

### Task 5: Aura server-owned AttentionEvent polling

**Files:**
- Create: `lib/services/patient_attention_event_service.dart`
- Modify: `lib/services/anxiety_feedback_service.dart`
- Modify: `lib/services/notification_helper.dart`
- Modify: `lib/main.dart`
- Test: `test/patient_attention_event_service_test.dart`
- Test: `test/anxiety_feedback_service_test.dart`

**Interfaces:**
- Consumes: patient JWT headers from Task 4 and `GET /v1/patients/me/attention-events` from Task 2.
- Produces: foreground/resume polling, exact-ID dedupe, server-event notification routing, and local feedback linked to the canonical `event_id`.

- [ ] Write failing tests for parsing, exact event ID, duplicate polls, restart persistence, minimal notification payload and network recovery.
- [ ] Add a source guard proving C1 forecast observation cannot create an authoritative event.
- [ ] Run focused tests and verify RED.
- [ ] Implement polling and server-event ingestion into the existing feedback/check-in flow.
- [ ] Remove local forecast-triggered authoritative event creation while preserving forecast display.
- [ ] Run full Flutter tests, analysis and debug APK build in CI.
- [ ] Commit the shared-event slice.

### Task 6: ClinAnx optional resolution note

**Files:**
- Modify: `lib/domain/contracts/attention_event.dart`
- Modify: `lib/domain/repositories/attention_event_repository.dart`
- Modify: `lib/data/repositories/central_backend_repositories.dart`
- Modify: `lib/state/attention_event_detail_controller.dart`
- Modify: `lib/features/attention_events/attention_event_detail_screen.dart`
- Modify matching fakes/tests under `test/`.

**Interfaces:**
- Produces: optional resolve note input and parsed `resolution_note`; existing note-less calls remain valid.

- [ ] Write failing repository/controller/widget tests for note-less and note-bearing resolve.
- [ ] Run focused tests and verify RED.
- [ ] Add the optional parameter, dialog and canonical-note display.
- [ ] Run focused and full ClinAnx suites plus analysis.
- [ ] Commit the ClinAnx note slice.

### Task 7: Cross-repository verification and delivery

**Files:**
- Modify relevant release evidence and OpenAPI/contract documentation in each repository.

**Interfaces:**
- Consumes all prior tasks.
- Produces three reviewable PRs and recorded CI evidence.

- [ ] Run backend compile, focused contracts, legacy regression and CARE boundary tests.
- [ ] Run both Flutter focused/full tests, analysis, authority/privacy guards and debug builds through CI.
- [ ] Review diffs for secrets, unrelated changes and frozen-contract drift.
- [ ] Push all three branches and create PRs with exact verification evidence.
- [ ] Merge only after required CI is successful and re-check all three `main` heads.
