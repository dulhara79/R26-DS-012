# Investor Mobile Demo Design

Date: 2026-08-31
Branch: `demo/investor-mobile-ready`
Base: `demo/login-reentry-fix` (itself based on `tcwpn/test_patch`)

## Goal

Produce one Android clinician APK that can be installed on a physical phone and demonstrated end-to-end without modifying `main`, `tcwpn/test_patch`, or unrelated branches.

The demo flow is:

1. Sign in as a clinician.
2. Add a patient by scanning the Aura QR code or entering the participant ID manually.
3. Resolve/attach the patient to the central backend.
4. Display current patient risk information.
5. Submit a clinical note.
6. Receive the deterministic investor-demo clinical score from the central backend simulator.
7. Refresh and display the authoritative fusion result from the central backend.
8. Raise an in-app alert and Android local notification for a newly observed high/urgent fusion state.
9. Allow the clinician to ask CARE evidence questions through the real CARE-AnxRAG service.
10. Keep dashboards, explanations, settings and research/model diagnostics available, but secondary to the doctor workflow.
11. Sign out and sign back in successfully without losing local patient records.

## System boundaries

### Clinician APK

The Flutter app remains a client. It does not compute the fusion score or fabricate RAG answers.

Primary responsibilities:

- clinician login/session persistence;
- QR scanning and manual participant ID entry;
- patient roster and local cache;
- clinical-note submission;
- rendering the backend clinical score and fusion timeline;
- displaying CARE-AnxRAG answers returned by the backend;
- Android local notifications for high/urgent results observed by the app;
- clinician-friendly navigation and terminology.

### Central Backend

Use `UVINDUSEN/component4final` branch `demo/investor-risk-simulation`.

The APK continues to call only the central backend for clinical workflow operations.

The demo backend already provides:

- subject enrolment / Aura attachment;
- clinical note ingestion;
- deterministic demo clinical scoring when `INVESTOR_DEMO_SIMULATION=1`;
- deterministic investor-demo fusion behavior;
- doctor timeline / patient risk endpoints;
- RAG proxy endpoints.

The demo simulation is opt-in and isolated from the normal research implementation.

### CARE-AnxRAG

Use `UVINDUSEN/Care-AnxRAG` `main`.

Relevant API contract:

- `GET /health`
- `POST /v1/ask` with `{ "question": "..." }`

`POST /v1/ask` is usable without the admin key for normal questions. Administrative/debug endpoints remain protected.

The central backend calls CARE-AnxRAG through its `RAG_URL` configuration. The clinician APK never calls CARE-AnxRAG directly.

For a phone demo, `RAG_URL` must resolve from the central backend deployment. `127.0.0.1` is valid only if the RAG service is running in the same reachable runtime environment as the central backend.

## Data flow

### Patient attachment

Aura QR -> Clinician APK -> participant ID `P_<16 hex>` -> Central Backend `/v1/subjects/attach` or resolve/enrol path -> backend `subject_id` -> cached locally per patient.

### Clinical scoring

Doctor enters note -> APK `POST /v1/clinical-notes` through `CentralBackendGateway` -> demo backend deterministic clinical simulator -> response score/detail -> APK stores and renders note result.

### Fusion

The central backend owns the fusion result. The APK never recreates it locally.

After ingestion or explicit refresh:

APK -> Central Backend doctor timeline -> latest fusion row -> clinician-friendly overall risk display -> local cache.

For the current investor simulation, the backend regression contract demonstrates a physiological `89.67` input producing fusion `0.8967` and the same composite being exposed to patient and doctor views.

### RAG

Doctor question -> APK `/v1/evidence/ask` -> Central Backend -> CARE-AnxRAG `/v1/ask` -> grounded answer/citations -> Central Backend -> APK.

No simulated RAG answer is created in Flutter.

## Android-specific fixes

### Camera

The app already uses `mobile_scanner` for Aura QR scanning. The Android release manifest must declare `android.permission.CAMERA`.

The app must gracefully fall back to manual participant ID entry when camera permission is denied or the camera cannot initialize.

### Notifications

The project already depends on `flutter_local_notifications`, but the notification system is not initialized and Android notification permission is not configured.

Add:

- notification service initialization at app startup;
- Android 13+ `POST_NOTIFICATIONS` manifest permission and runtime request;
- a high-priority anxiety-risk notification channel;
- notification delivery when the app observes a transition into `RED`/`DARK RED` (High/Urgent) fusion state;
- deduplication so repeatedly refreshing the same fusion result does not spam the clinician.

Scope: local Android notifications while the app is running/refreshing data. True server push while the app is fully closed is out of scope for this demo because the repository has no FCM/device-token backend infrastructure.

## Authentication

Use the login re-entry fix inherited from `demo/login-reentry-fix`:

- local demo credentials remain available for the isolated demo build when no remote auth service is configured;
- successful sign-in stores the session securely and populates the in-memory `Session` immediately;
- sign-out clears only the clinician session, not patient records.

This demo-only auth behavior must not be merged into production without replacing local demo credentials with real clinician authentication.

## Clinician-facing UX

The app should present clinical concepts first and research/model details second.

Primary navigation target:

- Home
- Patients
- Alerts
- Clinical Assistant
- More

Primary wording examples:

- `Overall anxiety risk`
- `Clinical note assessment`
- `Current risk`
- `Needs review`
- `Evidence assistant`

Avoid putting internal wire keys, component numbers, model versions, renormalisation terminology, calibration implementation detail, project IDs, or research-framework wording in the primary workflow unless the clinician explicitly opens technical details.

Research/model diagnostics remain accessible under secondary screens for the demo team.

## High-risk behavior

When a fresh fusion result is High/Urgent:

1. Save/update the authoritative fusion result.
2. Create an in-app `ClinicalAlert` if this fusion result has not already produced one.
3. Show an Android local notification with patient identifier/name and a clinician-safe summary.
4. Make the patient visible in the `Needs review` section.
5. Do not diagnose or prescribe in notification text.

## Error handling

The APK must distinguish:

- backend unavailable;
- patient not yet attached/enrolled;
- QR/camera permission failure;
- clinical-note analysis failure;
- fusion unavailable/insufficient;
- RAG unavailable;
- RAG abstention;
- RAG crisis bypass.

Do not fabricate scores, answers, citations, or service availability.

For the investor demo, a service failure should produce a clear retry action and concise clinician-readable message rather than exposing raw exception text in the primary UI.

## Verification requirements

Before calling the demo branch ready, verify as much as the available environment permits and explicitly report any physical-device-only checks that still require the user's handset.

Required checks:

- login -> logout -> login regression;
- camera permission present in release manifest;
- notification permission/configuration present;
- notification deduplication unit test;
- clinical note low/high result parsing;
- fusion timeline parsing and refresh;
- RAG response/abstention/unavailable parsing;
- `flutter analyze`;
- `flutter test`;
- release APK build;
- central backend investor-demo regression test;
- central backend health indicates RAG reachability in the deployment used for the demo;
- physical-device smoke test: install APK, login, scan QR, submit note, see fusion, trigger high-risk notification, ask CARE question, sign out and sign in again.

## Non-goals

Do not:

- modify `main` or `tcwpn/test_patch`;
- replace the real CARE-AnxRAG response with canned Flutter text;
- add Firebase/FCM server push in this demo hardening pass;
- rewrite the research fusion implementation;
- expose simulated values as real clinical predictions outside the clearly isolated investor-demo backend mode;
- perform unrelated refactors.
