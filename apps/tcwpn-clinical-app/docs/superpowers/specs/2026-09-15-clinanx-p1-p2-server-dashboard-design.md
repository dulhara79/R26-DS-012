# ClinAnx P1/P2 Server-Backed Dashboard Design

**Date:** 2026-09-15
**Repository:** `dulhara79/tcwpn_mobile_app`
**Target branch:** `integration/clinanx-p1-p2-server-dashboard`
**Status:** Approved in chat; written spec pending user review

## 1. Purpose

This slice moves ClinAnx from a local/demo-style clinician dashboard to the target architecture described by the ClinAnx technical architecture specification and the system integration handbook.

The goal is a single end-to-end path:

1. an authenticated clinician session is used for Central Backend requests;
2. the backend enforces clinician-to-patient assignment;
3. ClinAnx loads assignment-scoped patient summaries and server-created OPEN attention events;
4. the Dashboard renders those server-authoritative objects without locally recomputing risk, urgency, or event identity.

This slice deliberately combines the minimum P1 mobile-foundation work needed for a real P2 Dashboard. It does not attempt the full remaining ClinAnx roadmap.

## 2. Source-of-truth rules

The following rules are hard constraints for this slice:

- The Central Backend is authoritative for current multimodal assessment state.
- `FusionResult` / `AssessmentSummary` remains the source for current assessment identity and `fusion_result_id`.
- The near-term forecast remains a separate object from the current assessment.
- Unless the research method has genuinely changed, the forecast is presented as physiological/C1-led, not as a validated multimodal forecast.
- `AttentionEvent` is created by the backend. The mobile app must not mint a second authoritative urgent-event id.
- Patient access is assignment-scoped and enforced by the backend, not by hiding rows in Flutter.
- Missing, stale, unsupported, experimental, or unavailable data never becomes zero, low, green, or reassuring by default.
- C2 remains explicitly experimental / excluded from fusion unless the backend contract and research governance have changed.
- TC-WPN/C3 remains one component signal, not the overall patient risk.

## 3. Backend-contract verification gate

Implementation must begin by inspecting the actual updated Central Backend route definitions / OpenAPI / representative JSON. The client must not invent endpoint paths simply because the architecture document includes target examples.

Before network wiring, verify the actual backend provides the following semantics:

- clinician principal / bearer-session verification;
- assignment-scoped clinician dashboard or equivalent patient-list retrieval;
- canonical `PatientSummary` / latest `AssessmentSummary` retrieval;
- separate `ForecastResult` with explicit scope;
- server-persistent `AttentionEvent` list/detail;
- OPEN / ACKNOWLEDGED / RESOLVED lifecycle;
- server actor/timestamps for event transitions;
- 401 for invalid/expired identity;
- 403 for valid clinician without permission;
- 409 or equivalent concurrency/conflict behavior where applicable.

If the updated backend is not available when coding starts, the repository and UI layers may be completed against frozen P0 fixtures, but live-route wiring must stop until the real contract can be inspected.

## 4. Architecture

### 4.1 Target client layering

```text
Presentation / screens
        |
        v
Controllers / state
        |
        v
Repositories
        |
        v
CentralBackendGateway / ApiClient
        |
        v
Central Backend
```

Widgets must not know endpoint paths, HTTP status codes, or response-shape details.

### 4.2 Repository boundaries

Add or refine these repository interfaces for the new workflow:

- `AuthRepository`
- `PatientRepository`
- `AssessmentRepository`
- `AttentionEventRepository`

Do not refactor unrelated legacy features merely to make every feature use repositories in this PR. The new server-backed Dashboard path should use them first.

Future slices can add:

- `EvidenceRepository`
- `ClinicalNoteRepository`

### 4.3 Existing infrastructure to reuse

Reuse rather than rewrite:

- `ApiClient`
- `AuthService`
- `Session`
- `SecureStore`
- Provider-based state management
- existing P0 contracts under `lib/domain/contracts/`
- existing shell/navigation where it remains compatible

## 5. Authentication and authorization

### 5.1 Clinician session

The target flow is:

```text
Login
  -> AuthService
  -> SecureStore
  -> Session.token
  -> Central Backend Authorization header
```

The clinician session credential must be the credential used for assignment-scoped Central Backend calls.

No privileged shared backend token should be treated as a clinician identity.

### 5.2 Error semantics

Refine API failures so they preserve the authorization distinction:

- `401` -> session invalid/expired; prompt re-authentication;
- `403` -> clinician authenticated but not permitted; keep session active;
- `404` -> requested resource not found;
- `409` -> state/concurrency conflict; refresh canonical server state;
- `400/422` -> request validation problem;
- `5xx` -> backend/service failure;
- timeout -> timeout state;
- no network -> offline state.

A 403 must never be presented as "session expired".

## 6. Dashboard data contract

The new Dashboard state is composed from server-authoritative objects only.

Conceptually:

```text
DashboardState
  openEvents: AttentionEvent[]
  assignedPatients: PatientSummary[]
  lastUpdated: DateTime?
  isFromCache: bool
```

The exact transport JSON may differ; repositories adapt real backend responses into the P0 domain contracts.

### 6.1 Ordering

Render in this order:

1. OPEN attention events requiring clinician action;
2. assigned patients.

Within each section, use backend ordering if the server contract defines one. The client must not create a hidden risk formula for ranking.

### 6.2 Patient row contents

A patient row may show:

- safe display identifier / pseudonymous label;
- current assessment tier/status;
- current assessment timestamp/freshness;
- separate near-term forecast summary when present;
- assessment completeness (`complete`, `partial`, `unavailable`, or conservative unknown);
- open event count if provided;
- explicit unavailable/stale labels.

Do not derive an authoritative tier from a locally stored TC-WPN score.

### 6.3 Attention-event card contents

For OPEN events, preserve server fields such as:

- `event_id`;
- `subject_id`;
- `fusion_result_id`;
- optional `forecast_result_id`;
- severity;
- reason / event type;
- forecast horizon if provided;
- `created_at`;
- lifecycle status.

The Dashboard does not acknowledge or resolve events in this slice. Those actions belong to the P4 event workflow.

## 7. Replace the current local Dashboard source

The existing Dashboard currently summarizes local clinical-note assessments. That path must no longer drive the authoritative clinician Dashboard once the server target exists.

Remove from the authoritative Dashboard path:

- loading all local notes to compute patient urgency;
- computing `AlertBand` from local note probabilities;
- calculating server-facing priority from local TC-WPN results;
- KPI language that implies local note scores are current multimodal risk.

Local notes remain available in their clinical-note feature; they are not deleted.

## 8. Attention-event migration boundary

### 8.1 Stop creating new authoritative local urgent events

`ChartController.refreshFusion()` currently leads to `_raiseIfEscalated()` and creates local `ClinicalAlert` records for RED/DARK RED fusion bands.

Once server `AttentionEvent` exists, that local generation path must no longer create the current authoritative urgent event.

For this slice:

- Dashboard urgency comes from backend `AttentionEvent` only;
- stop minting new local risk-escalation alerts from fusion refresh;
- keep existing local alert storage/models readable temporarily so this PR does not mix in the full Activity migration.

The next Attention Event slice will replace old alert inbox/detail behavior and then remove obsolete local-authoritative paths cleanly.

## 9. P0 identity cleanup

Finish the temporary compatibility cleanup introduced during P0:

- `ChartController.ensureEnrolled()` should call `resolveAppUserId()` directly for Aura-style participant IDs;
- remove the deprecated gateway `attach()` compatibility shim once no call sites remain;
- preserve current valid identity behavior;
- do not reintroduce `/v1/subjects/attach`.

Lookup behavior should remain conservative and conflict-safe.

## 10. State model

Introduce a reusable typed state for new server-backed flows instead of continuing ad hoc combinations of booleans and nullable errors.

Conceptually:

```text
AsyncDataState<T>
  loading
  data(T)
  empty
  partial(T)
  unavailable
  offline(cached?)
  error
```

The exact Dart representation may be sealed classes or an enum + payload, whichever best fits the current project style.

The Dashboard-specific state should preserve:

- `openEvents`;
- `assignedPatients`;
- `lastUpdated`;
- cached/offline provenance;
- user-safe error information.

## 11. Safe-state rules

The following must be rendered explicitly rather than collapsed:

- no assessment yet;
- assessment unavailable;
- partial assessment;
- stale physiological signal;
- C3 unavailable;
- C2 experimental / excluded;
- forecast unavailable;
- offline with cached server data;
- no assigned patients;
- no open events;
- forbidden patient access;
- expired session;
- backend timeout / service failure.

Examples of forbidden shortcuts:

- null score -> `0.0`;
- missing forecast -> low forecast;
- unavailable C3 -> low C3;
- stale C1 -> healthy physiology;
- no event -> infer clearance.

## 12. UI direction

The Dashboard should be a clinician worklist rather than an analytics-first KPI screen.

Suggested hierarchy:

```text
Dashboard
Clinician identity / refresh status

Needs attention
  OPEN AttentionEvent cards

Assigned patients
  PatientSummary rows
```

The UI should prioritize clarity and actionability over charts.

This slice does not redesign Patient Overview, Contributions, Evidence, Data Quality, or history charts.

## 13. Caching and offline behavior

Server data may be cached for continuity, but cached data must never look live.

Rules:

- preserve last successful server payload where practical;
- stamp last-updated time;
- show an explicit offline/stale banner when network refresh fails;
- do not locally recalculate new assessment/event state while offline;
- do not convert cached data into a new server event or new fusion result.

## 14. Testing strategy

Use TDD for implementation.

Required tests include:

### 14.1 Auth/error contract

- clinician bearer token is sent to Central Backend requests;
- 401 maps to session-expired/unauthorized behavior;
- 403 maps to forbidden without clearing a valid session;
- 409 remains distinguishable from ordinary validation failures.

### 14.2 Repository contract

- real transport responses map to P0 domain contracts without unsafe defaults;
- unknown enum values remain conservative/unknown;
- missing scores remain nullable;
- current assessment and forecast remain separate.

### 14.3 Dashboard behavior

- only server-returned assigned patients are displayed;
- OPEN events render before ordinary patients;
- server `event_id` is preserved;
- patient rows preserve `fusion_result_id` when supplied;
- `partial` / `unavailable` render as such;
- C2 excluded is explicit;
- C3 unavailable is not mapped to zero/low;
- stale C1 remains stale;
- forecast scope/label remains separate from current assessment;
- cached/offline state is visibly marked.

### 14.4 Regression / migration

- a locally refreshed RED fusion does not mint a new authoritative urgent event once the server-event path is active;
- P0 identity resolution still works;
- no `/v1/subjects/attach` route is used;
- TC-WPN direct inference is not reintroduced;
- fusion math remains server-side.

### 14.5 Verification commands

At minimum:

```bash
flutter test
flutter analyze
```

If the repository still contains known pre-existing baseline failures, focused blocking gates may be used, but the PR must clearly distinguish new-slice verification from baseline noise.

## 15. Explicit non-goals

Do not implement in this slice:

- Patient Overview redesign;
- event acknowledge/resolve UI;
- event detail screen;
- Activity timeline migration;
- push/FCM notifications;
- websocket/realtime streaming;
- contributions redesign;
- assessment-history redesign;
- evidence/RAG redesign;
- Data Quality deep view;
- patient-app changes;
- central fusion changes;
- TC-WPN model changes;
- C1/C2/C4 model changes;
- research-method changes.

## 16. External repository policy

This implementation branch changes only `dulhara79/tcwpn_mobile_app`.

Repositories owned by teammates remain read-only from this workflow:

- `DewduSendanayake/anxiety_mobile_app`
- `UVINDUSEN/component4final`
- `UVINDUSEN/Care-AnxRAG`

If live integration reveals a required teammate change, prepare a precise message containing:

- observed contract mismatch;
- expected behavior;
- exact request/response example where available;
- acceptance criteria;
- no direct write to their repositories.

## 17. Acceptance criteria

This slice is complete when all of the following are true:

1. ClinAnx uses the authenticated clinician session for Central Backend clinician-scoped requests.
2. 401 and 403 are represented separately.
3. New Dashboard data comes from assignment-scoped backend responses, not local clinical-note scoring.
4. OPEN backend AttentionEvents are the Dashboard's urgent-work source.
5. Assigned patients come only from the backend assignment-scoped response.
6. Current multimodal assessment and near-term forecast remain separate in data and UI.
7. Missing/stale/unavailable signals never become low/zero/green defaults.
8. C2 remains explicitly experimental/excluded unless the real backend/research policy says otherwise.
9. New local risk-escalation alerts are no longer minted as authoritative events.
10. Existing legacy local alerts remain readable until the P4 migration.
11. The P0 `attach()` compatibility shim is removed and Aura IDs resolve via `resolveAppUserId()` directly.
12. No nonexistent endpoint is introduced.
13. The live route mapping is based on the actual updated backend implementation, not guessed target URLs.
14. Tests cover auth semantics, repository mapping, safe-state behavior, server event identity, and identity-regression cases.
15. `flutter test` / `flutter analyze` verification results are recorded accurately before the PR is opened.

## 18. Follow-on slices

After this PR:

1. **P3 Patient Overview** — fusion-first patient view, forecast separate, modality signals, freshness/status.
2. **P4 Attention Events** — event inbox/detail, ACK/RESOLVE, concurrency reconciliation, Activity migration.
3. **P5 Deep Clinical Views** — history, contributions, evidence, Data Quality.
4. **P6 Notifications** — push / foreground refresh using the same backend event id.
5. **P7 Hardening** — offline, security, accessibility, failure injection.
6. **P8 Research release** — release verification and reproducibility documentation.
