# ClinAnx P4 Server Attention Events Design

**Date:** 2026-09-16
**Repository:** `dulhara79/tcwpn_mobile_app`
**Target branch:** `integration/clinanx-p4-attention-events`
**Base commit:** `1477c8d665ecf4de5be9ba05428b1b972a893ef3`
**Status:** Approved in chat; written spec pending user review

## 1. Purpose

P4 replaces the legacy clinician-alert inbox as the target source of urgent clinician work with the server-owned persistent `AttentionEvent` lifecycle defined by the ClinAnx technical specification and the system integration handbook.

The target lifecycle is:

```text
OPEN
  -> ACKNOWLEDGED
  -> RESOLVED
```

The Central Backend remains authoritative for event creation, assignment filtering, event identity, lifecycle state, actors, timestamps, concurrency, and auditability. ClinAnx renders and requests state transitions; it does not create its own authoritative acute-escalation event.

This slice builds the clinician-side event workflow even though the live backend contract is not yet verified. Repository interfaces, controllers, screens, fixtures, and tests may be completed now; production event transport remains contract-gated until the backend owner provides an implemented/OpenAPI-verified contract.

## 2. Source-of-truth rules

The following are hard invariants:

- `AttentionEvent.id` is server-owned and must be preserved end to end.
- `subject_id`, `fusion_result_id`, optional `forecast_result_id`, event type, severity, reason, horizon, policy version, lifecycle state, actors, and timestamps come from the server object.
- ClinAnx must never infer a new authoritative event from a RED/DARK RED band, current fusion tier, TC-WPN score, or UI color.
- Acknowledge and resolve are server mutations. Local state may optimistically show a pending action, but it must never manufacture the final actor/time/state.
- If two assigned clinicians act concurrently, the server response wins. The second client refreshes and displays the current server state.
- Missing event data is unavailable/unknown, not LOW, cleared, acknowledged, or resolved.
- Assignment authorization is enforced by the backend. Flutter filtering is not access control.
- Forecast wording must remain scientifically cautious. The app may state that potential escalation was predicted within the forecast horizon; it must not claim an exact attack time.
- Push notifications are not implemented in P4. When P6 is added, push payloads should carry minimal technical identifiers and fetch sensitive detail after authenticated app open.

## 3. Backend contract verification gate

The backend repository `UVINDUSEN/component4final` was rechecked before this design. Its visible `main` remains at `a1ceef8daba268dac24d81aedd76c89e7c9ccc6a` and does not yet provide a verified persistent AttentionEvent list/detail/acknowledge/resolve contract.

Therefore this branch must not invent endpoint paths from architecture examples.

The following semantics must be verified against the running OpenAPI/current backend SHA before production transport is enabled:

- assignment-scoped event list;
- event detail by server `event_id`;
- acknowledge mutation;
- resolve mutation;
- canonical mutation response carrying current lifecycle state;
- actor/timestamp fields returned by the server;
- 401 for invalid/expired clinician identity;
- 403 for authenticated but unauthorized clinician;
- 404 for missing event where applicable;
- 409 or equivalent canonical concurrency behavior;
- filtering by assigned patient/event scope;
- audit persistence for creation, acknowledgement and resolution.

Until that contract exists, `CentralBackendAttentionEventRepository` must return `ApiFailure.notConfigured` for live event operations.

## 4. Existing code to reuse

P4 should extend existing P0-P3 infrastructure instead of introducing a parallel architecture.

Reuse:

- `AttentionEvent` in `lib/domain/contracts/attention_event.dart`;
- `AttentionEventStatus` / `AttentionSeverity` in `contract_enums.dart`;
- `AttentionEventRepository` as the repository boundary, extending it beyond `openEvents()`;
- `AsyncDataState<T>` for loading/error/authorization/offline state;
- `ApiException` / `ApiFailure` mappings;
- `CentralBackendAuthRepository` for session-expiry handling;
- the server Dashboard's existing `AttentionEvent` cards;
- P3 canonical-subject navigation;
- Provider/ChangeNotifier patterns already used by Dashboard and Patient Overview.

The large legacy `models.dart`, `RosterController`, `ClinicalAlert`, and local alert store remain compatibility code only. P4 must not make them the authoritative event source.

## 5. Domain and repository design

### 5.1 AttentionEvent contract

Keep the existing event model and only add fields if the verified source documents/backend contract require them.

Expected target fields already represented by the P0 contract:

```text
id
subject_id
fusion_result_id
forecast_result_id?
event_type
severity
reason
forecast_horizon
status
created_at
acknowledged_at?
acknowledged_by?
resolved_at?
resolved_by?
policy_version?
```

Unknown values remain conservative `unknown`/nullable values.

### 5.2 Repository interface

Expand `AttentionEventRepository` to express the workflow without leaking HTTP:

```text
Future<List<AttentionEvent>> openEvents()
Future<List<AttentionEvent>> activity({String? subjectId})
Future<AttentionEvent?> eventById(String eventId)
Future<AttentionEvent> acknowledge(String eventId)
Future<AttentionEvent> resolve(String eventId)
```

The exact method names may vary slightly to fit Dart style, but the semantics must stay the same.

The repository returns the server's canonical `AttentionEvent` after each mutation. It does not accept locally supplied actor/timestamp/state fields.

## 6. State/controllers

Introduce an `AttentionEventsController` responsible for the event worklist/activity feed and an `AttentionEventDetailController` for one event's lifecycle.

### 6.1 AttentionEventsController

Responsibilities:

- load event activity from `AttentionEventRepository`;
- preserve server ordering where defined;
- expose explicit loading/data/empty/unavailable/offline/error states;
- never synthesize events from local fusion data;
- refresh after detail mutations when the user returns to the list.

For fixture-driven development, a fake repository supplies OPEN/ACKNOWLEDGED/RESOLVED examples without pretending to be live backend data.

### 6.2 AttentionEventDetailController

Responsibilities:

- load the canonical event by `event_id`;
- acknowledge an OPEN event;
- resolve an ACKNOWLEDGED event;
- replace local display state with the canonical event returned by the repository;
- handle 401/403/404/409/network failures distinctly;
- on 409 or stale-state conflict, reload the event and show the server's current actor/time/state;
- disable duplicate mutation while a mutation is in flight;
- never locally stamp `acknowledged_by`, `acknowledged_at`, `resolved_by`, or `resolved_at`.

## 7. UI design

### 7.1 Navigation label

The current bottom-navigation `Alerts` destination should migrate to `Activity` for the target architecture.

The Activity destination represents persistent server event history rather than a transient local alert inbox.

### 7.2 Activity screen

Replace the current authoritative `AlertsScreen` behavior with a server-backed Attention Events screen.

Hierarchy:

```text
Activity

Open
  OPEN AttentionEvent rows

Acknowledged
  ACKNOWLEDGED rows

Resolved
  recent RESOLVED rows
```

Each row should display only fields supported by the event contract, for example:

- safe participant/display identifier when available from surrounding assignment data, otherwise canonical `subject_id`;
- event type / concise reason;
- severity;
- forecast horizon when supplied;
- created time;
- lifecycle status;
- acknowledged actor/time for acknowledged/resolved events;
- resolved actor/time for resolved events.

No UI color alone should imply a lifecycle transition.

### 7.3 Event detail screen

The event detail view should show:

- `event_id`;
- subject identity;
- status;
- severity;
- reason;
- event type;
- forecast horizon;
- created time;
- `fusion_result_id`;
- optional `forecast_result_id`;
- policy version when present;
- acknowledgement actor/time when present;
- resolution actor/time when present;
- link to Patient Overview using the canonical `subject_id`.

Actions:

```text
OPEN          -> Acknowledge
ACKNOWLEDGED  -> Resolve
RESOLVED      -> no mutation action
UNKNOWN       -> no mutation action
```

The app does not allow Resolve directly from OPEN unless the verified backend contract explicitly supports that transition. The current target lifecycle is sequential.

### 7.4 Dashboard integration

Dashboard OPEN event cards should become tappable and open the same event-detail flow by server `event_id`.

Dashboard must continue to use server-provided OPEN events only. P4 does not derive urgent cards from patient score/tier/color.

### 7.5 Patient Overview integration

P4 may add an entry point from Patient Overview to subject-scoped event activity if this can be done without implementing P5 history/timeline features. The primary required P4 navigation is Dashboard/Activity -> Event Detail -> Patient Overview.

## 8. Lifecycle mutation behavior

### 8.1 Acknowledge

When the clinician taps Acknowledge:

1. disable the action while the request is in flight;
2. call repository `acknowledge(eventId)`;
3. replace detail state with the returned canonical event;
4. display server `acknowledged_by` and `acknowledged_at`;
5. refresh list/dashboard state as appropriate.

The client must not send or manufacture the actor identifier if the target backend derives the actor from the authenticated clinician principal.

### 8.2 Resolve

When the clinician taps Resolve:

1. require the displayed canonical event to be ACKNOWLEDGED;
2. disable the action while the request is in flight;
3. call repository `resolve(eventId)`;
4. replace detail state with the returned canonical event;
5. display server `resolved_by` and `resolved_at`;
6. refresh list/dashboard state as appropriate.

### 8.3 Concurrency

If another clinician has already changed the event:

- treat 409/canonical-state conflict as a refresh condition;
- fetch the event again;
- show who acted and when from server state;
- do not overwrite server state with the local clinician's intended transition;
- show a concise message that the event changed on the server.

## 9. Legacy ClinicalAlert migration

P4 is the migration point away from the legacy local alert inbox.

Rules:

- `ClinicalAlert` remains readable only where necessary for backwards compatibility during this branch.
- `RosterController.alerts`, `SecureStore`/local alert persistence, and legacy acknowledge behavior must not drive the target Activity screen.
- No new authoritative `ClinicalAlert` is created from current fusion or TC-WPN data.
- The bottom-navigation badge must stop reading `roster.unacknowledgedCount` as the target urgent-event count.
- If no server AttentionEvent contract is configured, Activity must show an explicit unavailable/not-configured state rather than silently falling back to local alerts as if they were current server events.
- Complete deletion of unused `ClinicalAlert` persistence may be deferred if removal would create unrelated regression risk, but any retained path must be clearly non-authoritative.

## 10. Error and safe-state behavior

### 10.1 Authorization

- `401` -> expire clinician session and present session-expired state;
- `403` -> keep valid session; show permission/assignment denial;
- `404` -> event no longer available/not found; offer return/refresh;
- `409` -> refresh canonical server state;
- `400/422` -> validation/request problem;
- `5xx`, timeout, offline -> explicit service/offline state.

### 10.2 Safety invariants

Forbidden fallbacks:

- missing event -> cleared;
- missing status -> OPEN or RESOLVED;
- missing actor -> current clinician;
- missing timestamp -> device time;
- failed acknowledge -> acknowledged locally;
- failed resolve -> resolved locally;
- offline mutation -> queued as if committed;
- no open event -> infer patient is clinically safe;
- RED UI color -> create event.

## 11. Fixture strategy while backend is missing

P4 can be developed against typed fake repositories and existing P0 JSON fixtures.

Add fixture coverage as needed for:

- OPEN event;
- ACKNOWLEDGED event;
- RESOLVED event;
- event with unknown fields;
- concurrent/stale mutation scenario through fake repository behavior;
- empty activity list;
- unauthorized/forbidden/conflict/offline repository failures.

Fixture/demo data must be injected explicitly in tests or demo-only code. Production construction must continue to use the contract-gated Central Backend repository and must never masquerade fixture data as live research data.

## 12. Testing strategy

Use TDD. Each behavior is introduced by a failing test before implementation.

### 12.1 Repository contract tests

- `openEvents()` preserves server event ids;
- activity/detail preserve lifecycle fields;
- acknowledge returns and uses server actor/time;
- resolve returns and uses server actor/time;
- no mutation accepts locally supplied final actor/timestamp;
- unverified production adapter returns `ApiFailure.notConfigured` without making a guessed network request.

### 12.2 Controller tests

- OPEN event can be acknowledged;
- ACKNOWLEDGED event can be resolved;
- RESOLVED event exposes no mutation action;
- UNKNOWN event exposes no unsafe mutation action;
- 401 expires session;
- 403 does not expire session;
- 404 is explicit missing/not-found state;
- 409 reloads canonical event state;
- offline/server failure does not mutate displayed canonical state to a fake success;
- duplicate taps do not produce concurrent duplicate mutations.

### 12.3 Widget tests

- Activity groups OPEN / ACKNOWLEDGED / RESOLVED states correctly;
- event detail shows canonical IDs and lifecycle provenance;
- Acknowledge button appears only for OPEN;
- Resolve button appears only for ACKNOWLEDGED;
- no action appears for RESOLVED/UNKNOWN;
- Dashboard event card opens detail with the same `event_id`;
- Event detail opens Patient Overview with the same `subject_id`;
- unavailable event service is visibly unavailable rather than silently falling back to local alerts.

### 12.4 Authority regression tests

Guard active Dart against:

- creation of new authoritative `ClinicalAlert` from fusion/score/color;
- local mutation of `AttentionEvent.status` to fake success;
- `AlertBandX.fromScore` or equivalent local urgency derivation inside server-backed event views;
- direct C1/C3 model calls from Attention Event flows;
- guessed `/attention-events`, `/acknowledge`, `/resolve`, or similar live routes in production adapters before contract verification;
- bottom-nav urgent count sourced from legacy `RosterController.unacknowledgedCount` after migration.

### 12.5 Verification

Blocking checks for the P4 slice:

```bash
flutter test <focused P0-P4 test set>
flutter analyze <focused P0-P4 files>
authority/forbidden-pattern checks
flutter analyze
flutter test
```

The known baseline consent/sign-in widget expectation must continue to be reported accurately if it still fails. A visibility-only CI step must not be described as proof that all tests passed.

## 13. CI updates

Extend the existing clinician integration workflow so focused blocking checks include:

- AttentionEvent repository/controller tests;
- Activity/detail widget tests;
- Dashboard event-detail navigation test;
- P4 authority guards;
- focused analysis for the P4 files.

Keep whole-repository test/analyze visibility, while preserving truthful distinction between blocking focused gates and any pre-existing whole-suite failure.

## 14. Explicit non-goals

P4 does not implement:

- backend AttentionEngine policy logic;
- backend database migrations;
- clinician assignment persistence;
- push/FCM notifications;
- websocket/realtime streaming;
- patient-app event UI;
- event creation from Flutter;
- new fusion thresholds;
- forecast model changes;
- P5 contributions/timeline/notes/evidence/data-quality views;
- P6 notification delivery;
- P7 hardening beyond what is necessary for safe P4 state handling.

No teammate-owned repository will be modified.

## 15. Acceptance criteria

P4 is ready for PR review when all of the following are true:

1. Activity is backed by `AttentionEventRepository`, not `RosterController.alerts`.
2. OPEN, ACKNOWLEDGED and RESOLVED states render distinctly.
3. Event detail preserves server `event_id`, `subject_id`, source result ids and lifecycle provenance.
4. OPEN -> ACKNOWLEDGED and ACKNOWLEDGED -> RESOLVED are requested through repository mutations only.
5. Server-returned actor/time/state replaces local display state after mutations.
6. 401, 403, 404, 409 and offline/server failures remain distinguishable.
7. Concurrent server changes reconcile to the server's current state.
8. Dashboard OPEN event cards open detail using the same event id.
9. Event detail opens Patient Overview using the same canonical subject id.
10. The target bottom-nav Activity badge/count does not use legacy local alert authority.
11. Production event transport remains `notConfigured` until the exact live backend contract is verified.
12. No guessed AttentionEvent route is live-wired.
13. No local score/color/TC-WPN logic creates an authoritative event.
14. Retained `ClinicalAlert` code is compatibility-only and cannot silently substitute for server events.
15. Focused P4 tests, focused analysis, authority checks, whole analysis and whole-suite visibility are recorded accurately before the PR is opened.

## 16. Follow-on work

After P4:

1. **P5 Deep Clinical Views** — contributions, timeline, clinical notes, CARE-AnxRAG evidence, data quality, clinician assessment.
2. **P6 Notifications** — minimal-PHI push delivery keyed by the same server `event_id`, foreground refresh, notification navigation.
3. **P7 Hardening** — offline edge cases, accessibility, security, failure injection, release configuration.
4. **P8 Research Release** — reproducibility, build/version pinning, release verification and handoff.
