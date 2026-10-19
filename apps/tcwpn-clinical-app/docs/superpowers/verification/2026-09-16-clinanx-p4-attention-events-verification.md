# ClinAnx P4 Server Attention Events Verification

**Date:** 2026-09-16
**Repository:** `dulhara79/tcwpn_mobile_app`
**Branch:** `integration/clinanx-p4-attention-events`
**Base `main`:** `1477c8d665ecf4de5be9ba05428b1b972a893ef3`
**Verified implementation head:** `f779205a0e82ce688d29922da916945865b22f9d`
**GitHub Actions run:** `35061243450`
**Job:** `104681825787`

## Scope verified

P4 implements the clinician-side persistent AttentionEvent workflow while keeping the Central Backend authoritative for event creation and lifecycle state.

Verified behavior includes:

- repository boundary for event activity, detail, acknowledge and resolve;
- `OPEN -> ACKNOWLEDGED -> RESOLVED` client workflow;
- server-returned event object replaces client state after mutation;
- no locally manufactured actor, timestamp, event identity or successful lifecycle state;
- 401 session expiry and 403 authorization remain distinct;
- 404/unavailable handling;
- 409 reloads canonical server state and displays the server actor/time;
- duplicate lifecycle mutations are guarded while a request is in flight;
- Activity groups OPEN, ACKNOWLEDGED, RESOLVED and UNKNOWN events without inventing clinical ranking inside each group;
- Activity does not fall back to legacy `ClinicalAlert` records when target transport is unavailable;
- Dashboard opens Event Detail using the same server `event_id`;
- Event Detail opens Patient Overview using the same canonical `subject_id`;
- App shell target navigation is `Dashboard | Patients | Activity | Settings | Ask CARE`;
- the target Activity tab no longer derives its badge/count from `RosterController.unacknowledgedCount`;
- production AttentionEvent transport remains contract-gated instead of guessing target endpoint paths.

## Backend contract gate

The teammate-owned Central Backend repository was rechecked read-only before final verification.

Visible `UVINDUSEN/component4final` `main` remains:

```text
a1ceef8daba268dac24d81aedd76c89e7c9ccc6a
```

Repository search did not verify a persistent AttentionEvent list/detail/acknowledge/resolve implementation on that visible revision.

Therefore `CentralBackendAttentionEventRepository` deliberately returns `ApiFailure.notConfigured` for:

- open-event retrieval;
- activity retrieval;
- event detail;
- acknowledge;
- resolve.

No guessed `/attention-events`, `/acknowledge`, `/resolve`, or equivalent production path was introduced. The adapter can be wired only after the implemented backend/OpenAPI contract is inspected.

No teammate-owned repository was modified.

## TDD evidence

### Repository boundary

The first repository test run failed because `CentralBackendAttentionEventRepository` did not exist. The implementation then added the five-method `AttentionEventRepository` workflow and a no-network, contract-gated production adapter.

Final repository tests verify that every unverified target operation fails with `ApiFailure.notConfigured` and does not attempt a network request.

### Activity controller

The Activity controller test was introduced before `AttentionEventsController` existed. Final coverage verifies:

- server order preservation;
- explicit empty state;
- 401 expires session;
- 403 keeps session active;
- not-configured state is unavailable;
- offline, timeout and server failure are not collapsed into one misleading state;
- canonical subject ID is forwarded unchanged for subject-scoped activity.

### Event Detail controller

Lifecycle tests were introduced before `AttentionEventDetailController` existed. Final coverage verifies:

- OPEN acknowledge uses the canonical server response;
- ACKNOWLEDGED resolve uses the canonical server response;
- RESOLVED and UNKNOWN cannot issue unsafe lifecycle mutations;
- duplicate taps produce only one in-flight mutation;
- 401 expires the session;
- 403 preserves the last canonical event and does not expire the session;
- 404 is explicit unavailable;
- 409 fetches the current server event and preserves the server actor/time;
- offline mutation leaves the previous canonical lifecycle state unchanged.

### Activity and Event Detail UI

Widget tests were introduced before the new screens existed. During the GREEN cycle, one real Activity control-flow bug was found: `AsyncDataState.empty` carries no data, so the screen initially entered the generic unavailable branch. The production screen was corrected to handle `empty` before nullable data. Other initial widget failures were traced to lazily built `ListView` children below the test viewport; tests were updated to scroll to those controls without weakening lifecycle assertions.

Final widget coverage verifies:

- lifecycle grouping including UNKNOWN;
- server order within a lifecycle group;
- exact event identity on navigation;
- explicit unavailable/no-legacy-fallback state;
- explicit empty server activity state;
- canonical provenance fields;
- Acknowledge only for OPEN;
- Resolve only for ACKNOWLEDGED;
- no mutation action for RESOLVED/UNKNOWN;
- disabled action while mutation is in flight;
- exact canonical subject ID for Patient Overview navigation.

## Focused blocking verification

GitHub Actions run `35061243450`, job `104681825787`, checked out exact implementation head:

```text
f779205a0e82ce688d29922da916945865b22f9d
```

Focused P0-P4 test command result:

```text
🎉 143 tests passed.
```

Focused analyzer result:

```text
Analyzing 32 items...
No issues found! (ran in 9.5s)
```

Authority and forbidden-pattern checks completed successfully. These checks reject:

- retired `/v1/subjects/attach` compatibility;
- `_raiseIfEscalated` local event authority;
- local `AlertBandX.fromScore` authority in server-backed clinician views;
- direct model `/predict` calls from Patient Overview/Attention Event flows;
- guessed clinician/assessment/AttentionEvent target routes;
- legacy `roster.unacknowledgedCount` as the P4 Activity count.

The Dart authority suite also verifies no `ClinicalAlert`/`RosterController` authority in Activity, no device-time or SecureStore actor fabrication in lifecycle state, and no direct model inference from the Attention Event screens.

## Whole-repository visibility

Whole-repository Flutter test result:

```text
169 tests passed, 1 failed.
```

The single failure is the known pre-existing baseline expectation:

```text
test/widget_test.dart: boots to sign-in when there is no session
Expected: exactly one "Sign in"
Actual: 0 widgets with text "Sign in"
Location: test/widget_test.dart:130
```

This is the existing consent-first boot-flow expectation drift: when consent is not complete, the app shows the consent gate before sign-in. P4 does not change boot/consent behavior. The failure is not counted as a P4 pass and is not hidden by this verification document.

Whole-repository analyzer result:

```text
Analyzing tcwpn_mobile_app...
No issues found! (ran in 7.8s)
```

The workflow marks whole-suite visibility as `continue-on-error`; therefore the GitHub job conclusion alone is not used as evidence that the whole suite passed. The underlying test output above is the authoritative result.

## Branch diff review

Comparison against base `1477c8d665ecf4de5be9ba05428b1b972a893ef3` at verified implementation head `f779205a0e82ce688d29922da916945865b22f9d` reports:

```text
status: ahead
ahead_by: 30
behind_by: 0
```

Changed files are confined to the approved P4 scope:

- P4 design/implementation plan;
- clinician integration CI;
- AttentionEvent repository boundary and contract-gated adapter;
- Activity/Event Detail controllers;
- Activity/Event Detail screens;
- Dashboard event-detail navigation;
- shell migration from local Alerts to server Activity;
- minimal typed-state extension needed to preserve canonical event data on a forbidden mutation;
- P4 tests and authority guards.

No fusion/model implementation, research method, patient app, teammate backend, CARE-AnxRAG repository, push notification delivery, or P5 deep clinical view was changed.

## Final authority statement

P4 does **not** make ClinAnx an escalation engine. ClinAnx renders and requests transitions on a server-owned `AttentionEvent`. Event creation, assignment filtering, policy evaluation, deduplication/hysteresis/cooldown, final lifecycle state, clinician actor identity, timestamps and concurrency remain backend responsibilities.

Until the real AttentionEvent API exists and is verified, production P4 event transport remains visibly unavailable/not configured rather than fabricating a live integration.
