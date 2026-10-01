# ClinAnx Phase 7 Hardening Design

**Status:** Approved 17 Sep 2026

## Goal

Harden ClinAnx without changing its clinical authority boundary: the Central Backend remains authoritative for assignments, current assessment, forecast, AttentionEvent lifecycle, actor identity, timestamps, and multimodal fusion. ClinAnx owns presentation, authenticated interaction, local drafts/cache, and safe degraded behavior.

## Scope

Phase 7 covers the ClinAnx technical-specification hardening scope: offline/stale behavior, accessibility, privacy review, integration/failure tests, configuration/security truthfulness, CI release gates, and a repeatable usability checklist.

## Non-goals

- No new authoritative clinical calculation on device.
- No invented Central Backend dashboard/latest-assessment routes.
- No changes to `UVINDUSEN/component4final`, `DewduSendanayake/anxiety_mobile_app`, `UVINDUSEN/Care-AnxRAG`, or TC-WPN serving repositories.
- No WebSocket/FCM dependency for completion; polling remains an allowed research-prototype fallback.
- No advanced offline conflict engine.
- No fabricated TLS pins for unstable/unverified hosts.

## Hardening decisions

### 1. Clinician-scoped local privacy

Local clinical caches must be isolated by authenticated clinician identity. A clinician signing out and another clinician signing in on the same device must not expose the previous clinician's roster, dashboard cache, patient notes/support/fusion cache, legacy alerts, delivered-notification bookkeeping, or pending notification-open event.

A single storage-scope abstraction will normalize clinician IDs and build namespaced SharedPreferences keys. Legacy unscoped records will not be silently migrated into the first clinician account because ownership cannot be proven.

`RosterController` must not initialize before an authenticated clinician scope exists.

### 2. Offline/stale safety

Cached server results are display-only fallbacks and must remain visibly stale/offline. Network failure, backend failure, malformed responses, or missing modalities must never create a Low/Green/0 safe state.

401 means session-expired flow. 403 means authenticated but forbidden and does not clear a valid session. 409 means server state changed and the client reloads/reconciles canonical server state where the workflow supports mutation.

### 3. Accessibility

Clinical state is communicated by text/labels/icons, not color alone. Remove the global 1.3x text-scale cap and test critical screens under enlarged text.

### 4. Configuration/security truthfulness

`Env` is the sole build-time configuration source. `AuthService` consumes `Env` rather than declaring duplicate `String.fromEnvironment` values. `DEMO_DATA` defaults false; demo fixtures require explicit opt-in.

Settings must truthfully state that clinical notes are sent to the Central Backend, which orchestrates C3. It must not imply that all network hosts are certificate-pinned when only a subset is pinned. Research builds expose the app/config/backend/model information actually available; missing values remain unavailable rather than fabricated.

### 5. Failure-injection test matrix

Automated tests cover at least: C1 stale/unavailable representation, C3 unavailable, C4 missing, RAG timeout/abstention, backend unavailable, expired JWT, forbidden/unassigned access, duplicate event polling, concurrent mutation conflict behavior, malformed response, and network-loss cache behavior.

Where a failure depends on a backend capability not yet implemented, the mobile test uses repository/fixture doubles and is explicitly a client behavior test rather than a claim of live end-to-end verification.

### 6. CI release gate

Full `flutter test` and full `flutter analyze` become blocking. Phase 7 privacy, failure-safety, accessibility and authority tests are included in CI. Existing authority guards remain blocking.

### 7. Usability evidence

Add a repeatable research-prototype usability checklist covering sign-in, Needs Attention, current-vs-forecast interpretation, stale/unavailable modality interpretation, AttentionEvent lifecycle, evidence abstention/unavailability, and enlarged-text operation.

## External dependencies that remain outside ClinAnx

As of the hardening start recheck:

- ClinAnx `main`: `fc76fcea3c3276a398f5f4baefeb3cc735991c32`.
- Central Backend `main`: `a1ceef8daba268dac24d81aedd76c89e7c9ccc6a`; shared backend token remains the current backend auth model and target clinician assignment/AttentionEvent server support is external work.
- Patient app `main`: `2d7cb5936706c37ccbf462005b4e4b238bf3c1dd`; patient-side authoritative current-assessment/event alignment is external work.

Phase 7 must proceed against frozen client contracts/fixtures without claiming those external dependencies are live.