# ClinAnx P1/P2 Server-Backed Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local clinical-note-derived dashboard with an authenticated, assignment-scoped, server-backed clinician worklist that displays backend-created OPEN `AttentionEvent`s first and backend-provided `PatientSummary` records second, while preserving the P0 safety contracts and removing the remaining temporary identity/local-alert compatibility paths.

**Architecture:** Keep the Central Backend as the sole authoritative clinical integration boundary. Reuse `ApiClient`, `Session`, `SecureStore`, `AuthService`, Provider, and the P0 contract models. Add small repository interfaces between state and HTTP, introduce typed dashboard state, cache only the last server-derived snapshot for offline continuity, and keep current assessment, physiological forecast, and attention-event identity separate from each other. No client-side fusion, risk re-ranking, or urgent-event generation is allowed.

**Tech Stack:** Flutter >=3.27.0, Dart >=3.5.0 <4.0.0, `provider`, `http`, `shared_preferences`, `flutter_secure_storage`, `flutter_test`.

**Spec:** `docs/superpowers/specs/2026-09-15-clinanx-p1-p2-server-dashboard-design.md`

## Global Constraints

- Modify only `dulhara79/tcwpn_mobile_app`.
- Treat teammate repositories as read-only. If a backend contract mismatch is discovered, stop live wiring and draft a precise message for Uvindu instead of modifying his repository.
- Central Backend owns authoritative current assessment, forecast persistence, assignment enforcement, and `AttentionEvent` lifecycle.
- ClinAnx must never compute an authoritative fusion/composite score locally.
- Current assessment and forecast remain separate domain objects and separate UI concepts.
- Forecast wording remains physiological/C1-led unless the verified backend and research method explicitly establish a multimodal forecast.
- Missing, stale, unsupported, excluded, or unknown values never become zero, low, green, or reassuring defaults.
- C2 remains experimental / excluded unless the verified server contract says otherwise.
- TC-WPN/C3 remains a contributing modality, not the patient-level risk result.
- Server event IDs must be preserved; the mobile app must not mint a replacement urgent-event ID.
- The old local `ClinicalAlert` records remain readable until the later P4 migration, but this slice must stop creating new authoritative risk-escalation alerts locally.
- Do not invent endpoint names. Task 1 is a hard verification gate.

## Target route assumptions to verify before coding

The two approved project documents describe the following target operations and suggested paths. This plan may use these paths only after Task 1 verifies that Uvindu's implemented backend exposes the same semantics. If his final route names differ, stop and revise the route literals in this plan before implementation rather than translating them ad hoc.

```text
GET /v1/me
GET /v1/clinicians/me/dashboard
GET /v1/clinicians/me/patients
GET /v1/patients/{subject_id}/assessment/latest
GET /v1/attention-events?status=OPEN
```

The implementation only needs one assignment-scoped source for the dashboard patient list. Prefer `/v1/clinicians/me/dashboard` if it returns both `patient_summaries` and `open_attention_events`; otherwise use the verified `/v1/clinicians/me/patients` plus `/v1/attention-events?status=OPEN` pair. Do not make both paths authoritative simultaneously.

---

### Task 1: Verify and freeze the actual backend contract

**Files:**
- Read only: `UVINDUSEN/component4final` route/OpenAPI/model files
- Read only: deployed/open API description if Uvindu provides it
- No ClinAnx production file changes in this task

**Purpose:** Prove that the live integration semantics exist before adding route literals to ClinAnx.

- [ ] **Step 1: Verify clinician identity semantics**

Inspect the actual backend implementation and confirm all of these are true:

```text
Bearer clinician credential accepted
GET /v1/me or verified equivalent exists
401 = invalid/expired identity
403 = authenticated but not authorized
actor identity is derived from authenticated clinician principal
```

Do not accept an `author` string supplied by Flutter as a substitute for authenticated actor identity for assignment/event authorization.

- [ ] **Step 2: Verify assignment-scoped dashboard semantics**

Confirm that the actual backend has either:

```text
GET /v1/clinicians/me/dashboard
```

returning the authenticated clinician's patient summaries and OPEN events, or a verified equivalent split across assignment-scoped patient and event endpoints.

The response must not require the client to submit a clinician ID to choose whose patients are returned.

- [ ] **Step 3: Verify target objects**

Confirm representative responses preserve these fields/semantics:

```text
PatientSummary:
  subject_id
  display_id or safe display label
  fusion_result_id
  current/current_assessment
  forecast (separate)
  assessment_status
  last_updated
  open_event_count

AttentionEvent:
  id
  subject_id
  fusion_result_id
  forecast_result_id optional
  event_type
  severity
  reason
  forecast_horizon optional
  status
  created_at
  acknowledged/resolved actor+time fields

AssessmentSummary:
  subject_id
  fusion_result_id
  current_assessment
  forecast
  confidence and/or uncertainty with unambiguous semantics
  assessment_status
  modalities
  computed_at
  model_version
```

- [ ] **Step 4: Hard stop on mismatch**

If any required semantic is missing, do not add a fake Flutter endpoint. Prepare a message to Uvindu containing the observed route/shape, the missing semantic, and acceptance criteria. Continue only fixture/repository work that does not claim live integration.

- [ ] **Step 5: Record verification in the implementation commit notes**

When execution begins, include the verified route names and response envelope keys in the Task 3 commit message/body or PR notes. No separate guessed contract file is required.

---

### Task 2: Correct transport semantics and P0 contract ambiguity

**Files:**
- Modify: `lib/data/api/api_client.dart`
- Modify: `lib/domain/contracts/assessment_summary.dart`
- Modify: `test/gateway_contract_test.dart`
- Modify: `test/contracts/assessment_summary_test.dart`

**Interfaces:**
- `ApiFailure` gains `forbidden` and `conflict`.
- `AssessmentSummary` represents `confidence` and `uncertainty` separately instead of treating one as the other.

- [ ] **Step 1: Write failing API status tests**

Update `test/gateway_contract_test.dart` so status mapping asserts:

```dart
expect(await failureForStatus(401), ApiFailure.unauthorized);
expect(await failureForStatus(403), ApiFailure.forbidden);
expect(await failureForStatus(409), ApiFailure.conflict);
expect(await failureForStatus(422), ApiFailure.validation);
```

Also assert the 403 clinician-facing message does not say the session expired.

- [ ] **Step 2: Run the focused gateway test and verify RED**

```bash
flutter test test/gateway_contract_test.dart
```

Expected: FAIL because `forbidden` and `conflict` do not yet exist and 403 currently maps to `unauthorized`.

- [ ] **Step 3: Implement the minimal status mapping**

In `lib/data/api/api_client.dart`, extend `ApiFailure` and the status switch:

```dart
enum ApiFailure {
  offline,
  timeout,
  unauthorized,
  forbidden,
  notFound,
  conflict,
  validation,
  server,
  malformed,
  notConfigured,
  insecureConnection,
  unknown,
}
```

```dart
kind: switch (res.statusCode) {
  401 => ApiFailure.unauthorized,
  403 => ApiFailure.forbidden,
  404 => ApiFailure.notFound,
  409 => ApiFailure.conflict,
  400 || 422 => ApiFailure.validation,
  >= 500 => ApiFailure.server,
  _ => ApiFailure.unknown,
},
```

Add user-safe messages:

```dart
ApiFailure.forbidden =>
  'You are signed in, but you do not have permission to access this patient or action.',
ApiFailure.conflict =>
  'This record changed on the server. Refresh to see the current state before trying again.',
```

- [ ] **Step 4: Write the contract-semantics test first**

Extend `test/contracts/assessment_summary_test.dart` with an input containing both fields:

```dart
final json = loadContractFixture('assessment_complete.json');
json['confidence'] = 0.71;
json['uncertainty'] = 0.19;
final result = AssessmentSummary.fromJson(json);
expect(result.confidence, 0.71);
expect(result.uncertainty, 0.19);
```

- [ ] **Step 5: Run and verify RED**

```bash
flutter test test/contracts/assessment_summary_test.dart
```

Expected: FAIL because `uncertainty` is not represented independently.

- [ ] **Step 6: Separate the fields**

Change `AssessmentSummary` to:

```dart
final double? confidence;
final double? uncertainty;
```

and parse them independently:

```dart
confidence: contractDouble(json['confidence']),
uncertainty: contractDouble(json['uncertainty']),
```

Do not derive one from the other.

- [ ] **Step 7: Run GREEN verification**

```bash
flutter test test/gateway_contract_test.dart test/contracts/assessment_summary_test.dart
flutter analyze lib/data/api/api_client.dart lib/domain/contracts/assessment_summary.dart test/gateway_contract_test.dart test/contracts/assessment_summary_test.dart
```

Expected: PASS with no analyzer errors.

- [ ] **Step 8: Commit**

```bash
git add lib/data/api/api_client.dart lib/domain/contracts/assessment_summary.dart \
  test/gateway_contract_test.dart test/contracts/assessment_summary_test.dart
git commit -m "fix: preserve clinician auth and assessment semantics"
```

---

### Task 3: Add clinician principal and authenticated Central Backend read operations

**Files:**
- Create: `lib/domain/contracts/clinician_principal.dart`
- Modify: `lib/data/api/gateways.dart`
- Modify: `lib/core/config/env.dart`
- Create: `test/contracts/clinician_principal_test.dart`
- Modify: `test/gateway_contract_test.dart`

**Interfaces:**

```dart
class ClinicianPrincipal {
  final String clinicianId;
  final String? displayName;
  final String? role;
  final DateTime? expiresAt;
}
```

After Task 1 verifies the target route names, `CentralBackendGateway` exposes:

```dart
Future<ClinicianPrincipal> me();
Future<List<PatientSummary>> assignedPatients();
Future<List<AttentionEvent>> openAttentionEvents();
Future<AssessmentSummary?> latestAssessment(String subjectId);
```

- [ ] **Step 1: Add failing principal-parser tests**

Use a conservative parser accepting either `clinician_id` or JWT-style `sub` as the principal identifier, but never inventing one:

```dart
test('clinician principal preserves authenticated identity', () {
  final p = ClinicianPrincipal.fromJson({
    'clinician_id': 'DR001',
    'display_name': 'Dr D. Kaushalya',
    'role': 'clinician',
    'exp': '2026-09-16T10:00:00Z',
  });
  expect(p.clinicianId, 'DR001');
  expect(p.role, 'clinician');
});
```

- [ ] **Step 2: Add failing gateway auth tests**

In `test/gateway_contract_test.dart`, seed:

```dart
Session.set(token: 'clinician-jwt', clinicianId: 'DR001');
```

Inject a mock HTTP client and assert the request to `/v1/me` carries:

```text
Authorization: Bearer clinician-jwt
```

Also assert the Central Backend gateway no longer substitutes `Env.backendToken` for clinician-scoped reads.

- [ ] **Step 3: Run RED**

```bash
flutter test test/contracts/clinician_principal_test.dart test/gateway_contract_test.dart
```

Expected: FAIL because the principal contract/read methods do not exist and the gateway currently overrides auth with `Env.backendToken`.

- [ ] **Step 4: Implement `ClinicianPrincipal`**

Create a parser using the existing P0 parsing helpers. Missing identity remains an empty/invalid principal that repository code rejects; do not synthesize `unknown-doctor`.

- [ ] **Step 5: Switch Central Backend default bearer source**

Change the gateway constructor from the shared build credential override to the default `ApiClient` session bearer:

```dart
CentralBackendGateway([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);
```

Update misleading comments in `env.dart`: `BACKEND_TOKEN` is no longer the clinician identity path. If the verified backend still needs a separate service credential for any non-clinician operation, isolate that later in a separately named client; do not silently send it as the doctor credential.

- [ ] **Step 6: Add verified read methods**

If Task 1 confirms the documented target paths exactly, implement:

```dart
Future<ClinicianPrincipal> me() async =>
    ClinicianPrincipal.fromJson(await _api.get('/v1/me'));

Future<List<PatientSummary>> assignedPatients() async {
  final json = await _api.get('/v1/clinicians/me/patients');
  final rows = contractMapList(json['patients'] ?? json['patient_summaries']);
  return rows.map(PatientSummary.fromJson).toList(growable: false);
}

Future<List<AttentionEvent>> openAttentionEvents() async {
  final json = await _api.get('/v1/attention-events?status=OPEN');
  final rows = contractMapList(json['events'] ?? json['attention_events']);
  return rows.map(AttentionEvent.fromJson).toList(growable: false);
}
```

For latest assessment:

```dart
Future<AssessmentSummary?> latestAssessment(String subjectId) async {
  try {
    final json = await _api.get(
      '/v1/patients/${Uri.encodeComponent(subjectId)}/assessment/latest',
    );
    return AssessmentSummary.fromJson(json);
  } on ApiException catch (e) {
    if (e.kind == ApiFailure.notFound) return null;
    rethrow;
  }
}
```

If Task 1 verifies a combined dashboard endpoint instead, use that exact envelope in the gateway and adapt the repository task accordingly; revise this plan before coding rather than supporting two competing authoritative paths.

- [ ] **Step 7: Run GREEN**

```bash
flutter test test/contracts/clinician_principal_test.dart test/gateway_contract_test.dart
flutter analyze lib/domain/contracts/clinician_principal.dart lib/data/api/gateways.dart lib/core/config/env.dart test/contracts/clinician_principal_test.dart test/gateway_contract_test.dart
```

- [ ] **Step 8: Commit**

```bash
git add lib/domain/contracts/clinician_principal.dart lib/data/api/gateways.dart \
  lib/core/config/env.dart test/contracts/clinician_principal_test.dart test/gateway_contract_test.dart
git commit -m "feat: add clinician-scoped backend reads"
```

---

### Task 4: Add repository boundaries and server-derived cache

**Files:**
- Create: `lib/domain/repositories/auth_repository.dart`
- Create: `lib/domain/repositories/patient_repository.dart`
- Create: `lib/domain/repositories/assessment_repository.dart`
- Create: `lib/domain/repositories/attention_event_repository.dart`
- Create: `lib/data/repositories/central_backend_repositories.dart`
- Create: `lib/domain/contracts/dashboard_snapshot.dart`
- Create: `lib/data/local/dashboard_cache.dart`
- Create: `test/repositories/central_backend_repositories_test.dart`
- Create: `test/data/dashboard_cache_test.dart`

**Repository interfaces:**

```dart
abstract interface class AuthRepository {
  Future<ClinicianPrincipal> currentClinician();
  Future<void> expireSession();
}

abstract interface class PatientRepository {
  Future<List<PatientSummary>> assignedPatients();
}

abstract interface class AssessmentRepository {
  Future<AssessmentSummary?> latestAssessment(String subjectId);
}

abstract interface class AttentionEventRepository {
  Future<List<AttentionEvent>> openEvents();
}
```

- [ ] **Step 1: Write repository tests first**

Use fake/injected `CentralBackendGateway` or injected functions so tests prove:

```text
only gateway-provided assigned patients are returned
open event IDs are preserved exactly
current assessment and forecast remain separate
unknown/null values remain conservative
AuthRepository.expireSession clears Session + SecureStore
```

- [ ] **Step 2: Run RED**

```bash
flutter test test/repositories/central_backend_repositories_test.dart
```

Expected: FAIL because repository interfaces/implementations do not exist.

- [ ] **Step 3: Implement thin repositories**

Repositories should delegate to the verified gateway and contain no risk logic. `AssessmentRepository` is included now because P3 will use the same boundary; it should not trigger N+1 assessment reads from the P2 Dashboard unless the verified dashboard contract lacks needed summary fields.

- [ ] **Step 4: Add a cacheable `DashboardSnapshot`**

Define:

```dart
class DashboardSnapshot {
  final ClinicianPrincipal? clinician;
  final List<AttentionEvent> openEvents;
  final List<PatientSummary> assignedPatients;
  final DateTime fetchedAt;
  final bool fromCache;
}
```

Add explicit `toJson`/`fromJson` for snapshot persistence. Add corresponding conservative `toJson` methods to `PatientSummary`, `CurrentAssessment`, `ForecastResult`, and `AttentionEvent` as needed. Round-tripping must preserve null scores and server IDs exactly.

- [ ] **Step 5: Write cache round-trip tests before implementation**

Prove:

```text
null current score remains null
forecast remains a separate object
server event id remains evt-001
fusion_result_id remains 123
fetchedAt survives persistence
```

- [ ] **Step 6: Implement `DashboardCacheStore`**

Use one SharedPreferences key such as `server_dashboard_v1` and store only the serialized server-derived snapshot. Do not store a recalculated risk summary.

- [ ] **Step 7: Run GREEN**

```bash
flutter test test/repositories/central_backend_repositories_test.dart test/data/dashboard_cache_test.dart
flutter analyze lib/domain/repositories lib/data/repositories lib/domain/contracts/dashboard_snapshot.dart lib/data/local/dashboard_cache.dart test/repositories test/data/dashboard_cache_test.dart
```

- [ ] **Step 8: Commit**

```bash
git add lib/domain/repositories lib/data/repositories \
  lib/domain/contracts/dashboard_snapshot.dart lib/data/local/dashboard_cache.dart \
  test/repositories test/data/dashboard_cache_test.dart
git commit -m "feat: add server dashboard repositories and cache"
```

---

### Task 5: Add typed async state and DashboardController

**Files:**
- Create: `lib/state/async_data_state.dart`
- Create: `lib/state/dashboard_controller.dart`
- Create: `test/state/dashboard_controller_test.dart`

**State shape:**

```dart
enum AsyncPhase {
  loading,
  data,
  empty,
  partial,
  unavailable,
  offline,
  error,
  sessionExpired,
  forbidden,
}

class AsyncDataState<T> {
  final AsyncPhase phase;
  final T? data;
  final String? message;
}
```

`DashboardController` receives repository interfaces and `DashboardCacheStore` through its constructor so tests need no network or platform secrets.

- [ ] **Step 1: Write controller tests first**

Cover these cases independently:

```text
success -> data snapshot
empty patient/event lists -> empty
patient success + event failure -> partial
401 -> Session expired, cache not presented as live
403 -> forbidden and valid Session remains active
offline/timeout/server error + cache -> offline(snapshot)
offline/timeout/server error without cache -> error/unavailable
server data save -> cache updated with fetchedAt
```

Also prove OPEN events are not re-sorted using a locally calculated risk formula; preserve server order unless the verified contract explicitly requires another order.

- [ ] **Step 2: Run RED**

```bash
flutter test test/state/dashboard_controller_test.dart
```

- [ ] **Step 3: Implement minimal `AsyncDataState<T>`**

Use named constructors or const factories for each phase. Avoid an impossible state such as `data` phase with `data == null`.

- [ ] **Step 4: Implement `DashboardController.load()`**

Required sequence:

```text
1. state = loading (unless silent refresh with existing data)
2. AuthRepository.currentClinician()
3. fetch assignedPatients + openEvents concurrently
4. build server-derived DashboardSnapshot
5. cache successful snapshot
6. publish data/empty/partial
7. map auth/network failures conservatively
```

For 401:

```dart
await authRepository.expireSession();
state = AsyncDataState.sessionExpired(...);
```

For 403, do not call `expireSession()`.

- [ ] **Step 5: Run GREEN**

```bash
flutter test test/state/dashboard_controller_test.dart
flutter analyze lib/state/async_data_state.dart lib/state/dashboard_controller.dart test/state/dashboard_controller_test.dart
```

- [ ] **Step 6: Commit**

```bash
git add lib/state/async_data_state.dart lib/state/dashboard_controller.dart test/state/dashboard_controller_test.dart
git commit -m "feat: add server dashboard state controller"
```

---

### Task 6: Replace the local KPI dashboard with the server worklist

**Files:**
- Create: `lib/features/dashboard/server_dashboard_screen.dart`
- Delete after migration: `lib/features/dashboard/kpi_dashboard_screen.dart`
- Replace: `test/clinical_dashboard_test.dart` with `test/features/dashboard/server_dashboard_screen_test.dart`
- Modify: `lib/main.dart` or composition root used by `AppShell` to provide `DashboardController`

**UI contract:**

```text
Dashboard
  clinician identity / last refresh
  offline/partial/error notice when applicable

Needs attention
  OPEN AttentionEvent cards

Assigned patients
  PatientSummary rows
```

- [ ] **Step 1: Write widget tests before the screen**

Use an injected/fake `DashboardController` snapshot containing `evt-001` and `subject-001`. Test:

```text
Needs attention appears before Assigned patients
evt-001 data is rendered from server event fields
Patient A comes from PatientSummary
fusion result identity is retained in the model
Medium current assessment and High physiological forecast are not merged
unavailable current assessment displays "Assessment unavailable" or equivalent, never Low/0
partial displays Partial assessment
physiological forecast is labelled as near-term physiological forecast, not multimodal
no open events renders a neutral empty event section, not "patient cleared"
offline cached snapshot shows an explicit stale/offline notice
```

- [ ] **Step 2: Add modality-state widget coverage only when the verified summary contains modality metadata**

If Task 1 verifies modality statuses are present in Dashboard patient summaries, test and render:

```text
C1 stale -> "Physiological signal stale"
C2 not_validated/excluded -> "Experimental — not included in fusion"
C3 unavailable -> "Clinical NLP unavailable"
```

If the verified Dashboard summary intentionally omits modality details, do not fabricate them in P2. Keep those assertions in `AssessmentSummary` contract tests and render them in P3 Patient Overview instead. Record this distinction in PR notes.

- [ ] **Step 3: Run RED**

```bash
flutter test test/features/dashboard/server_dashboard_screen_test.dart
```

Expected: FAIL because the server Dashboard screen does not exist.

- [ ] **Step 4: Implement the worklist**

Use existing design-system primitives (`Panel`, `InlineNotice`, typography/tokens). Do not import `RecordStore`, `ClinicalNote`, `TcwpnResult`, or call `AlertBandX.fromScore` in the new Dashboard.

Use only:

```text
DashboardController
DashboardSnapshot
PatientSummary
AttentionEvent
AssessmentStatus
RiskTier
ForecastScope
```

The Dashboard does not acknowledge/resolve events in this slice.

- [ ] **Step 5: Remove old local KPI code/tests**

Delete `kpi_dashboard_screen.dart` after all shell imports move to `ServerDashboardScreen`. Remove the old tests that assert local note scores drive caseload risk/trajectory; those behaviors are intentionally obsolete.

- [ ] **Step 6: Run GREEN**

```bash
flutter test test/features/dashboard/server_dashboard_screen_test.dart
flutter analyze lib/features/dashboard lib/state/dashboard_controller.dart
```

- [ ] **Step 7: Commit**

```bash
git add lib/features/dashboard lib/main.dart test/features/dashboard test/clinical_dashboard_test.dart
git commit -m "feat: replace local dashboard with server worklist"
```

---

### Task 7: Make the server Dashboard the primary shell destination

**Files:**
- Modify: `lib/features/shell.dart`
- Modify: `test/shell_navigation_structure_test.dart`

- [ ] **Step 1: Update the structural test first**

Assert the visible shell contains exactly one server Dashboard destination and does not register `KpiDashboardScreen`:

```dart
expect(indexedStack, contains('ServerDashboardScreen()'));
expect(indexedStack, isNot(contains('KpiDashboardScreen()')));
```

Also assert Dashboard is the first `IndexedStack` child / first navigation destination.

- [ ] **Step 2: Run RED**

```bash
flutter test test/shell_navigation_structure_test.dart
```

- [ ] **Step 3: Update shell composition**

Replace the existing local `_CaseloadTab` first destination with `ServerDashboardScreen` and remove the duplicate old Dashboard destination at the end.

Keep the existing Patients, legacy Alerts, Settings, and Ask CARE destinations in this slice. The Alerts screen remains legacy/read-only authoritative-wise until P4; do not expand this task into event ACK/RESOLVE.

- [ ] **Step 4: Run GREEN**

```bash
flutter test test/shell_navigation_structure_test.dart test/features/dashboard/server_dashboard_screen_test.dart
flutter analyze lib/features/shell.dart test/shell_navigation_structure_test.dart
```

- [ ] **Step 5: Commit**

```bash
git add lib/features/shell.dart test/shell_navigation_structure_test.dart
git commit -m "refactor: make server dashboard the primary worklist"
```

---

### Task 8: Finish P0 identity cleanup and stop local authoritative alert creation

**Files:**
- Modify: `lib/state/controllers.dart`
- Modify: `lib/data/api/gateways.dart`
- Modify: `test/contracts/identity_resolution_test.dart`
- Create: `test/state/no_local_attention_event_test.dart`

- [ ] **Step 1: Update identity tests first**

Remove the test that legitimizes the deprecated `attach()` compatibility shim. Replace it with regression checks that:

```text
resolveAppUserId uses /v1/subjects/resolve?app_user_id=
no /v1/subjects/attach literal exists in active gateway/controller source
ChartController uses resolveAppUserId directly for P_[A-F0-9]{16}
```

- [ ] **Step 2: Add the local-alert regression test first**

Use a source-structure test or injected controller seam to prove the active refresh path no longer calls `_raiseIfEscalated` and no longer creates a UUID `ClinicalAlert` merely because fusion band is RED/DARK RED.

The test must not delete or invalidate already persisted legacy alerts.

- [ ] **Step 3: Run RED**

```bash
flutter test test/contracts/identity_resolution_test.dart test/state/no_local_attention_event_test.dart
```

- [ ] **Step 4: Remove the compatibility shim**

In `ChartController.ensureEnrolled()`, change the Aura participant path to:

```dart
final resolvedByAppId = await _backend.resolveAppUserId(
  mrn.trim().toUpperCase(),
);
```

Persist the returned subject id as before. Remove `CentralBackendGateway.attach()` entirely when no call sites remain.

- [ ] **Step 5: Remove new local escalation generation**

From `refreshFusion()`, remove:

```dart
await _raiseIfEscalated(state);
```

and remove `_raiseIfEscalated()` itself if unused.

Do not remove `RecordStore.loadAlerts/saveAlerts`, `ClinicalAlert`, or `AlertsScreen` yet; P4 owns the complete legacy-alert migration.

- [ ] **Step 6: Run GREEN**

```bash
flutter test test/contracts/identity_resolution_test.dart test/state/no_local_attention_event_test.dart test/gateway_contract_test.dart
flutter analyze lib/state/controllers.dart lib/data/api/gateways.dart test/contracts/identity_resolution_test.dart test/state/no_local_attention_event_test.dart
```

- [ ] **Step 7: Commit**

```bash
git add lib/state/controllers.dart lib/data/api/gateways.dart \
  test/contracts/identity_resolution_test.dart test/state/no_local_attention_event_test.dart
git commit -m "fix: retire local escalation and attach compatibility paths"
```

---

### Task 9: Add P1/P2 CI gates and verify the full slice

**Files:**
- Create: `.github/workflows/p1_p2_dashboard_ci.yml`
- No application behavior changes unless verification finds a regression

- [ ] **Step 1: Add a branch-scoped focused CI workflow**

Configure triggers for:

```yaml
on:
  push:
    branches: [integration/clinanx-p1-p2-server-dashboard]
  pull_request:
    branches: [main]
```

Blocking focused commands:

```bash
flutter pub get
flutter test test/contracts test/repositories test/data test/state \
  test/features/dashboard test/gateway_contract_test.dart \
  test/session_sign_out_test.dart test/shell_navigation_structure_test.dart

flutter analyze lib/domain/contracts lib/domain/repositories \
  lib/data/api lib/data/repositories lib/data/local/dashboard_cache.dart \
  lib/state lib/features/dashboard test/contracts test/repositories \
  test/data test/state test/features/dashboard test/gateway_contract_test.dart \
  test/session_sign_out_test.dart test/shell_navigation_structure_test.dart
```

Run full-suite visibility separately. If known pre-existing failures remain, use `continue-on-error: true` only for the full regression/full analyzer steps and describe the exact failures in PR notes. Never claim whole-suite success from a green job if those steps are non-blocking.

- [ ] **Step 2: Run focused verification locally or through CI**

```bash
flutter test test/contracts test/repositories test/data test/state test/features/dashboard test/gateway_contract_test.dart test/session_sign_out_test.dart test/shell_navigation_structure_test.dart
flutter analyze lib/domain/contracts lib/domain/repositories lib/data/api lib/data/repositories lib/data/local/dashboard_cache.dart lib/state lib/features/dashboard test/contracts test/repositories test/data test/state test/features/dashboard test/gateway_contract_test.dart test/session_sign_out_test.dart test/shell_navigation_structure_test.dart
```

Record the exact output before claiming success.

- [ ] **Step 3: Run full repository checks**

```bash
flutter test
flutter analyze
```

Record exact failures/warnings if any. Distinguish pre-existing baseline noise from branch regressions.

- [ ] **Step 4: Run forbidden-pattern checks**

```bash
rg -n "/v1/subjects/attach|_raiseIfEscalated|KpiDashboardScreen|ClinicalDashboardStats" lib test
rg -n "AlertBandX\.fromScore" lib/features/dashboard
rg -n "(/predict|TcwpnGateway|C3Gateway)" lib
rg -n "Env\.backendToken" lib/data/api/gateways.dart
```

Expected for active production paths:

```text
/v1/subjects/attach -> no active matches
_raiseIfEscalated -> no active matches
KpiDashboardScreen / ClinicalDashboardStats -> no active matches
AlertBandX.fromScore in server Dashboard -> no matches
direct TC-WPN /predict client -> no authoritative workflow matches
Env.backendToken in CentralBackendGateway -> no matches
```

Backup/archive files are not acceptable evidence of active behavior, but if they trigger `rg`, scope the final check to active `.dart` files and document that distinction.

- [ ] **Step 5: Review branch diff against main**

```bash
git diff --stat main...HEAD
git diff main...HEAD
```

Confirm every changed production file is required by the approved P1/P2 slice and no teammate repository or unrelated feature was touched.

- [ ] **Step 6: Commit CI**

```bash
git add .github/workflows/p1_p2_dashboard_ci.yml
git commit -m "ci: gate ClinAnx P1 P2 server dashboard"
```

- [ ] **Step 7: Do not open the PR until verification is reviewed**

Before creating the PR, summarize:

```text
verified backend routes actually used
focused test result
focused analyzer result
full test result
full analyzer result
known pre-existing failures if any
forbidden-pattern check result
branch-vs-main file list
```

Only after that review should the branch be opened as a PR to `main`.

---

## PR scope after successful verification

Suggested title:

```text
ClinAnx P1/P2: server-backed clinician dashboard
```

Suggested PR summary points:

```text
- use clinician session identity for assignment-scoped backend reads
- distinguish 401, 403 and 409 transport semantics
- add repository boundaries and conservative dashboard cache
- replace local clinical-note KPI dashboard with server worklist
- show backend OPEN AttentionEvents before assigned PatientSummary rows
- preserve current assessment vs physiological forecast separation
- remove deprecated Aura attach shim
- stop minting new local authoritative escalation alerts
- retain legacy alert records/screens until P4 migration
```

Backend dependencies must list the exact verified Uvindu routes rather than the document's suggested examples.

## Follow-on work explicitly excluded from this plan

```text
P3 Patient Overview
P4 Attention Event detail + ACK/RESOLVE + Activity migration
P5 history / contributions / evidence / data quality
P6 push notifications
P7 security/offline/accessibility hardening
P8 research release
```
