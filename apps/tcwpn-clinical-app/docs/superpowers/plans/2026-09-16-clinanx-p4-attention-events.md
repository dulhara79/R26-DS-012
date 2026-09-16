# ClinAnx P4 Server Attention Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the clinician-side persistent `AttentionEvent` workflow (`OPEN -> ACKNOWLEDGED -> RESOLVED`) with server-authoritative identity/state, Activity and Event Detail UI, safe concurrency reconciliation, and migration away from the legacy local-alert inbox as the target source of urgent clinician work.

**Architecture:** Extend the existing P0-P3 domain/repository/controller pattern. `AttentionEventRepository` is the only event data boundary; `AttentionEventsController` owns list/activity state and `AttentionEventDetailController` owns one event lifecycle. Production Central Backend event methods remain contract-gated with `ApiFailure.notConfigured` until the real backend routes/OpenAPI are verified, while tests use explicit fakes/fixtures.

**Tech Stack:** Flutter >=3.27.0, Dart >=3.5.0 <4.0.0, Provider/ChangeNotifier, existing `ApiClient`/`ApiFailure`, `flutter_test`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-clinanx-p4-attention-events-design.md`

## Global Constraints

- Central Backend owns event creation, assignment scope, event identity, lifecycle state, actors, timestamps, concurrency, and audit history.
- ClinAnx must not create authoritative acute-escalation events from fusion score/tier, TC-WPN score, local band, or UI color.
- `AttentionEvent.id`, `subject_id`, `fusion_result_id`, optional `forecast_result_id`, lifecycle status, actor fields, timestamps, reason, horizon, severity, event type, and policy version are server-provided values.
- `OPEN -> ACKNOWLEDGED -> RESOLVED` is sequential; no direct OPEN -> RESOLVED UI action.
- Mutation methods accept only `eventId`; the client does not submit final actor/timestamp/state fields.
- 401 expires the clinician session; 403 does not. 404 is explicit not-found/unavailable. 409 reloads canonical server state. Offline/timeout/server failures never fabricate a successful mutation.
- Production event transport remains `notConfigured` until the exact live backend contract is verified. No guessed `/attention-events`, `/acknowledge`, `/resolve`, or equivalent target route may be live-wired.
- Legacy `ClinicalAlert` storage may remain for backwards compatibility, but it must not drive the new Activity screen or bottom-navigation urgent count.
- No teammate-owned repository is modified.
- Preserve the known whole-suite consent/sign-in baseline failure truthfully if it still exists; focused P4 gates are blocking, whole-suite visibility must not be misreported as fully green.

---

## File Structure

**Create**
- `lib/state/attention_events_controller.dart` — server event activity/worklist state.
- `lib/state/attention_event_detail_controller.dart` — one event detail + lifecycle mutation state.
- `lib/features/attention_events/activity_screen.dart` — persistent Activity list UI.
- `lib/features/attention_events/attention_event_detail_screen.dart` — canonical event detail/actions/navigation.
- `test/attention_events_controller_test.dart`
- `test/attention_event_detail_controller_test.dart`
- `test/features/attention_events/activity_screen_test.dart`
- `test/features/attention_events/attention_event_detail_screen_test.dart`
- `test/p4_attention_event_authority_test.dart`
- `docs/superpowers/verification/2026-09-16-clinanx-p4-attention-events-verification.md` after verification.

**Modify**
- `lib/domain/repositories/attention_event_repository.dart`
- `lib/data/repositories/central_backend_repositories.dart`
- `test/repositories/central_backend_repositories_test.dart`
- `test/composite_dashboard_repository_test.dart`
- `lib/features/dashboard/server_dashboard_screen.dart`
- `test/features/dashboard/server_dashboard_screen_test.dart`
- `lib/features/shell.dart`
- `test/shell_navigation_structure_test.dart`
- `.github/workflows/p1_p2_dashboard_ci.yml`

**Do not make authoritative**
- `lib/features/alerts/alerts_screen.dart`
- `RosterController.alerts` / `RosterController.unacknowledgedCount`
- local `ClinicalAlert` persistence.

---

### Task 1: Expand the AttentionEvent repository contract and keep production transport gated

**Files:**
- Modify: `lib/domain/repositories/attention_event_repository.dart`
- Modify: `lib/data/repositories/central_backend_repositories.dart`
- Modify: `test/repositories/central_backend_repositories_test.dart`
- Modify: `test/composite_dashboard_repository_test.dart`

**Interfaces:**
- Consumes: existing `AttentionEvent`, `ApiException`, `ApiFailure.notConfigured`.
- Produces:

```dart
abstract interface class AttentionEventRepository {
  Future<List<AttentionEvent>> openEvents();
  Future<List<AttentionEvent>> activity({String? subjectId});
  Future<AttentionEvent?> eventById(String eventId);
  Future<AttentionEvent> acknowledge(String eventId);
  Future<AttentionEvent> resolve(String eventId);
}
```

- Produces: `CentralBackendAttentionEventRepository implements AttentionEventRepository`, with every method throwing `_contractGate(...)` and performing no network request.

- [ ] **Step 1: Write failing repository tests**

Extend `test/repositories/central_backend_repositories_test.dart` with one assertion per event operation:

```dart
test('attention-event target adapter remains blocked without verified routes', () async {
  final repo = CentralBackendAttentionEventRepository(noNetworkApi());

  await expectLater(repo.openEvents(), _notConfigured());
  await expectLater(repo.activity(), _notConfigured());
  await expectLater(repo.activity(subjectId: 'subject-001'), _notConfigured());
  await expectLater(repo.eventById('evt-001'), _notConfigured());
  await expectLater(repo.acknowledge('evt-001'), _notConfigured());
  await expectLater(repo.resolve('evt-001'), _notConfigured());
});
```

Add a local matcher helper so the test is explicit and DRY:

```dart
Matcher _notConfigured() => throwsA(
      isA<ApiException>().having(
        (e) => e.kind,
        'kind',
        ApiFailure.notConfigured,
      ),
    );
```

- [ ] **Step 2: Run the repository test and confirm RED**

Run:

```bash
flutter test test/repositories/central_backend_repositories_test.dart
```

Expected: compile/test failure because `CentralBackendAttentionEventRepository` and the expanded interface methods do not yet exist.

- [ ] **Step 3: Expand the repository interface**

Replace the interface body with the exact five methods above.

- [ ] **Step 4: Add the contract-gated production repository**

In `central_backend_repositories.dart`, import `AttentionEvent` and `AttentionEventRepository`, then add:

```dart
class CentralBackendAttentionEventRepository implements AttentionEventRepository {
  CentralBackendAttentionEventRepository([ApiClient? api]);

  @override
  Future<List<AttentionEvent>> openEvents() async {
    throw _contractGate('attention-events-open-contract');
  }

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async {
    throw _contractGate('attention-events-activity-contract');
  }

  @override
  Future<AttentionEvent?> eventById(String eventId) async {
    throw _contractGate('attention-event-detail-contract');
  }

  @override
  Future<AttentionEvent> acknowledge(String eventId) async {
    throw _contractGate('attention-event-acknowledge-contract');
  }

  @override
  Future<AttentionEvent> resolve(String eventId) async {
    throw _contractGate('attention-event-resolve-contract');
  }
}
```

No endpoint string or `ApiClient.get/post` call is allowed in this class yet.

- [ ] **Step 5: Repair the existing dashboard fake for the expanded interface**

In `test/composite_dashboard_repository_test.dart`, keep `openEvents()` returning the fixture list and add methods that throw `UnimplementedError` because that fake is only used by the composite-dashboard test:

```dart
@override
Future<List<AttentionEvent>> activity({String? subjectId}) =>
    throw UnimplementedError();

@override
Future<AttentionEvent?> eventById(String eventId) =>
    throw UnimplementedError();

@override
Future<AttentionEvent> acknowledge(String eventId) =>
    throw UnimplementedError();

@override
Future<AttentionEvent> resolve(String eventId) =>
    throw UnimplementedError();
```

- [ ] **Step 6: Run repository/contract tests**

```bash
flutter test test/repositories/central_backend_repositories_test.dart test/composite_dashboard_repository_test.dart test/contracts/attention_event_test.dart
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/domain/repositories/attention_event_repository.dart lib/data/repositories/central_backend_repositories.dart test/repositories/central_backend_repositories_test.dart test/composite_dashboard_repository_test.dart
git commit -m "feat: define P4 attention-event repository boundary"
```

---

### Task 2: Add the Activity controller with explicit safe states

**Files:**
- Create: `lib/state/attention_events_controller.dart`
- Create: `test/attention_events_controller_test.dart`

**Interfaces:**
- Consumes: `AttentionEventRepository`, `AuthRepository`, `ApiException`, `AsyncDataState<List<AttentionEvent>>`.
- Produces:

```dart
class AttentionEventsController extends ChangeNotifier {
  final AttentionEventRepository repository;
  final AuthRepository? authRepository;
  final String? subjectId;

  AsyncDataState<List<AttentionEvent>> get state;
  Future<void> load({bool showLoading = true});
}
```

- [ ] **Step 1: Write failing controller tests**

Cover these behaviors using a fake repository that returns or throws on `activity(...)`:

```dart
test('loads server event activity without reordering', () async { ... });
test('empty server activity becomes empty state', () async { ... });
test('401 expires clinician session', () async { ... });
test('403 keeps clinician session active', () async { ... });
test('notConfigured becomes unavailable', () async { ... });
test('offline timeout and server failures are explicit offline state', () async { ... });
test('subject-scoped load forwards only the canonical subject id', () async { ... });
```

The first test must assert object order and exact IDs, e.g. `['evt-003', 'evt-002', 'evt-001']`, proving the controller does not invent a clinical ranking.

- [ ] **Step 2: Run and confirm RED**

```bash
flutter test test/attention_events_controller_test.dart
```

Expected: failure because `AttentionEventsController` is absent.

- [ ] **Step 3: Implement minimal controller**

Use this error mapping:

```dart
switch (e.kind) {
  case ApiFailure.unauthorized:
    await authRepository?.expireCurrentSession();
    _state = AsyncDataState.sessionExpired(message: e.message);
  case ApiFailure.forbidden:
    _state = AsyncDataState.forbidden(message: e.message);
  case ApiFailure.conflict:
    _state = AsyncDataState.conflict(message: e.message);
  case ApiFailure.notConfigured:
  case ApiFailure.notFound:
    _state = AsyncDataState.unavailable(message: e.message);
  case ApiFailure.offline:
  case ApiFailure.timeout:
  case ApiFailure.server:
    _state = AsyncDataState.offline(message: e.message);
  default:
    _state = AsyncDataState.error(message: e.message);
}
```

`load()` calls exactly:

```dart
final events = await repository.activity(subjectId: subjectId);
_state = events.isEmpty
    ? const AsyncDataState<List<AttentionEvent>>.empty()
    : AsyncDataState<List<AttentionEvent>>.data(events);
```

Do not sort, filter, derive severity, or create events in the controller.

- [ ] **Step 4: Run controller tests**

```bash
flutter test test/attention_events_controller_test.dart
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/state/attention_events_controller.dart test/attention_events_controller_test.dart
git commit -m "feat: add P4 attention activity controller"
```

---

### Task 3: Add the Event Detail controller and canonical lifecycle reconciliation

**Files:**
- Create: `lib/state/attention_event_detail_controller.dart`
- Create: `test/attention_event_detail_controller_test.dart`

**Interfaces:**
- Consumes: `AttentionEventRepository`, `AuthRepository`, `AttentionEventStatus`, `AsyncDataState<AttentionEvent>`.
- Produces:

```dart
class AttentionEventDetailController extends ChangeNotifier {
  final String eventId;
  final AttentionEventRepository repository;
  final AuthRepository? authRepository;

  AsyncDataState<AttentionEvent> get state;
  bool get isMutating;
  String? get mutationMessage;

  Future<void> load({bool showLoading = true});
  Future<void> acknowledge();
  Future<void> resolve();
}
```

- [ ] **Step 1: Write failing lifecycle tests**

Required tests:

```dart
test('OPEN event acknowledge replaces state with canonical server response', () async { ... });
test('ACKNOWLEDGED event resolve replaces state with canonical server response', () async { ... });
test('resolved event does not call mutation repository', () async { ... });
test('unknown event does not call mutation repository', () async { ... });
test('duplicate acknowledge tap sends one mutation while request is in flight', () async { ... });
test('401 expires clinician session and does not fabricate success', () async { ... });
test('403 preserves valid session and original canonical event', () async { ... });
test('404 becomes unavailable', () async { ... });
test('409 reloads canonical event and shows server actor and time', () async { ... });
test('offline acknowledge leaves OPEN event OPEN', () async { ... });
test('offline resolve leaves ACKNOWLEDGED event ACKNOWLEDGED', () async { ... });
```

For 409, the fake repository should throw `ApiFailure.conflict` from `acknowledge`, then return an already-acknowledged event from `eventById`. Assert the controller ends with that server object, including its actor/timestamp.

- [ ] **Step 2: Run and confirm RED**

```bash
flutter test test/attention_event_detail_controller_test.dart
```

Expected: failure because the controller does not exist.

- [ ] **Step 3: Implement load and mutation guard**

`load()` fetches only `repository.eventById(eventId)`. `null` becomes unavailable.

Both mutation methods must start with:

```dart
if (_isMutating) return;
final current = _state.data;
if (current == null) return;
```

Then enforce legal source state:

```dart
if (current.status != AttentionEventStatus.open) return; // acknowledge
if (current.status != AttentionEventStatus.acknowledged) return; // resolve
```

- [ ] **Step 4: Implement server-response replacement**

Successful mutation must assign the returned object directly:

```dart
final canonical = await repository.acknowledge(eventId);
_state = AsyncDataState<AttentionEvent>.data(canonical);
```

and similarly for `resolve`. Do not call `copyWith`, device clock, SecureStore clinician id, or local status mutation.

- [ ] **Step 5: Implement 409 reconciliation**

On `ApiFailure.conflict`:

```dart
final canonical = await repository.eventById(eventId);
_state = canonical == null
    ? const AsyncDataState<AttentionEvent>.unavailable(
        message: 'This attention event is no longer available on the server.',
      )
    : AsyncDataState<AttentionEvent>.data(canonical);
_mutationMessage =
    'This event changed on the server. The current server state is shown.';
```

If the reconciliation fetch itself fails, map that second failure normally; never retain a fake transitioned state.

- [ ] **Step 6: Run lifecycle tests**

```bash
flutter test test/attention_event_detail_controller_test.dart
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/state/attention_event_detail_controller.dart test/attention_event_detail_controller_test.dart
git commit -m "feat: add canonical attention-event lifecycle controller"
```

---

### Task 4: Build the server-backed Activity screen

**Files:**
- Create: `lib/features/attention_events/activity_screen.dart`
- Create: `test/features/attention_events/activity_screen_test.dart`

**Interfaces:**
- Consumes: `AttentionEventsController`.
- Produces two constructors matching P3's injection/production pattern:

```dart
const ActivityScreen({
  super.key,
  required AttentionEventsController controller,
  ValueChanged<AttentionEvent>? onOpenEvent,
});

const ActivityScreen.production({super.key});
```

Production construction creates `AttentionEventsController(repository: CentralBackendAttentionEventRepository(), authRepository: CentralBackendAuthRepository())` and loads once.

- [ ] **Step 1: Write failing widget tests**

Tests must assert:

```dart
testWidgets('groups OPEN ACKNOWLEDGED and RESOLVED server events', ...);
testWidgets('preserves server order within each lifecycle group', ...);
testWidgets('opens event detail with the same server event id', ...);
testWidgets('notConfigured renders Activity unavailable and never legacy alerts', ...);
testWidgets('empty activity says no server attention-event activity', ...);
testWidgets('unknown status is visible as unknown rather than treated as open or resolved', ...);
```

The unavailable test must assert the screen does not contain legacy wording such as `RED or DARK RED band`.

- [ ] **Step 2: Run and confirm RED**

```bash
flutter test test/features/attention_events/activity_screen_test.dart
```

- [ ] **Step 3: Implement Activity layout**

Render sections in this order:

```text
Activity
OPEN
ACKNOWLEDGED
RESOLVED
UNKNOWN (only when present)
```

Each row may show only contract-backed values: subject id, event id, reason/event type, status, severity, horizon, created time, and lifecycle actor/time when present.

Do not show any locally calculated band/tier as event authority.

- [ ] **Step 4: Implement event navigation callback/default**

If `onOpenEvent` is supplied, call it with the exact `AttentionEvent` object. Otherwise push `AttentionEventDetailScreen.production(eventId: event.id)`.

- [ ] **Step 5: Run widget tests**

```bash
flutter test test/features/attention_events/activity_screen_test.dart
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lib/features/attention_events/activity_screen.dart test/features/attention_events/activity_screen_test.dart
git commit -m "feat: add server-backed attention Activity screen"
```

---

### Task 5: Build Event Detail UI and canonical Patient Overview navigation

**Files:**
- Create: `lib/features/attention_events/attention_event_detail_screen.dart`
- Create: `test/features/attention_events/attention_event_detail_screen_test.dart`

**Interfaces:**
- Consumes: `AttentionEventDetailController`.
- Produces injection + production constructors:

```dart
const AttentionEventDetailScreen({
  super.key,
  required AttentionEventDetailController controller,
  ValueChanged<String>? onOpenPatient,
});

const AttentionEventDetailScreen.production({
  super.key,
  required String eventId,
});
```

- [ ] **Step 1: Write failing widget tests**

Cover:

```dart
testWidgets('detail shows server identity and provenance fields', ...);
testWidgets('OPEN shows Acknowledge and no Resolve action', ...);
testWidgets('ACKNOWLEDGED shows Resolve and server acknowledgement actor/time', ...);
testWidgets('RESOLVED shows no mutation action and server resolution actor/time', ...);
testWidgets('UNKNOWN shows no mutation action', ...);
testWidgets('mutation button is disabled while request is in flight', ...);
testWidgets('409 message says server state changed and renders canonical returned state', ...);
testWidgets('View patient uses the exact canonical subject id', ...);
```

The provenance test must assert `event.id`, `subjectId`, `fusionResultId`, `forecastResultId`, `policyVersion`, horizon, event type/reason where present.

- [ ] **Step 2: Run and confirm RED**

```bash
flutter test test/features/attention_events/attention_event_detail_screen_test.dart
```

- [ ] **Step 3: Implement detail presentation**

Use `AnimatedBuilder` and explicit loading/unavailable/offline/session/forbidden states. Render status labels directly from `AttentionEventStatus`.

For forecast wording use cautious copy, e.g.:

```text
Potential escalation was flagged within the server-provided forecast horizon.
```

Do not say the patient will experience an attack at an exact time.

- [ ] **Step 4: Implement legal lifecycle actions**

Render:

```dart
if (event.status == AttentionEventStatus.open)
  OutlinedButton(onPressed: controller.isMutating ? null : controller.acknowledge, child: const Text('Acknowledge'));

if (event.status == AttentionEventStatus.acknowledged)
  OutlinedButton(onPressed: controller.isMutating ? null : controller.resolve, child: const Text('Resolve'));
```

No mutation action for resolved/unknown.

- [ ] **Step 5: Implement Patient Overview navigation**

The injected callback receives only `event.subjectId`. The production default pushes:

```dart
PatientOverviewScreen.production(subjectId: event.subjectId)
```

Never derive the subject from MRN/local roster data.

- [ ] **Step 6: Run widget tests**

```bash
flutter test test/features/attention_events/attention_event_detail_screen_test.dart
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/features/attention_events/attention_event_detail_screen.dart test/features/attention_events/attention_event_detail_screen_test.dart
git commit -m "feat: add attention-event detail and lifecycle actions"
```

---

### Task 6: Wire Dashboard -> Event Detail and migrate shell Alerts -> Activity

**Files:**
- Modify: `lib/features/dashboard/server_dashboard_screen.dart`
- Modify: `test/features/dashboard/server_dashboard_screen_test.dart`
- Modify: `lib/features/shell.dart`
- Modify: `test/shell_navigation_structure_test.dart`

**Interfaces:**
- `ServerDashboardScreen` gains:

```dart
final ValueChanged<AttentionEvent>? onOpenEvent;
```

- Shell uses `ActivityScreen.production()` as tab 3 and label `Activity`.
- Shell no longer watches `RosterController` solely to display the legacy local-alert badge.

- [ ] **Step 1: Write failing Dashboard navigation test**

Add:

```dart
testWidgets('opens event detail with the same server event id', (tester) async {
  AttentionEvent? opened;
  ...
  onOpenEvent: (event) => opened = event,
  ...
  await tester.tap(find.text('evt-001'));
  expect(opened?.id, 'evt-001');
  expect(opened?.subjectId, 'subject-001');
});
```

- [ ] **Step 2: Write failing shell migration assertions**

Extend `test/shell_navigation_structure_test.dart`:

```dart
expect(shell, contains('ActivityScreen.production()'));
expect(shell, contains("label: 'Activity'"));
expect(shell, isNot(contains('AlertsScreen()')));
expect(shell, isNot(contains('roster.unacknowledgedCount')));
expect(shell, isNot(contains("label: 'Alerts'")));
```

- [ ] **Step 3: Run tests and confirm RED**

```bash
flutter test test/features/dashboard/server_dashboard_screen_test.dart test/shell_navigation_structure_test.dart
```

- [ ] **Step 4: Make Dashboard event cards tappable**

Add `_openEvent(...)` parallel to `_openPatient(...)`. Default navigation pushes `AttentionEventDetailScreen.production(eventId: event.id)`. Pass `onTap` into `_AttentionEventCard` and then into `Panel`.

- [ ] **Step 5: Replace shell target tab with Activity**

Remove the target shell import of `alerts_screen.dart`; import `attention_events/activity_screen.dart`; replace `AlertsScreen()` with `ActivityScreen.production()` and use a normal notifications/history icon with label `Activity`.

Do not show a badge sourced from local `RosterController.unacknowledgedCount`. It is safer to show no event count until a server-backed count is available through a verified contract than to present the legacy local count as authoritative.

- [ ] **Step 6: Run navigation tests**

```bash
flutter test test/features/dashboard/server_dashboard_screen_test.dart test/shell_navigation_structure_test.dart
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/features/dashboard/server_dashboard_screen.dart test/features/dashboard/server_dashboard_screen_test.dart lib/features/shell.dart test/shell_navigation_structure_test.dart
git commit -m "feat: route server attention events through Activity and detail"
```

---

### Task 7: Add P4 authority guards and CI blocking checks

**Files:**
- Create: `test/p4_attention_event_authority_test.dart`
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`

**Interfaces:**
- Produces static regression guards for authority boundaries.

- [ ] **Step 1: Write failing authority test**

Read active source files with `dart:io` and assert:

```dart
final activity = File('lib/features/attention_events/activity_screen.dart').readAsStringSync();
final detail = File('lib/features/attention_events/attention_event_detail_screen.dart').readAsStringSync();
final detailController = File('lib/state/attention_event_detail_controller.dart').readAsStringSync();
final shell = File('lib/features/shell.dart').readAsStringSync();
final repos = File('lib/data/repositories/central_backend_repositories.dart').readAsStringSync();

expect(activity, isNot(contains('ClinicalAlert')));
expect(activity, isNot(contains('RosterController')));
expect(detail, isNot(contains('AlertBandX.fromScore')));
expect(detailController, isNot(contains('DateTime.now()')));
expect(detailController, isNot(contains('SecureStore.clinicianId')));
expect(shell, isNot(contains('roster.unacknowledgedCount')));
expect(repos, isNot(contains('/attention-events')));
expect(repos, isNot(contains('/acknowledge')));
expect(repos, isNot(contains('/resolve')));
```

Also preserve the existing active-source checks that no `_raiseIfEscalated(` / `roster.raiseAlert(ClinicalAlert(` path has returned.

- [ ] **Step 2: Run authority test**

```bash
flutter test test/p4_attention_event_authority_test.dart
```

Expected after Tasks 1-6: PASS. If it fails, fix the authority violation instead of weakening the assertion.

- [ ] **Step 3: Extend workflow branch/focused tests**

Add `integration/clinanx-p4-attention-events` to the push branches.

Add to focused test command:

```text
test/attention_events_controller_test.dart
test/attention_event_detail_controller_test.dart
test/features/attention_events/activity_screen_test.dart
test/features/attention_events/attention_event_detail_screen_test.dart
test/p4_attention_event_authority_test.dart
```

- [ ] **Step 4: Extend focused analyzer inputs**

Add:

```text
lib/state/attention_events_controller.dart
lib/state/attention_event_detail_controller.dart
lib/features/attention_events/activity_screen.dart
lib/features/attention_events/attention_event_detail_screen.dart
test/attention_events_controller_test.dart
test/attention_event_detail_controller_test.dart
test/features/attention_events/activity_screen_test.dart
test/features/attention_events/attention_event_detail_screen_test.dart
test/p4_attention_event_authority_test.dart
```

- [ ] **Step 5: Extend shell authority checks**

Add bash checks that fail if active P4 files contain guessed attention-event route strings, local score-to-event derivation, direct `'/predict'` calls, or if `lib/features/shell.dart` contains `roster.unacknowledgedCount`.

- [ ] **Step 6: Commit**

```bash
git add test/p4_attention_event_authority_test.dart .github/workflows/p1_p2_dashboard_ci.yml
git commit -m "ci: enforce P4 attention-event authority boundaries"
```

---

### Task 8: Full verification, documentation, diff review, and PR handoff

**Files:**
- Create: `docs/superpowers/verification/2026-09-16-clinanx-p4-attention-events-verification.md`

**Interfaces:**
- Produces evidence only; no feature behavior changes.

- [ ] **Step 1: Recheck teammate backend before final verification**

Read-only verify the visible current SHA of `UVINDUSEN/component4final` and search for persistent AttentionEvent list/detail/acknowledge/resolve implementation. If no verified contract exists, keep production adapter gated. Do not write to the teammate repo.

- [ ] **Step 2: Run focused P0-P4 tests**

```bash
flutter test \
  test/contracts \
  test/repositories \
  test/gateway_contract_test.dart \
  test/p1_p2_api_failure_test.dart \
  test/p1_p2_authority_cleanup_test.dart \
  test/p3_patient_overview_authority_test.dart \
  test/p4_attention_event_authority_test.dart \
  test/composite_dashboard_repository_test.dart \
  test/dashboard_controller_test.dart \
  test/dashboard_cache_test.dart \
  test/patient_overview_controller_test.dart \
  test/attention_events_controller_test.dart \
  test/attention_event_detail_controller_test.dart \
  test/features/dashboard/server_dashboard_screen_test.dart \
  test/features/patients/patient_overview_screen_test.dart \
  test/features/patients/patients_screen_navigation_test.dart \
  test/features/attention_events/activity_screen_test.dart \
  test/features/attention_events/attention_event_detail_screen_test.dart \
  test/shell_navigation_structure_test.dart
```

Record the exact passed/failed counts.

- [ ] **Step 3: Run focused analysis**

```bash
flutter analyze \
  lib/domain/contracts \
  lib/domain/repositories \
  lib/data/repositories \
  lib/state/async_data_state.dart \
  lib/state/dashboard_controller.dart \
  lib/state/patient_overview_controller.dart \
  lib/state/attention_events_controller.dart \
  lib/state/attention_event_detail_controller.dart \
  lib/features/dashboard/server_dashboard_screen.dart \
  lib/features/patients/patient_overview_screen.dart \
  lib/features/attention_events/activity_screen.dart \
  lib/features/attention_events/attention_event_detail_screen.dart \
  lib/features/shell.dart \
  test/attention_events_controller_test.dart \
  test/attention_event_detail_controller_test.dart \
  test/features/attention_events/activity_screen_test.dart \
  test/features/attention_events/attention_event_detail_screen_test.dart \
  test/p4_attention_event_authority_test.dart
```

Expected: `No issues found!`.

- [ ] **Step 4: Run whole-repository visibility**

```bash
flutter analyze
flutter test
```

Record the exact analyzer output and exact test pass/fail count. If the known consent-first test remains the only failure, document it as pre-existing baseline drift; do not claim the whole suite passed.

- [ ] **Step 5: Inspect branch diff against main**

Confirm changes are limited to the approved P4 spec/plan, repository boundary, event controllers/screens/navigation/tests/CI/verification. Confirm no teammate repo changed and no unrelated research/model code was altered.

- [ ] **Step 6: Write verification document**

Record:
- branch/base/head SHAs;
- backend SHA and contract-gate finding;
- RED/GREEN TDD evidence by task;
- focused test count;
- focused analyzer result;
- authority check result;
- whole-suite result including any baseline failure;
- whole analyzer result;
- explicit statement that production AttentionEvent transport is still gated if backend routes remain unverified;
- explicit statement that no local event authority was introduced.

- [ ] **Step 7: Commit verification document**

```bash
git add docs/superpowers/verification/2026-09-16-clinanx-p4-attention-events-verification.md
git commit -m "docs: record ClinAnx P4 verification"
```

- [ ] **Step 8: Open PR without merging**

PR title:

```text
Add ClinAnx P4 Server Attention Events
```

PR body must summarize:
- Activity + Event Detail workflow;
- server-owned lifecycle and concurrency reconciliation;
- legacy local-alert authority migration;
- backend contract gate;
- exact verification evidence;
- known baseline failure if still present.

Do not merge the PR. User approval remains required for merge.

---

## Self-Review Result

- **Spec coverage:** repository boundary, worklist/detail state, Activity, Event Detail, Dashboard navigation, Patient Overview navigation, legal lifecycle transitions, canonical server responses, 401/403/404/409 handling, concurrency reconciliation, legacy alert migration, production contract gate, authority tests, CI, and verification are all mapped to tasks.
- **Placeholder scan:** no `TBD`, unresolved implementation placeholder, or guessed endpoint is used in this plan.
- **Type consistency:** repository signatures, controller method names, injected callbacks, production constructors, and server identity fields are consistent across tasks.
- **Scope:** P4 remains clinician-app event workflow only. Backend AttentionEngine implementation, patient app, push notifications, realtime transport, and P5 deep clinical views remain out of scope.
