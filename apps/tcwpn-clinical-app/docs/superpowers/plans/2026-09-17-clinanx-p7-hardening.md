# ClinAnx Phase 7 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden ClinAnx for research-demo use by isolating local clinical data per clinician, preserving explicit stale/offline/failure semantics, improving accessibility/configuration truthfulness, and making the full test/analyze suite blocking.

**Architecture:** Keep the Central Backend as the only clinical authority. Add one clinician storage-scope abstraction below repositories/controllers, move authenticated local state initialization into authenticated context, preserve server-provided values as immutable domain state, and enforce failure semantics through tests rather than client-side clinical derivation.

**Tech Stack:** Flutter/Dart, provider, shared_preferences, flutter_secure_storage, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-17-clinanx-p7-hardening-design.md`

## Global Constraints

- Modify only `dulhara79/tcwpn_mobile_app`.
- Do not invent Central Backend routes or payloads.
- Central Backend remains authoritative for assignments, fusion/current assessment, forecast, AttentionEvent lifecycle, actor identity, and timestamps.
- Missing/stale/unavailable data must never become Low/Green/0/safe.
- Legacy unscoped local clinical records must not be silently assigned to the first clinician after upgrade.
- TDD: every behavior change starts with a failing test and ends with focused/full verification.
- No merge to `main`; create a PR only after fresh verification.

---

### Task 1: Clinician storage scope and cache isolation

**Files:**
- Create: `lib/data/local/clinician_storage_scope.dart`
- Modify: `lib/data/local/stores.dart`
- Modify: `lib/data/local/dashboard_cache.dart`
- Modify: `lib/data/local/attention_notification_store.dart`
- Test: `test/p7_clinician_storage_scope_test.dart`
- Test: `test/dashboard_cache_test.dart`
- Test: `test/attention_notification_store_test.dart`

**Interfaces:**
- Produces: `ClinicianStorageScope` with a normalized `scopeId` and `key(String logicalKey)` method.
- `RecordStore` read/write methods consume the active clinician scope explicitly or through a single bound scope API.
- `DashboardCacheStore` is constructed with a clinician scope.
- `SharedPreferencesAttentionNotificationStore` uses the same clinician scope for delivered IDs and pending-open state.

- [ ] **Step 1: Write failing clinician-isolation tests**

```dart
final drA = ClinicianStorageScope('DR001');
final drB = ClinicianStorageScope('DR002');
expect(drA.key('roster'), isNot(drB.key('roster')));
```

Add persistence tests proving records written under DR001 are invisible under DR002, dashboard cache does not cross scopes, and a pending notification-open saved under DR001 is not returned under DR002.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
flutter test test/p7_clinician_storage_scope_test.dart test/dashboard_cache_test.dart test/attention_notification_store_test.dart
```

Expected: FAIL because no common clinician storage scope exists and current keys are global.

- [ ] **Step 3: Implement `ClinicianStorageScope`**

Normalize clinician IDs by trim + uppercase and reject blank scopes. Build keys as:

```dart
String key(String logicalKey) => 'clinanx_v1::$scopeId::$logicalKey';
```

Do not read legacy unscoped keys as fallback.

- [ ] **Step 4: Scope RecordStore/dashboard/notification keys**

Use clinician-scoped keys for roster, alerts, site support, notes, support, cached fusion, subject IDs, dashboard snapshots, delivered notification IDs, and pending notification-open IDs.

- [ ] **Step 5: Re-run focused tests and verify GREEN**

Run the same focused test command; expect all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/data/local test/p7_clinician_storage_scope_test.dart test/dashboard_cache_test.dart test/attention_notification_store_test.dart
git commit -m "feat: isolate local clinical state by clinician"
```

---

### Task 2: Authenticated lifecycle for roster and local state

**Files:**
- Modify: `lib/main.dart`
- Modify: `lib/features/auth/login_screen.dart`
- Modify: `lib/features/settings/settings_screen.dart`
- Modify: `lib/state/controllers.dart`
- Test: `test/p7_authenticated_roster_lifecycle_test.dart`

**Interfaces:**
- `RosterController` receives a `ClinicianStorageScope` at construction.
- Root app does not initialize roster data before an authenticated clinician exists.
- Sign-out removes secure session/in-memory session and disposes authenticated providers by navigation replacement.

- [ ] **Step 1: Write failing lifecycle tests**

Prove the consent/login route can build without constructing a roster controller and that an authenticated shell constructs roster state with the current clinician scope only.

- [ ] **Step 2: Run test and verify RED**

```bash
flutter test test/p7_authenticated_roster_lifecycle_test.dart
```

- [ ] **Step 3: Move `RosterController` creation into authenticated app/shell context**

Remove root-level unconditional `ChangeNotifierProvider(create: (_) => RosterController()..init())`. Construct it only after `Session.clinicianId` is nonblank and bind its `RecordStore` operations to that scope.

- [ ] **Step 4: Ensure sign-in/sign-out transition creates a fresh authenticated scope**

No clinical provider from the prior clinician may survive a sign-out/navigation reset.

- [ ] **Step 5: Re-run test and verify GREEN**

- [ ] **Step 6: Commit**

```bash
git add lib/main.dart lib/features/auth/login_screen.dart lib/features/settings/settings_screen.dart lib/state/controllers.dart test/p7_authenticated_roster_lifecycle_test.dart
git commit -m "refactor: bind local clinical state to authenticated clinician"
```

---

### Task 3: Failure-safe offline/stale semantics

**Files:**
- Modify only if tests expose a gap: `lib/state/dashboard_controller.dart`
- Modify only if tests expose a gap: `lib/state/patient_overview_controller.dart`
- Modify only if tests expose a gap: `lib/state/attention_event_detail_controller.dart`
- Test: `test/p7_failure_injection_test.dart`

**Interfaces:**
- 401 -> session expired and secure session cleared through auth repository.
- 403 -> forbidden without clearing valid session.
- 409 -> conflict state / reload canonical state in mutation flow.
- offline/timeout/5xx -> cached stale/offline data only when cache exists.
- malformed/missing modality -> explicit error/unavailable, never Low/Green/0.

- [ ] **Step 1: Add the failure-matrix tests**

Cover backend unavailable, expired JWT, forbidden access, conflict, malformed payload, network loss with and without cache, stale C1, unavailable C3, missing C4, RAG unavailable/abstained fixture rendering, and duplicate event delivery behavior.

- [ ] **Step 2: Run and verify RED only where a real client gap exists**

```bash
flutter test test/p7_failure_injection_test.dart
```

- [ ] **Step 3: Implement only the minimal missing behavior**

Do not add new backend routes or clinical threshold logic. Preserve existing safe behavior where tests already pass.

- [ ] **Step 4: Re-run focused failure tests and existing related controller tests**

```bash
flutter test test/p7_failure_injection_test.dart test/dashboard_controller_test.dart test/patient_overview_controller_test.dart test/attention_event_detail_controller_test.dart test/attention_notification_controller_test.dart
```

- [ ] **Step 5: Commit**

```bash
git add lib/state test/p7_failure_injection_test.dart
git commit -m "test: harden ClinAnx failure-safe behavior"
```

---

### Task 4: Accessibility hardening

**Files:**
- Modify: `lib/main.dart`
- Modify only if overflow tests expose defects: critical feature widgets
- Test: `test/p7_accessibility_test.dart`

**Interfaces:**
- The app respects system text scaling instead of globally clamping to 1.3x.
- Critical clinical state remains conveyed by readable text/labels/icons.

- [ ] **Step 1: Write failing large-text tests**

Pump Dashboard/Patient Overview/Activity/Event Detail/Settings/Login under a large `TextScaler` and assert no exceptions/overflow in the tested viewport. Assert semantic/status text is present independently of color.

- [ ] **Step 2: Run and verify RED**

```bash
flutter test test/p7_accessibility_test.dart
```

- [ ] **Step 3: Remove the restrictive global text-scale clamp**

Replace the clamped `MediaQuery` builder with a pass-through themed container. Fix only actual overflow points exposed by tests using wrapping/flexible layout, not smaller fonts.

- [ ] **Step 4: Re-run accessibility tests and verify GREEN**

- [ ] **Step 5: Commit**

```bash
git add lib/main.dart lib/features test/p7_accessibility_test.dart
git commit -m "fix: respect clinician accessibility text scaling"
```

---

### Task 5: Configuration and security truthfulness

**Files:**
- Modify: `lib/core/config/env.dart`
- Modify: `lib/data/api/auth_service.dart`
- Modify: `lib/features/settings/settings_screen.dart`
- Modify: `lib/core/security/secure_http.dart` and/or `lib/core/security/pinned_certificates.dart` only if needed for truthful diagnostics
- Test: `test/p7_release_config_test.dart`
- Test: `test/p7_settings_truthfulness_test.dart`

**Interfaces:**
- `AuthService` reads `Env.authBase`, `Env.authSalt`, `Env.authLocalAccounts`.
- `Env.demoData` defaults `false`.
- Settings says note text goes to Central Backend, not directly to the model service.
- Pinning UI distinguishes pinned hosts from platform-TLS hosts and does not claim all ClinAnx traffic is pinned.

- [ ] **Step 1: Write failing configuration/truthfulness tests**

Assert no `String.fromEnvironment('AUTH_` remains in `auth_service.dart`; `DEMO_DATA` default is false; Settings source text contains Central Backend wording and does not contain the false direct-model statement; TLS diagnostics do not make a blanket all-traffic pinning claim.

- [ ] **Step 2: Run and verify RED**

```bash
flutter test test/p7_release_config_test.dart test/p7_settings_truthfulness_test.dart
```

- [ ] **Step 3: Centralize auth config and change demo default**

Use `Env` values in `AuthService`. Set `bool.fromEnvironment('DEMO_DATA', defaultValue: false)`.

- [ ] **Step 4: Correct Settings/security wording**

Describe Central Backend orchestration accurately. Display pin coverage honestly. Do not generate or hardcode new pins.

- [ ] **Step 5: Re-run tests and verify GREEN**

- [ ] **Step 6: Commit**

```bash
git add lib/core/config/env.dart lib/data/api/auth_service.dart lib/features/settings/settings_screen.dart lib/core/security test/p7_release_config_test.dart test/p7_settings_truthfulness_test.dart
git commit -m "fix: make release configuration and security diagnostics truthful"
```

---

### Task 6: Blocking CI and Phase 7 authority/privacy guards

**Files:**
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`
- Create: `test/p7_hardening_authority_test.dart`

**Interfaces:**
- Full `flutter test` and full `flutter analyze` fail the workflow on failure.
- Branch `integration/clinanx-p7-hardening` triggers CI.
- Guards reject unscoped clinical SharedPreferences keys, restored direct model prediction calls, local clinical threshold derivation in authoritative views, and blanket unsafe config regressions.

- [ ] **Step 1: Add failing/source-inspection authority tests**

Check that critical stores use clinician scope, `DEMO_DATA` does not default true, and server-backed clinical views do not introduce local fusion/risk authority.

- [ ] **Step 2: Update workflow**

Add Phase 7 focused tests, add branch trigger, and remove `continue-on-error: true` from full tests and full analysis.

- [ ] **Step 3: Run local/source test where possible**

```bash
flutter test test/p7_hardening_authority_test.dart
flutter analyze
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/p1_p2_dashboard_ci.yml test/p7_hardening_authority_test.dart
git commit -m "ci: make Phase 7 hardening checks blocking"
```

---

### Task 7: Usability checklist and verification evidence

**Files:**
- Create: `docs/qa/phase7_usability_checklist.md`
- Create: `docs/superpowers/verification/2026-09-17-clinanx-p7-hardening.md`

**Interfaces:**
- Checklist contains repeatable tasks, expected safe behavior, and a defect-recording section.
- Verification document records branch/base/head, exact commands, pass counts, known external blockers, and does not claim live backend E2E where unavailable.

- [ ] **Step 1: Add usability checklist**

Cover sign-in, Needs Attention, current-vs-forecast interpretation, unavailable/stale modality, event ACK/resolve, RAG abstention/unavailable, and enlarged-text navigation.

- [ ] **Step 2: Run fresh full verification**

```bash
flutter test
flutter analyze
```

Also run the focused Phase 7 tests and existing authority checks.

- [ ] **Step 3: Record evidence exactly**

Record actual commands/results only. If verification occurs only in GitHub Actions, say so explicitly.

- [ ] **Step 4: Commit**

```bash
git add docs/qa/phase7_usability_checklist.md docs/superpowers/verification/2026-09-17-clinanx-p7-hardening.md
git commit -m "docs: record Phase 7 hardening verification"
```

---

### Task 8: Final branch review and PR

**Files:** none unless review finds a defect.

- [ ] **Step 1: Re-fetch `main` and compare branch drift**

Confirm no unreviewed main changes require rebasing/reconciliation.

- [ ] **Step 2: Review branch diff against approved Phase 7 spec**

No teammate repository changes, no invented backend routes, no local authoritative risk computation, and all local clinical caches clinician-scoped.

- [ ] **Step 3: Confirm fresh GitHub Actions success on branch head**

Do not use an older commit's run as completion evidence.

- [ ] **Step 4: Create PR to `main`**

PR body must state which Phase 7 requirements are implemented and which whole-system requirements still depend on Uvindu/patient-app changes.

- [ ] **Step 5: Do not merge**

Hand the PR to the user for review/merge.