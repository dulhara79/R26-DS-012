# ClinAnx P5A Signals, Contributions & Data Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two clinician deep views — Signals & Contributions and Data Quality — as read-only projections of the exact `AssessmentSummary` already loaded by Patient Overview, with no new backend routes, no second assessment authority, and no local reconstruction of fusion or missing values.

**Architecture:** `PatientOverviewController` remains the sole loader for the current assessment. `PatientOverviewScreen` passes the same immutable `AssessmentSummary` instance into two stateless deep-view screens. Shared presentation helpers provide consistent labels for C1-C4, assessment state, modality state, inclusion state, missing values, and timestamps; they do not compute clinical outputs.

**Tech Stack:** Flutter >=3.27.0, Dart >=3.5.0 <4.0.0, existing ClinAnx design components, `intl ^0.19.0`, `flutter_test`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-clinanx-p5a-signals-data-quality-design.md`

## Global Constraints

- One backend-owned `AssessmentSummary` is the source of truth for Patient Overview, Signals & Contributions, and Data Quality.
- The deep screens receive the same `AssessmentSummary` object already displayed by Patient Overview; they do not fetch a second assessment on navigation.
- Show only backend-provided values from `AssessmentSummary` / `ModalityStatus`.
- Do not infer or reconstruct fusion weights, contribution values, risk scores, tiers, completeness, exclusion reasons, or freshness reasons.
- Missing numeric values render as `—` or `Not reported`, never `0`.
- Missing/stale/unavailable data never becomes Low or reassuring output.
- C3 remains labelled `Clinical NLP / TC-WPN` and subordinate to the multimodal assessment.
- C2 must explicitly show experimental/excluded semantics when `state == ModalityState.notValidated && includedInFusion == false`.
- No direct model `/predict` call, no guessed `/contributions` or `/data-quality` route, and no legacy `FusionResult` authority may be introduced into P5A.
- P5A performs no backend mutation and does not change authentication, assignment, attention-event, notification, timeline, note, evidence, or clinician-assessment behavior.
- No teammate-owned repository is modified.
- Preserve the known unrelated whole-suite baseline failure truthfully if it still exists; focused P0-P5A gates are blocking, full-suite visibility must not be described as green unless it is actually green.

---

## File Structure

**Create**
- `lib/features/patients/assessment_detail_presentation.dart` — presentation-only labels/formatters shared by the two deep views and Patient Overview where useful.
- `lib/features/patients/signals_contributions_screen.dart` — read-only explanation of C1-C4 server signals and server-provided contribution values.
- `lib/features/patients/data_quality_screen.dart` — read-only assessment/modality availability, state, confidence, coverage, capture time, and inclusion view.
- `test/features/patients/signals_contributions_screen_test.dart`
- `test/features/patients/data_quality_screen_test.dart`
- `test/p5a_assessment_authority_test.dart`
- `docs/superpowers/verification/2026-09-16-clinanx-p5a-signals-data-quality-verification.md` after final verification.

**Modify**
- `lib/features/patients/patient_overview_screen.dart` — add links and default navigation while preserving the loaded assessment object.
- `test/features/patients/patient_overview_screen_test.dart` — prove object identity and no second assessment fetch.
- `.github/workflows/p1_p2_dashboard_ci.yml` — run P5A branch and include P5A focused tests/analyzer/authority checks.

**Do not modify for P5A authority**
- `lib/domain/contracts/assessment_summary.dart` unless execution exposes a genuine contract-parsing defect already supported by the verified payload; no new speculative fields are added.
- `lib/features/fusion/fusion_detail_screen.dart` as a source of truth.
- `lib/data/api/gateways.dart` for new P5A endpoints.
- `lib/data/repositories/central_backend_repositories.dart` for new contribution/data-quality routes.

---

### Task 1: Enable the branch CI harness before writing feature code

**Files:**
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`

**Interfaces:**
- Consumes: existing P0-P4 focused CI job.
- Produces: branch-triggered CI for `integration/clinanx-p5a-signals-data-quality`, with the whole patient-feature test directory automatically included so newly added P5A widget tests are picked up immediately.

- [ ] **Step 1: Add the P5A branch trigger**

Under `on.push.branches`, add exactly:

```yaml
      - integration/clinanx-p5a-signals-data-quality
```

- [ ] **Step 2: Make patient feature tests directory-scoped**

Replace these two separate focused-test entries:

```yaml
          test/features/patients/patient_overview_screen_test.dart
          test/features/patients/patients_screen_navigation_test.dart
```

with:

```yaml
          test/features/patients
```

This allows RED tests created later in P5A to execute without another workflow edit.

- [ ] **Step 3: Keep focused analysis limited to files that already exist**

Do not add `signals_contributions_screen.dart`, `data_quality_screen.dart`, or `p5a_assessment_authority_test.dart` yet because those files do not exist. The final CI expansion happens in Task 5 after creation.

- [ ] **Step 4: Commit the CI harness change**

```bash
git add .github/workflows/p1_p2_dashboard_ci.yml
git commit -m "ci: enable P5A branch verification"
```

Expected GitHub Actions result at this stage: current focused P0-P4 checks run on the P5A branch. Any pre-existing full-suite visibility failure remains non-blocking and must be recorded, not hidden.

---

### Task 2: Add shared presentation helpers and Signals & Contributions with TDD

**Files:**
- Create: `test/features/patients/signals_contributions_screen_test.dart`
- Create: `lib/features/patients/assessment_detail_presentation.dart`
- Create: `lib/features/patients/signals_contributions_screen.dart`

**Interfaces:**
- Consumes: `AssessmentSummary`, `ModalityStatus`, `AssessmentStatus`, `ModalityState`, existing `Panel`, `SectionLabel`, `InlineNotice`, `DecisionSupportNotice`, `AppTheme`, `Ds`.
- Produces:

```dart
const p5aComponentOrder = <String>[
  'c1_physiological',
  'c2_behavioral',
  'c3_clinical_nlp',
  'c4_demographic',
];

String p5aModalityLabel(String componentId);
String p5aAssessmentStatusLabel(AssessmentStatus status);
String p5aModalityStateLabel(ModalityStatus? modality);
String p5aInclusionLabel(bool? includedInFusion);
String p5aScoreLabel(double? value);
String p5aReportedDecimal(double? value);
String p5aReportedTime(DateTime? value);

class SignalsContributionsScreen extends StatelessWidget {
  final AssessmentSummary assessment;
  final String? displayId;

  const SignalsContributionsScreen({
    super.key,
    required this.assessment,
    this.displayId,
  });
}
```

- [ ] **Step 1: Write the failing Signals & Contributions widget tests**

Create `test/features/patients/signals_contributions_screen_test.dart` with a fixture that includes all four modalities and a second fixture with missing/partial fields. Use the same wire meanings already established in P3:

```dart
AssessmentSummary completeAssessment() => AssessmentSummary(
  subjectId: 'subject-001',
  fusionResultId: 123,
  currentAssessment: const CurrentAssessment(
    score: 0.58,
    tier: RiskTier.medium,
    band: 'AMBER',
  ),
  forecast: null,
  confidence: 0.71,
  uncertainty: 0.18,
  assessmentStatus: AssessmentStatus.complete,
  modalities: [
    ModalityStatus(
      componentId: 'c1_physiological',
      score: 0.82,
      available: true,
      includedInFusion: true,
      state: ModalityState.ok,
      confidence: 0.50,
      coverage: 0.50,
      capturedAt: DateTime.utc(2026, 9, 16, 7, 59),
      contribution: 0.24,
    ),
    const ModalityStatus(
      componentId: 'c2_behavioral',
      score: null,
      available: false,
      includedInFusion: false,
      state: ModalityState.notValidated,
      confidence: null,
      coverage: null,
      capturedAt: null,
      contribution: null,
    ),
    ModalityStatus(
      componentId: 'c3_clinical_nlp',
      score: 0.67,
      available: true,
      includedInFusion: true,
      state: ModalityState.ok,
      confidence: 0.61,
      coverage: 1.0,
      capturedAt: DateTime.utc(2026, 9, 16, 7, 30),
      contribution: 0.22,
    ),
    ModalityStatus(
      componentId: 'c4_demographic',
      score: 0.43,
      available: true,
      includedInFusion: true,
      state: ModalityState.ok,
      confidence: 0.70,
      coverage: 1.0,
      capturedAt: DateTime.utc(2026, 9, 1, 8),
      contribution: 0.12,
    ),
  ],
  computedAt: DateTime.utc(2026, 9, 16, 8),
  modelVersion: 'ragf-v0.4',
);
```

Required test names and assertions:

```dart
testWidgets('shows canonical assessment provenance', (tester) async { ... });
testWidgets('renders C1 C2 C3 C4 in canonical order', (tester) async { ... });
testWidgets('renders exact server scores and contributions', (tester) async { ... });
testWidgets('missing contribution is Not reported and no weight is invented', (tester) async { ... });
testWidgets('C3 remains Clinical NLP signal and never overall risk', (tester) async { ... });
testWidgets('C2 experimental exclusion is explicit', (tester) async { ... });
testWidgets('missing modality gets an explicit unavailable card', (tester) async { ... });
testWidgets('unknown and stale states remain explicit', (tester) async { ... });
testWidgets('null score never becomes zero', (tester) async { ... });
```

For order, compare vertical positions after scrolling each label into view:

```dart
final c1Y = tester.getTopLeft(find.text('Physiological').first).dy;
final c2Y = tester.getTopLeft(find.text('Behavioural').first).dy;
final c3Y = tester.getTopLeft(find.text('Clinical NLP / TC-WPN').first).dy;
final c4Y = tester.getTopLeft(find.text('Contextual').first).dy;
expect(c1Y, lessThan(c2Y));
expect(c2Y, lessThan(c3Y));
expect(c3Y, lessThan(c4Y));
```

For missing contribution, assert:

```dart
expect(find.text('Contribution: Not reported'), findsWidgets);
expect(find.textContaining('weight'), findsNothing);
```

For C3 authority, assert:

```dart
expect(find.text('Clinical NLP / TC-WPN'), findsOneWidget);
expect(find.text('TC-WPN Risk'), findsNothing);
expect(find.textContaining('Overall TC-WPN'), findsNothing);
```

- [ ] **Step 2: Run the new test and verify RED**

```bash
flutter test test/features/patients/signals_contributions_screen_test.dart
```

Expected: compile failure because `SignalsContributionsScreen` does not exist.

- [ ] **Step 3: Add presentation-only helper functions**

Create `lib/features/patients/assessment_detail_presentation.dart` with exact state mappings:

```dart
import 'package:intl/intl.dart';

import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/contract_enums.dart';

const p5aComponentOrder = <String>[
  'c1_physiological',
  'c2_behavioral',
  'c3_clinical_nlp',
  'c4_demographic',
];

String p5aModalityLabel(String componentId) => switch (componentId) {
  'c1_physiological' => 'Physiological',
  'c2_behavioral' => 'Behavioural',
  'c3_clinical_nlp' => 'Clinical NLP / TC-WPN',
  'c4_demographic' => 'Contextual',
  _ => componentId,
};

String p5aAssessmentStatusLabel(AssessmentStatus status) => switch (status) {
  AssessmentStatus.complete => 'Complete assessment',
  AssessmentStatus.partial => 'Partial assessment',
  AssessmentStatus.unavailable => 'Assessment unavailable',
  AssessmentStatus.unknown => 'Assessment status unknown',
};

String p5aModalityStateLabel(ModalityStatus? modality) {
  if (modality == null) return 'Unavailable';
  if (modality.isExperimentalExcluded) {
    return 'Experimental — not included in fusion';
  }
  return switch (modality.state) {
    ModalityState.ok => 'Available',
    ModalityState.stale => 'Stale',
    ModalityState.notValidated => 'Experimental / not validated',
    ModalityState.unavailable => 'Unavailable',
    ModalityState.error => 'Service error',
    ModalityState.unknown => 'Status unknown',
  };
}

String p5aInclusionLabel(bool? includedInFusion) => switch (includedInFusion) {
  true => 'Included in fusion',
  false => 'Not included in fusion',
  null => 'Inclusion not reported',
};

String p5aScoreLabel(double? value) =>
    value == null || !value.isFinite ? '—' : value.toStringAsFixed(2);

String p5aReportedDecimal(double? value) =>
    value == null || !value.isFinite ? 'Not reported' : value.toStringAsFixed(2);

String p5aReportedTime(DateTime? value) => value == null
    ? 'Not reported'
    : DateFormat('d MMM y, HH:mm').format(value.toLocal());
```

These helpers format values only. They must not compute tier, contribution, confidence, completeness, or freshness.

- [ ] **Step 4: Implement SignalsContributionsScreen minimally**

Use `ListView` with:

1. App bar title `Signals & contributions`.
2. Participant display ID if provided, plus canonical `subjectId`.
3. `SectionLabel('Assessment provenance')` with a `Panel` containing:
   - `Fusion result #123` or `Fusion result ID not reported`;
   - assessment status label;
   - `Computed ...` or `Computed: Not reported`;
   - `Model ...` or `Model: Not reported`.
4. `SectionLabel('Signals & contributions')`.
5. Four cards in `p5aComponentOrder`, even when a modality record is missing.
6. A contribution interpretation `InlineNotice`.
7. `DecisionSupportNotice`.

Create a local lookup once:

```dart
final byId = {
  for (final modality in assessment.modalities)
    modality.componentId: modality,
};
```

For each ordered component, pass `byId[componentId]` into the card. The card renders:

```text
<modality label>
Score: <server score or —>
Status: <typed state label>
Fusion: <inclusion label>
Contribution: <server contribution or Not reported>
Captured: <server time or Not reported>
```

Do not add a `weight` row.

Use this exact interpretation copy:

```text
Contribution reflects the value reported by the fusion backend for this assessment. It does not establish that a modality caused the patient's state.
```

- [ ] **Step 5: Run Signals tests GREEN**

```bash
flutter test test/features/patients/signals_contributions_screen_test.dart
```

Expected: PASS.

- [ ] **Step 6: Run analyzer for the new files**

```bash
flutter analyze \
  lib/features/patients/assessment_detail_presentation.dart \
  lib/features/patients/signals_contributions_screen.dart \
  test/features/patients/signals_contributions_screen_test.dart
```

Expected: `No issues found!`

- [ ] **Step 7: Commit**

```bash
git add \
  lib/features/patients/assessment_detail_presentation.dart \
  lib/features/patients/signals_contributions_screen.dart \
  test/features/patients/signals_contributions_screen_test.dart
git commit -m "feat: add P5A signals and contributions view"
```

---

### Task 3: Build Data Quality with TDD

**Files:**
- Create: `test/features/patients/data_quality_screen_test.dart`
- Create: `lib/features/patients/data_quality_screen.dart`
- Reuse: `lib/features/patients/assessment_detail_presentation.dart`

**Interfaces:**
- Consumes: same `AssessmentSummary` object and P5A presentation helpers.
- Produces:

```dart
class DataQualityScreen extends StatelessWidget {
  final AssessmentSummary assessment;
  final String? displayId;

  const DataQualityScreen({
    super.key,
    required this.assessment,
    this.displayId,
  });
}
```

- [ ] **Step 1: Write failing Data Quality tests**

Required tests:

```dart
testWidgets('shows assessment status exactly as supplied', (tester) async { ... });
testWidgets('shows availability and fusion inclusion as separate fields', (tester) async { ... });
testWidgets('shows exact confidence coverage and captured time when reported', (tester) async { ... });
testWidgets('missing values are Not reported and never zero', (tester) async { ... });
testWidgets('stale unavailable error and unknown remain distinct', (tester) async { ... });
testWidgets('C2 experimental exclusion stays explicit', (tester) async { ... });
testWidgets('missing modality record remains visibly unavailable', (tester) async { ... });
testWidgets('reason detail is not fabricated', (tester) async { ... });
```

Construct separate assessments for `AssessmentStatus.complete`, `partial`, `unavailable`, and `unknown`, and assert their exact labels from `p5aAssessmentStatusLabel`.

For availability vs inclusion, use a modality that is available but excluded:

```dart
const ModalityStatus(
  componentId: 'c2_behavioral',
  score: 0.40,
  available: true,
  includedInFusion: false,
  state: ModalityState.notValidated,
  confidence: null,
  coverage: null,
  capturedAt: null,
  contribution: null,
)
```

Assert both concepts are visible:

```dart
expect(find.textContaining('Experimental'), findsWidgets);
expect(find.text('Not included in fusion'), findsWidgets);
```

For missing values:

```dart
expect(find.text('Score: —'), findsWidgets);
expect(find.text('Confidence: Not reported'), findsWidgets);
expect(find.text('Coverage: Not reported'), findsWidgets);
expect(find.text('Captured: Not reported'), findsWidgets);
expect(find.text('Reason detail: Not reported by backend'), findsWidgets);
expect(find.text('0.00'), findsNothing);
```

- [ ] **Step 2: Run Data Quality test and verify RED**

```bash
flutter test test/features/patients/data_quality_screen_test.dart
```

Expected: compile failure because `DataQualityScreen` does not exist.

- [ ] **Step 3: Implement DataQualityScreen minimally**

Use `ListView` with:

1. App bar title `Data quality`.
2. Participant display ID and canonical `subjectId`.
3. Assessment status panel with:

```text
Assessment: Complete assessment / Partial assessment / Assessment unavailable / Assessment status unknown
Fusion result: #<id> / Not reported
Computed: <time> / Not reported
Model: <version> / Not reported
```

4. `SectionLabel('Modality quality')`.
5. Four cards in `p5aComponentOrder`.
6. `DecisionSupportNotice`.

Each modality card renders independent fields:

```dart
String availabilityLabel(ModalityStatus? modality) {
  if (modality == null) return 'Unavailable';
  return switch (modality.available) {
    true => 'Available',
    false => 'Unavailable',
    null => 'Availability not reported',
  };
}
```

Then display:

```text
Availability: ...
Status: ...
Score: ...
Confidence: ...
Coverage: ...
Captured: ...
Fusion inclusion: ...
Reason detail: Not reported by backend
```

`Reason detail` is intentionally neutral because no free-form reason exists in the current verified `ModalityStatus` contract.

- [ ] **Step 4: Run Data Quality tests GREEN**

```bash
flutter test test/features/patients/data_quality_screen_test.dart
```

Expected: PASS.

- [ ] **Step 5: Run analyzer**

```bash
flutter analyze \
  lib/features/patients/data_quality_screen.dart \
  lib/features/patients/assessment_detail_presentation.dart \
  test/features/patients/data_quality_screen_test.dart
```

Expected: `No issues found!`

- [ ] **Step 6: Commit**

```bash
git add \
  lib/features/patients/data_quality_screen.dart \
  test/features/patients/data_quality_screen_test.dart
git commit -m "feat: add P5A data quality view"
```

---

### Task 4: Wire Patient Overview navigation while preserving object identity

**Files:**
- Modify: `test/features/patients/patient_overview_screen_test.dart`
- Modify: `lib/features/patients/patient_overview_screen.dart`

**Interfaces:**
- Consumes: loaded `AssessmentSummary` in `PatientOverviewController.state.data`.
- Produces optional test/injection callbacks and production default navigation:

```dart
final ValueChanged<AssessmentSummary>? onOpenSignalsContributions;
final ValueChanged<AssessmentSummary>? onOpenDataQuality;
```

Both constructors accept the callbacks optionally. Production defaults navigate to:

```dart
SignalsContributionsScreen(
  assessment: assessment,
  displayId: widget.displayId,
)
```

and:

```dart
DataQualityScreen(
  assessment: assessment,
  displayId: widget.displayId,
)
```

- [ ] **Step 1: Add repository call counting to the existing test fake**

Change the local `_Repository` test fake so `latestAssessment` increments a counter:

```dart
class _Repository implements AssessmentRepository {
  final AssessmentSummary? assessment;
  int latestAssessmentCalls = 0;

  _Repository(this.assessment);

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    latestAssessmentCalls++;
    return assessment;
  }
}
```

Adjust `_pumpOverview` to return both the controller/repository or otherwise make the fake available to navigation tests.

- [ ] **Step 2: Write failing navigation tests**

Add:

```dart
testWidgets('shows both P5A deep-view actions for a loaded assessment', (tester) async { ... });
testWidgets('signals action receives the exact loaded AssessmentSummary object', (tester) async { ... });
testWidgets('data quality action receives the exact loaded AssessmentSummary object', (tester) async { ... });
testWidgets('opening a P5A deep view does not fetch a second assessment', (tester) async { ... });
```

For object identity:

```dart
AssessmentSummary? received;

await tester.pumpWidget(MaterialApp(
  home: PatientOverviewScreen(
    displayId: 'Patient A',
    controller: controller,
    onOpenSignalsContributions: (value) => received = value,
  ),
));

await tester.scrollUntilVisible(
  find.text('View signals & contributions'),
  250,
  scrollable: find.byType(Scrollable).first,
);
await tester.tap(find.text('View signals & contributions'));

expect(identical(received, assessment), isTrue);
```

Do the same for Data Quality. Assert `repository.latestAssessmentCalls == 1` after tapping the deep-view action.

- [ ] **Step 3: Run Patient Overview tests and verify RED**

```bash
flutter test test/features/patients/patient_overview_screen_test.dart
```

Expected: failure because the new callback fields/actions do not exist.

- [ ] **Step 4: Add callback fields and imports**

Import:

```dart
import 'data_quality_screen.dart';
import 'signals_contributions_screen.dart';
```

Add the two optional `ValueChanged<AssessmentSummary>` callback fields to both constructors without changing the existing `subjectId` / injected-controller behavior.

- [ ] **Step 5: Add Signals deep-view action**

Change `_SignalsSection` to accept a `VoidCallback onOpenDetails` and append a `TextButton`:

```dart
TextButton.icon(
  onPressed: onOpenDetails,
  icon: const Icon(Icons.arrow_forward_rounded, size: 16),
  label: const Text('View signals & contributions'),
)
```

Pass:

```dart
onOpenDetails: () {
  final callback = widget.onOpenSignalsContributions;
  if (callback != null) {
    callback(assessment);
    return;
  }
  Navigator.push(
    context,
    MaterialPageRoute(
      builder: (_) => SignalsContributionsScreen(
        assessment: assessment,
        displayId: widget.displayId,
      ),
    ),
  );
},
```

- [ ] **Step 6: Add compact Data Quality summary and action**

Create a private `_DataQualitySummary` widget that receives only:

```dart
final AssessmentSummary assessment;
final VoidCallback onOpenDetails;
```

Render the server-derived assessment status and a `View data quality` button. Do not recompute completeness from modality count.

The callback uses the same pattern as Signals and pushes `DataQualityScreen(assessment: assessment, displayId: widget.displayId)`.

Place the summary after `_SignalsSection` and before `DecisionSupportNotice`.

- [ ] **Step 7: Run Patient Overview tests GREEN**

```bash
flutter test test/features/patients/patient_overview_screen_test.dart
```

Expected: PASS, including all prior P3 semantics and new identity/navigation tests.

- [ ] **Step 8: Run all patient feature tests**

```bash
flutter test test/features/patients
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add \
  lib/features/patients/patient_overview_screen.dart \
  test/features/patients/patient_overview_screen_test.dart
git commit -m "feat: link Patient Overview to P5A deep views"
```

---

### Task 5: Add P5A authority regression tests and complete CI coverage

**Files:**
- Create: `test/p5a_assessment_authority_test.dart`
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`

**Interfaces:**
- Consumes: P5A screen source files and existing authority-test pattern from P3/P4.
- Produces: structural guardrails against endpoint guessing, direct prediction, legacy authority, and local reconstruction.

- [ ] **Step 1: Write the authority test**

Create `test/p5a_assessment_authority_test.dart`:

```dart
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  final signalsPath =
      'lib/features/patients/signals_contributions_screen.dart';
  final qualityPath = 'lib/features/patients/data_quality_screen.dart';
  final overviewPath = 'lib/features/patients/patient_overview_screen.dart';

  test('P5A deep views use AssessmentSummary and not legacy FusionResult authority', () {
    final signals = File(signalsPath).readAsStringSync();
    final quality = File(qualityPath).readAsStringSync();

    expect(signals, contains('AssessmentSummary'));
    expect(quality, contains('AssessmentSummary'));
    expect(signals, isNot(contains('FusionResult')));
    expect(quality, isNot(contains('FusionResult')));
  });

  test('P5A deep views contain no direct model or guessed deep-view route', () {
    final source = [
      File(signalsPath).readAsStringSync(),
      File(qualityPath).readAsStringSync(),
      File(overviewPath).readAsStringSync(),
    ].join('\n');

    expect(source, isNot(contains("'/predict'")));
    expect(source, isNot(contains('"/predict"')));
    expect(source, isNot(contains('/contributions')));
    expect(source, isNot(contains('/data-quality')));
    expect(source, isNot(contains('CentralBackendGateway')));
    expect(source, isNot(contains('AssessmentRepository')));
  });

  test('P5A does not locally derive risk or missing contribution', () {
    final source = [
      File(signalsPath).readAsStringSync(),
      File(qualityPath).readAsStringSync(),
    ].join('\n');

    expect(source, isNot(contains('AlertBandX.fromScore')));
    expect(source, isNot(contains('weight * score')));
    expect(source, isNot(contains('contribution =')));
    expect(source, isNot(contains('.reduce(')));
    expect(source, isNot(contains('.fold(')));
  });

  test('P5A preserves C3 subordinate wording and explicit missing values', () {
    final source = [
      File(signalsPath).readAsStringSync(),
      File(qualityPath).readAsStringSync(),
      File('lib/features/patients/assessment_detail_presentation.dart')
          .readAsStringSync(),
    ].join('\n');

    expect(source, contains('Clinical NLP / TC-WPN'));
    expect(source, contains('Not reported'));
    expect(source, isNot(contains('Overall TC-WPN risk')));
  });
}
```

The `AssessmentRepository` assertion applies only to deep-view source files; if it is present in Patient Overview for its existing controller wiring, keep Patient Overview out of that particular joined string.

- [ ] **Step 2: Run authority test**

```bash
flutter test test/p5a_assessment_authority_test.dart
```

Expected: PASS. If it fails because the implementation performs a prohibited action, fix the implementation rather than weakening the guard.

- [ ] **Step 3: Expand focused CI tests**

In `.github/workflows/p1_p2_dashboard_ci.yml`, add:

```yaml
          test/p5a_assessment_authority_test.dart
```

after the P4 authority test. Keep `test/features/patients` as the directory-scoped patient widget gate.

- [ ] **Step 4: Expand focused analysis**

Add these implementation files:

```yaml
          lib/features/patients/assessment_detail_presentation.dart
          lib/features/patients/signals_contributions_screen.dart
          lib/features/patients/data_quality_screen.dart
```

and these tests:

```yaml
          test/p5a_assessment_authority_test.dart
          test/features/patients/signals_contributions_screen_test.dart
          test/features/patients/data_quality_screen_test.dart
```

- [ ] **Step 5: Expand shell grep guards for P5A**

Add a new block:

```bash
if grep -R -nE "['\"]/?predict['\"]|/contributions|/data-quality|AlertBandX\.fromScore" \
  lib/features/patients/signals_contributions_screen.dart \
  lib/features/patients/data_quality_screen.dart 2>/dev/null; then
  echo 'P5A deep views must not call models, guess routes, or derive risk locally.'
  exit 1
fi
```

Do not grep generic words such as `contribution` because the screen legitimately displays the backend-provided `contribution` field.

- [ ] **Step 6: Run focused local/remote gate**

```bash
flutter test \
  test/contracts \
  test/repositories \
  test/p3_patient_overview_authority_test.dart \
  test/p4_attention_event_authority_test.dart \
  test/p5a_assessment_authority_test.dart \
  test/features/patients
```

Then:

```bash
flutter analyze \
  lib/domain/contracts \
  lib/features/patients/assessment_detail_presentation.dart \
  lib/features/patients/patient_overview_screen.dart \
  lib/features/patients/signals_contributions_screen.dart \
  lib/features/patients/data_quality_screen.dart \
  test/p3_patient_overview_authority_test.dart \
  test/p4_attention_event_authority_test.dart \
  test/p5a_assessment_authority_test.dart \
  test/features/patients
```

Expected: all focused tests pass; analyzer reports no issues.

- [ ] **Step 7: Commit**

```bash
git add \
  test/p5a_assessment_authority_test.dart \
  .github/workflows/p1_p2_dashboard_ci.yml
git commit -m "test: add P5A assessment authority gate"
```

---

### Task 6: Final verification, diff review, and PR readiness

**Files:**
- Create: `docs/superpowers/verification/2026-09-16-clinanx-p5a-signals-data-quality-verification.md`
- Review only: all P5A changed files.

**Interfaces:**
- Consumes: completed P5A branch and GitHub Actions evidence.
- Produces: auditable verification record and a PR-ready branch; this task does not merge.

- [ ] **Step 1: Run focused P5A verification**

Run or verify in GitHub Actions:

```bash
flutter test \
  test/contracts \
  test/repositories \
  test/p3_patient_overview_authority_test.dart \
  test/p4_attention_event_authority_test.dart \
  test/p5a_assessment_authority_test.dart \
  test/features/patients
```

Record the exact passed-test count from the log. Do not estimate it.

- [ ] **Step 2: Run focused analyzer**

```bash
flutter analyze \
  lib/domain/contracts \
  lib/features/patients/assessment_detail_presentation.dart \
  lib/features/patients/patient_overview_screen.dart \
  lib/features/patients/signals_contributions_screen.dart \
  lib/features/patients/data_quality_screen.dart \
  test/p3_patient_overview_authority_test.dart \
  test/p4_attention_event_authority_test.dart \
  test/p5a_assessment_authority_test.dart \
  test/features/patients
```

Record the analyzer result verbatim.

- [ ] **Step 3: Run whole-suite visibility**

```bash
flutter test
flutter analyze
```

If the known `test/widget_test.dart` consent/sign-in baseline still fails, report the exact pass/fail count and exact failing test. Do not call the full suite green. If it is fixed independently, report that new fact with fresh logs.

- [ ] **Step 4: Review branch diff against base**

Compare:

```text
base: 02cff6a5048e89362e34fd6426adb2527c2ae1d6
head: current integration/clinanx-p5a-signals-data-quality
```

Confirm the diff is limited to:

```text
.github/workflows/p1_p2_dashboard_ci.yml
docs/superpowers/specs/2026-09-16-clinanx-p5a-signals-data-quality-design.md
docs/superpowers/plans/2026-09-16-clinanx-p5a-signals-data-quality.md
docs/superpowers/verification/2026-09-16-clinanx-p5a-signals-data-quality-verification.md
lib/features/patients/assessment_detail_presentation.dart
lib/features/patients/patient_overview_screen.dart
lib/features/patients/signals_contributions_screen.dart
lib/features/patients/data_quality_screen.dart
test/features/patients/patient_overview_screen_test.dart
test/features/patients/signals_contributions_screen_test.dart
test/features/patients/data_quality_screen_test.dart
test/p5a_assessment_authority_test.dart
```

If another file changed, inspect and justify it before PR creation.

- [ ] **Step 5: Write verification record**

The verification markdown must include:

- base commit and final head commit;
- GitHub Actions run ID/job ID used as evidence;
- focused test count/result;
- focused analyzer result;
- authority checks result;
- full-suite test count/result;
- full analyzer result;
- explicit statement that no guessed P5A backend endpoint was introduced;
- explicit statement that both deep views receive the same loaded `AssessmentSummary` and do not create a second fetch boundary;
- known baseline failure, if still present.

- [ ] **Step 6: Commit verification doc**

```bash
git add docs/superpowers/verification/2026-09-16-clinanx-p5a-signals-data-quality-verification.md
git commit -m "docs: record ClinAnx P5A verification"
```

Because this documentation-only commit creates a new branch head, verify the final-head CI run before making completion claims.

- [ ] **Step 7: Prepare PR but do not merge**

PR title:

```text
Add ClinAnx P5A Signals and Data Quality Views
```

PR body must state:

- two distinct deep views were added;
- both are read-only projections of the same canonical `AssessmentSummary` from Patient Overview;
- no new backend route was guessed or wired;
- no local weight, contribution, tier, risk, or completeness reconstruction was added;
- C2/C3 semantics and missing-data behavior;
- exact focused test/analyzer evidence;
- exact whole-suite visibility result and any known unrelated baseline failure.

Do **not** merge the PR without explicit user approval.

---

## Self-Review Checklist

Before execution, verify this plan against the approved spec:

- [ ] Two separate screens, not one combined screen or tabbed screen.
- [ ] Same `AssessmentSummary` instance passed from Patient Overview.
- [ ] No second network fetch in deep views.
- [ ] C1-C4 canonical order with explicit placeholder for missing modality records.
- [ ] Contribution shown only when backend supplied it.
- [ ] No weight field or reconstructed weight.
- [ ] Data Quality keeps availability, modality state, and fusion inclusion separate.
- [ ] Missing score/confidence/coverage/capture/contribution never becomes zero.
- [ ] Missing free-form reason is labelled `Not reported by backend` rather than invented.
- [ ] C2 experimental/excluded semantics retained.
- [ ] C3 remains `Clinical NLP / TC-WPN`, never overall risk.
- [ ] No P5B/P5C scope included.
- [ ] Existing P3/P4 authority gates remain in the focused verification set.
- [ ] PR is opened only after final-head verification and is not merged automatically.
