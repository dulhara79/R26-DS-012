# ClinAnx P0 Contract-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, typed ClinAnx P0 contract layer for canonical assessment, forecast, modality, patient-summary and attention-event data, and remove the verified `/v1/subjects/attach` client/backend mismatch without changing fusion mathematics or final UI behavior.

**Architecture:** Keep the existing central-backend-only integration boundary. Add new P0 contract models under `lib/domain/contracts/` instead of expanding the existing 55k-line `models.dart`, exercise them against frozen JSON fixtures, and preserve nullable/unknown states rather than inventing low-risk defaults. Resolve Aura participant identity through the backend's existing `/v1/subjects/resolve` contract and retain the current local alert path until server `AttentionEvent` APIs exist.

**Tech Stack:** Flutter >=3.27.0, Dart >=3.5.0 <4.0.0, `flutter_test`, `http`, `provider`, existing `ApiClient`/`CentralBackendGateway`.

**Spec:** `docs/superpowers/specs/2026-09-15-clinanx-p0-contract-first-design.md`

## Global Constraints

- Central backend remains the authoritative owner of current `FusionResult`.
- No client-side authoritative fusion calculation is added.
- Current assessment and forecast remain separate domain objects and fields.
- Forecast scope is explicit; the current near-term forecast is represented as `physiological` unless a future server contract explicitly says otherwise.
- Missing, stale, insufficient or unknown data never becomes zero/low risk.
- C2 remains experimental/excluded from active fusion.
- TC-WPN/C3 remains a contributing Clinical NLP signal, not the primary patient assessment.
- Unknown wire enum values map to explicit `unknown` states.
- ClinAnx continues to call TC-WPN only indirectly through the central backend for authoritative inference.
- The local `ClinicalAlert` path is not removed in this slice because the central backend does not yet expose persistent `AttentionEvent` APIs.
- No teammate-owned repository is modified in this plan.
- Do not add target attention/dashboard HTTP calls until the central backend owner freezes those routes.

---

## File Structure Locked by This Plan

New production files:

```text
lib/domain/contracts/
  contract_parsing.dart      # safe nullable primitive/date helpers
  contract_enums.dart        # typed wire enums with explicit unknown states
  assessment_summary.dart    # CurrentAssessment, ForecastResult, ModalityStatus, AssessmentSummary
  attention_event.dart       # AttentionEvent
  patient_summary.dart       # assignment/dashboard patient projection
```

New test files:

```text
test/contracts/
  fixture_loader.dart
  contract_enums_test.dart
  assessment_summary_test.dart
  attention_event_test.dart
  patient_summary_test.dart
  p0_invariants_test.dart

test/fixtures/contracts/
  assessment_complete.json
  assessment_partial_stale_c1.json
  assessment_unavailable_c3.json
  assessment_unknown_enum.json
  attention_event_open.json
  attention_event_acknowledged.json
  attention_event_resolved.json
  patient_summary.json
```

Existing files modified:

```text
lib/data/api/gateways.dart
lib/state/controllers.dart
test/gateway_contract_test.dart
```

The existing `lib/domain/models.dart` remains unchanged in this slice except if a compile-time import conflict forces a minimal import adjustment. The new target contracts are deliberately isolated from legacy/current `FusionResult` parsing so the running timeline path remains stable.

---

### Task 1: Add safe contract parsing primitives and enums

**Files:**
- Create: `lib/domain/contracts/contract_parsing.dart`
- Create: `lib/domain/contracts/contract_enums.dart`
- Create: `test/contracts/contract_enums_test.dart`

**Interfaces:**
- Produces: `contractString`, `contractInt`, `contractDouble`, `contractBool`, `contractDateTime`, `contractMap`, `contractMapList`.
- Produces enums: `RiskTier`, `AssessmentStatus`, `ForecastScope`, `ModalityState`, `AttentionEventStatus`, `AttentionSeverity`.
- Later tasks import these files directly.

- [ ] **Step 1: Write enum tests first**

Create `test/contracts/contract_enums_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

void main() {
  group('RiskTier', () {
    test('parses the documented Low/Medium/High values', () {
      expect(RiskTier.fromWire('Low'), RiskTier.low);
      expect(RiskTier.fromWire('Medium'), RiskTier.medium);
      expect(RiskTier.fromWire('High'), RiskTier.high);
    });

    test('unknown risk tiers do not become low', () {
      expect(RiskTier.fromWire('SEVERE_V2'), RiskTier.unknown);
      expect(RiskTier.fromWire(null), RiskTier.unknown);
    });
  });

  group('AssessmentStatus', () {
    test('parses the target contract', () {
      expect(AssessmentStatus.fromWire('complete'), AssessmentStatus.complete);
      expect(AssessmentStatus.fromWire('partial'), AssessmentStatus.partial);
      expect(
        AssessmentStatus.fromWire('unavailable'),
        AssessmentStatus.unavailable,
      );
    });

    test('maps current backend compatibility terms without changing meaning', () {
      expect(
        AssessmentStatus.fromWire('provisional'),
        AssessmentStatus.partial,
      );
      expect(
        AssessmentStatus.fromWire('insufficient'),
        AssessmentStatus.unavailable,
      );
    });

    test('unknown assessment state stays unknown', () {
      expect(AssessmentStatus.fromWire('future_state'), AssessmentStatus.unknown);
    });
  });

  test('forecast scope is explicit', () {
    expect(ForecastScope.fromWire('physiological'), ForecastScope.physiological);
    expect(ForecastScope.fromWire('multimodal'), ForecastScope.multimodal);
    expect(ForecastScope.fromWire('sensor_fusion_v2'), ForecastScope.unknown);
  });

  test('modality state preserves unavailable/experimental/error semantics', () {
    expect(ModalityState.fromWire('ok'), ModalityState.ok);
    expect(ModalityState.fromWire('stale'), ModalityState.stale);
    expect(ModalityState.fromWire('not_validated'), ModalityState.notValidated);
    expect(ModalityState.fromWire('absent'), ModalityState.unavailable);
    expect(ModalityState.fromWire('error'), ModalityState.error);
    expect(ModalityState.fromWire('degraded_v2'), ModalityState.unknown);
  });

  test('attention lifecycle only accepts documented states', () {
    expect(AttentionEventStatus.fromWire('OPEN'), AttentionEventStatus.open);
    expect(
      AttentionEventStatus.fromWire('ACKNOWLEDGED'),
      AttentionEventStatus.acknowledged,
    );
    expect(
      AttentionEventStatus.fromWire('RESOLVED'),
      AttentionEventStatus.resolved,
    );
    expect(AttentionEventStatus.fromWire('SUPPRESSED'), AttentionEventStatus.unknown);
  });

  test('severity only claims values documented in the current target example', () {
    expect(AttentionSeverity.fromWire('high'), AttentionSeverity.high);
    expect(AttentionSeverity.fromWire('critical'), AttentionSeverity.unknown);
  });
}
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
flutter test test/contracts/contract_enums_test.dart
```

Expected: FAIL because `lib/domain/contracts/contract_enums.dart` does not exist.

- [ ] **Step 3: Add nullable parsing helpers**

Create `lib/domain/contracts/contract_parsing.dart`:

```dart
String? contractString(Object? value) {
  if (value == null) return null;
  final text = value.toString().trim();
  return text.isEmpty ? null : text;
}

int? contractInt(Object? value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  return int.tryParse(value?.toString() ?? '');
}

double? contractDouble(Object? value) {
  if (value is num) return value.toDouble();
  return double.tryParse(value?.toString() ?? '');
}

bool? contractBool(Object? value) => value is bool ? value : null;

DateTime? contractDateTime(Object? value) {
  final raw = contractString(value);
  if (raw == null) return null;
  final hasOffset =
      raw.endsWith('Z') || RegExp(r'[+-]\d{2}:?\d{2}$').hasMatch(raw);
  return DateTime.tryParse(hasOffset ? raw : '${raw}Z')?.toLocal();
}

Map<String, dynamic>? contractMap(Object? value) {
  if (value is! Map) return null;
  return Map<String, dynamic>.from(value);
}

List<Map<String, dynamic>> contractMapList(Object? value) {
  if (value is! List) return const [];
  return value
      .whereType<Map>()
      .map((item) => Map<String, dynamic>.from(item))
      .toList(growable: false);
}
```

The important difference from the legacy helper in `models.dart` is intentional: malformed or absent P0 fields remain `null`; no score or timestamp is fabricated.

- [ ] **Step 4: Add enum implementations**

Create `lib/domain/contracts/contract_enums.dart`:

```dart
enum RiskTier {
  low,
  medium,
  high,
  unknown;

  static RiskTier fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'low' => low,
        'medium' => medium,
        'high' => high,
        _ => unknown,
      };
}

enum AssessmentStatus {
  complete,
  partial,
  unavailable,
  unknown;

  static AssessmentStatus fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'complete' => complete,
        'partial' || 'provisional' => partial,
        'unavailable' || 'insufficient' => unavailable,
        _ => unknown,
      };
}

enum ForecastScope {
  physiological,
  multimodal,
  unknown;

  static ForecastScope fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'physiological' => physiological,
        'multimodal' => multimodal,
        _ => unknown,
      };
}

enum ModalityState {
  ok,
  stale,
  notValidated,
  unavailable,
  error,
  unknown;

  static ModalityState fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'ok' => ok,
        'stale' => stale,
        'not_validated' => notValidated,
        'absent' || 'unavailable' || 'no_support_set' => unavailable,
        'error' => error,
        _ => unknown,
      };
}

enum AttentionEventStatus {
  open,
  acknowledged,
  resolved,
  unknown;

  static AttentionEventStatus fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toUpperCase()) {
        'OPEN' => open,
        'ACKNOWLEDGED' => acknowledged,
        'RESOLVED' => resolved,
        _ => unknown,
      };
}

enum AttentionSeverity {
  high,
  unknown;

  static AttentionSeverity fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'high' => high,
        _ => unknown,
      };
}
```

`AttentionSeverity` intentionally recognizes only `high` because that is the only concrete severity value supplied by the current target documents. New server values remain `unknown` until the contract is explicitly expanded.

- [ ] **Step 5: Run tests**

Run:

```bash
flutter test test/contracts/contract_enums_test.dart
flutter analyze lib/domain/contracts test/contracts/contract_enums_test.dart
```

Expected: PASS with no analyzer errors.

- [ ] **Step 6: Commit Task 1**

```bash
git add lib/domain/contracts/contract_parsing.dart \
  lib/domain/contracts/contract_enums.dart \
  test/contracts/contract_enums_test.dart
git commit -m "feat: add P0 contract vocabulary"
```

---

### Task 2: Add canonical AssessmentSummary, forecast and modality contracts

**Files:**
- Create: `lib/domain/contracts/assessment_summary.dart`
- Create: `test/contracts/fixture_loader.dart`
- Create: `test/contracts/assessment_summary_test.dart`
- Create: `test/fixtures/contracts/assessment_complete.json`
- Create: `test/fixtures/contracts/assessment_partial_stale_c1.json`
- Create: `test/fixtures/contracts/assessment_unavailable_c3.json`
- Create: `test/fixtures/contracts/assessment_unknown_enum.json`

**Interfaces:**
- Consumes: Task 1 parsing helpers and enums.
- Produces: `CurrentAssessment`, `ForecastResult`, `ModalityStatus`, `AssessmentSummary`.
- `AssessmentSummary.fromJson(Map<String, dynamic>)` is the only parser entry point for the canonical latest-assessment shape.

- [ ] **Step 1: Add a fixture loader**

Create `test/contracts/fixture_loader.dart`:

```dart
import 'dart:convert';
import 'dart:io';

Map<String, dynamic> loadContractFixture(String name) {
  final raw = File('test/fixtures/contracts/$name').readAsStringSync();
  return jsonDecode(raw) as Map<String, dynamic>;
}
```

- [ ] **Step 2: Add the canonical complete-assessment fixture**

Create `test/fixtures/contracts/assessment_complete.json`:

```json
{
  "subject_id": "subject-001",
  "fusion_result_id": 123,
  "current_assessment": {
    "score": 0.58,
    "tier": "Medium",
    "band": "AMBER"
  },
  "forecast": {
    "forecast_result_id": "fcst-001",
    "scope": "physiological",
    "horizon_minutes": 10,
    "score": 0.84,
    "tier": "High",
    "escalation_probability": null,
    "escalation_predicted": true,
    "generated_at": "2026-09-15T10:00:00Z",
    "valid_until": "2026-09-15T10:10:00Z"
  },
  "confidence": 0.71,
  "assessment_status": "complete",
  "modalities": [
    {
      "component_id": "c1_physiological",
      "score": 0.82,
      "available": true,
      "included_in_fusion": true,
      "status": "ok",
      "confidence": 0.5,
      "coverage": 0.5,
      "captured_at": "2026-09-15T09:59:00Z",
      "contribution": 0.24
    },
    {
      "component_id": "c2_behavioral",
      "score": null,
      "available": false,
      "included_in_fusion": false,
      "status": "not_validated",
      "confidence": null,
      "coverage": null,
      "captured_at": null,
      "contribution": null
    },
    {
      "component_id": "c3_clinical_nlp",
      "score": 0.67,
      "available": true,
      "included_in_fusion": true,
      "status": "ok",
      "confidence": 0.61,
      "coverage": 1.0,
      "captured_at": "2026-09-15T09:30:00Z",
      "contribution": 0.22
    },
    {
      "component_id": "c4_demographic",
      "score": 0.43,
      "available": true,
      "included_in_fusion": true,
      "status": "ok",
      "confidence": 0.7,
      "coverage": 1.0,
      "captured_at": "2026-09-01T08:00:00Z",
      "contribution": 0.12
    }
  ],
  "computed_at": "2026-09-15T10:00:00Z",
  "model_version": "ragf-v0.4"
}
```

- [ ] **Step 3: Add partial, unavailable and unknown fixtures**

Create `assessment_partial_stale_c1.json` with a `partial` assessment, a nullable/unused stale C1, and C3+C4 still represented separately:

```json
{
  "subject_id": "subject-002",
  "fusion_result_id": 124,
  "current_assessment": {"score": 0.51, "tier": "Medium", "band": "AMBER"},
  "forecast": null,
  "confidence": 0.62,
  "assessment_status": "partial",
  "modalities": [
    {
      "component_id": "c1_physiological",
      "score": 0.79,
      "available": true,
      "included_in_fusion": false,
      "status": "stale",
      "confidence": 0.44,
      "coverage": 0.5,
      "captured_at": "2026-09-15T09:00:00Z",
      "contribution": null
    },
    {
      "component_id": "c2_behavioral",
      "score": null,
      "available": false,
      "included_in_fusion": false,
      "status": "not_validated"
    },
    {
      "component_id": "c3_clinical_nlp",
      "score": 0.60,
      "available": true,
      "included_in_fusion": true,
      "status": "ok"
    },
    {
      "component_id": "c4_demographic",
      "score": 0.43,
      "available": true,
      "included_in_fusion": true,
      "status": "ok"
    }
  ],
  "computed_at": "2026-09-15T10:00:00Z",
  "model_version": "ragf-v0.4"
}
```

Create `assessment_unavailable_c3.json`:

```json
{
  "subject_id": "subject-003",
  "fusion_result_id": 125,
  "current_assessment": {"score": null, "tier": null, "band": "GREY"},
  "forecast": null,
  "confidence": null,
  "assessment_status": "unavailable",
  "modalities": [
    {
      "component_id": "c1_physiological",
      "score": null,
      "available": false,
      "included_in_fusion": false,
      "status": "stale"
    },
    {
      "component_id": "c2_behavioral",
      "score": null,
      "available": false,
      "included_in_fusion": false,
      "status": "not_validated"
    },
    {
      "component_id": "c3_clinical_nlp",
      "score": null,
      "available": false,
      "included_in_fusion": false,
      "status": "no_support_set"
    },
    {
      "component_id": "c4_demographic",
      "score": 0.43,
      "available": true,
      "included_in_fusion": false,
      "status": "ok"
    }
  ],
  "computed_at": "2026-09-15T10:00:00Z",
  "model_version": "ragf-v0.4"
}
```

Create `assessment_unknown_enum.json`:

```json
{
  "subject_id": "subject-004",
  "fusion_result_id": 126,
  "current_assessment": {"score": null, "tier": "Severe", "band": "NEW_BAND"},
  "forecast": {
    "forecast_result_id": "fcst-004",
    "scope": "sensor_fusion_v2",
    "horizon_minutes": 10,
    "score": null,
    "tier": "Severe",
    "escalation_predicted": null
  },
  "confidence": null,
  "assessment_status": "future_state",
  "modalities": [
    {
      "component_id": "c1_physiological",
      "score": null,
      "available": null,
      "included_in_fusion": null,
      "status": "degraded_v2"
    }
  ],
  "computed_at": null,
  "model_version": "ragf-next"
}
```

- [ ] **Step 4: Write failing assessment parser tests**

Create `test/contracts/assessment_summary_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

import 'fixture_loader.dart';

void main() {
  test('complete assessment preserves the authoritative fusion id', () {
    final result = AssessmentSummary.fromJson(
      loadContractFixture('assessment_complete.json'),
    );
    expect(result.subjectId, 'subject-001');
    expect(result.fusionResultId, 123);
    expect(result.currentAssessment.score, 0.58);
    expect(result.currentAssessment.tier, RiskTier.medium);
    expect(result.assessmentStatus, AssessmentStatus.complete);
  });

  test('forecast is a separate physiological object', () {
    final result = AssessmentSummary.fromJson(
      loadContractFixture('assessment_complete.json'),
    );
    expect(result.forecast, isNotNull);
    expect(result.forecast!.scope, ForecastScope.physiological);
    expect(result.forecast!.horizonMinutes, 10);
    expect(result.forecast!.score, 0.84);
    expect(result.forecast!.escalationPredicted, isTrue);
    expect(result.currentAssessment.score, isNot(result.forecast!.score));
  });

  test('C2 remains explicitly excluded', () {
    final result = AssessmentSummary.fromJson(
      loadContractFixture('assessment_complete.json'),
    );
    final c2 = result.modalities.singleWhere(
      (item) => item.componentId == 'c2_behavioral',
    );
    expect(c2.state, ModalityState.notValidated);
    expect(c2.includedInFusion, isFalse);
    expect(c2.score, isNull);
  });

  test('stale C1 remains stale and is not treated as fused evidence', () {
    final result = AssessmentSummary.fromJson(
      loadContractFixture('assessment_partial_stale_c1.json'),
    );
    final c1 = result.modalities.singleWhere(
      (item) => item.componentId == 'c1_physiological',
    );
    expect(result.assessmentStatus, AssessmentStatus.partial);
    expect(c1.state, ModalityState.stale);
    expect(c1.includedInFusion, isFalse);
  });

  test('unavailable C3 stays null instead of becoming zero', () {
    final result = AssessmentSummary.fromJson(
      loadContractFixture('assessment_unavailable_c3.json'),
    );
    final c3 = result.modalities.singleWhere(
      (item) => item.componentId == 'c3_clinical_nlp',
    );
    expect(result.assessmentStatus, AssessmentStatus.unavailable);
    expect(result.currentAssessment.score, isNull);
    expect(c3.score, isNull);
    expect(c3.state, ModalityState.unavailable);
  });

  test('unknown values remain explicit unknowns', () {
    final result = AssessmentSummary.fromJson(
      loadContractFixture('assessment_unknown_enum.json'),
    );
    expect(result.assessmentStatus, AssessmentStatus.unknown);
    expect(result.currentAssessment.tier, RiskTier.unknown);
    expect(result.forecast!.scope, ForecastScope.unknown);
    expect(result.modalities.single.state, ModalityState.unknown);
    expect(result.currentAssessment.score, isNull);
  });
}
```

- [ ] **Step 5: Run the assessment test and verify it fails**

Run:

```bash
flutter test test/contracts/assessment_summary_test.dart
```

Expected: FAIL because `assessment_summary.dart` does not exist.

- [ ] **Step 6: Implement the assessment models**

Create `lib/domain/contracts/assessment_summary.dart` with these public types and parsing rules:

```dart
import 'contract_enums.dart';
import 'contract_parsing.dart';

class CurrentAssessment {
  final double? score;
  final RiskTier tier;
  final String? band;

  const CurrentAssessment({
    required this.score,
    required this.tier,
    required this.band,
  });

  factory CurrentAssessment.fromJson(Map<String, dynamic> json) =>
      CurrentAssessment(
        score: contractDouble(json['score']),
        tier: RiskTier.fromWire(json['tier']),
        band: contractString(json['band']),
      );
}

class ForecastResult {
  final String? forecastResultId;
  final ForecastScope scope;
  final int? horizonMinutes;
  final double? score;
  final RiskTier tier;
  final double? escalationProbability;
  final bool? escalationPredicted;
  final DateTime? generatedAt;
  final DateTime? validUntil;

  const ForecastResult({
    required this.forecastResultId,
    required this.scope,
    required this.horizonMinutes,
    required this.score,
    required this.tier,
    required this.escalationProbability,
    required this.escalationPredicted,
    required this.generatedAt,
    required this.validUntil,
  });

  factory ForecastResult.fromJson(Map<String, dynamic> json) => ForecastResult(
        forecastResultId: contractString(json['forecast_result_id']),
        scope: ForecastScope.fromWire(json['scope']),
        horizonMinutes: contractInt(json['horizon_minutes']),
        score: contractDouble(json['score']),
        tier: RiskTier.fromWire(json['tier']),
        escalationProbability: contractDouble(json['escalation_probability']),
        escalationPredicted: contractBool(json['escalation_predicted']),
        generatedAt: contractDateTime(json['generated_at']),
        validUntil: contractDateTime(json['valid_until']),
      );
}

class ModalityStatus {
  final String componentId;
  final double? score;
  final bool? available;
  final bool? includedInFusion;
  final ModalityState state;
  final double? confidence;
  final double? coverage;
  final DateTime? capturedAt;
  final double? contribution;

  const ModalityStatus({
    required this.componentId,
    required this.score,
    required this.available,
    required this.includedInFusion,
    required this.state,
    required this.confidence,
    required this.coverage,
    required this.capturedAt,
    required this.contribution,
  });

  bool get isExperimentalExcluded =>
      state == ModalityState.notValidated && includedInFusion == false;

  factory ModalityStatus.fromJson(Map<String, dynamic> json) => ModalityStatus(
        componentId: contractString(json['component_id']) ?? '',
        score: contractDouble(json['score']),
        available: contractBool(json['available']),
        includedInFusion: contractBool(json['included_in_fusion']),
        state: ModalityState.fromWire(json['status']),
        confidence: contractDouble(json['confidence']),
        coverage: contractDouble(json['coverage']),
        capturedAt: contractDateTime(json['captured_at']),
        contribution: contractDouble(json['contribution']),
      );
}

class AssessmentSummary {
  final String subjectId;
  final int? fusionResultId;
  final CurrentAssessment currentAssessment;
  final ForecastResult? forecast;
  final double? confidence;
  final AssessmentStatus assessmentStatus;
  final List<ModalityStatus> modalities;
  final DateTime? computedAt;
  final String? modelVersion;

  const AssessmentSummary({
    required this.subjectId,
    required this.fusionResultId,
    required this.currentAssessment,
    required this.forecast,
    required this.confidence,
    required this.assessmentStatus,
    required this.modalities,
    required this.computedAt,
    required this.modelVersion,
  });

  factory AssessmentSummary.fromJson(Map<String, dynamic> json) {
    final current = contractMap(json['current_assessment']) ?? const {};
    final forecast = contractMap(json['forecast']);
    return AssessmentSummary(
      subjectId: contractString(json['subject_id']) ?? '',
      fusionResultId: contractInt(json['fusion_result_id']),
      currentAssessment: CurrentAssessment.fromJson(current),
      forecast: forecast == null ? null : ForecastResult.fromJson(forecast),
      confidence: contractDouble(json['confidence'] ?? json['uncertainty']),
      assessmentStatus: AssessmentStatus.fromWire(json['assessment_status']),
      modalities: contractMapList(json['modalities'])
          .map(ModalityStatus.fromJson)
          .toList(growable: false),
      computedAt: contractDateTime(json['computed_at']),
      modelVersion: contractString(json['model_version']),
    );
  }
}
```

Do not add any score derivation, band thresholds or forecast threshold logic to these classes.

- [ ] **Step 7: Run the assessment tests**

Run:

```bash
flutter test test/contracts/assessment_summary_test.dart
flutter analyze lib/domain/contracts test/contracts
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add lib/domain/contracts/assessment_summary.dart \
  test/contracts/fixture_loader.dart \
  test/contracts/assessment_summary_test.dart \
  test/fixtures/contracts/assessment_*.json
git commit -m "feat: add canonical assessment contracts"
```

---

### Task 3: Add AttentionEvent and PatientSummary contracts

**Files:**
- Create: `lib/domain/contracts/attention_event.dart`
- Create: `lib/domain/contracts/patient_summary.dart`
- Create: `test/contracts/attention_event_test.dart`
- Create: `test/contracts/patient_summary_test.dart`
- Create: `test/fixtures/contracts/attention_event_open.json`
- Create: `test/fixtures/contracts/attention_event_acknowledged.json`
- Create: `test/fixtures/contracts/attention_event_resolved.json`
- Create: `test/fixtures/contracts/patient_summary.json`

**Interfaces:**
- Consumes: Task 1 enums/parsing and Task 2 `CurrentAssessment`/`ForecastResult`.
- Produces: `AttentionEvent.fromJson(Map<String, dynamic>)` and `PatientSummary.fromJson(Map<String, dynamic>)`.

- [ ] **Step 1: Add event fixtures**

Create `attention_event_open.json`:

```json
{
  "id": "evt-001",
  "subject_id": "subject-001",
  "fusion_result_id": 123,
  "forecast_result_id": "fcst-001",
  "event_type": "acute_escalation_forecast",
  "severity": "high",
  "reason": "Forecast crossed versioned escalation policy",
  "forecast_horizon": 10,
  "status": "OPEN",
  "created_at": "2026-09-15T10:01:00Z",
  "acknowledged_at": null,
  "acknowledged_by": null,
  "resolved_at": null,
  "resolved_by": null,
  "policy_version": "escalation-v1"
}
```

Create `attention_event_acknowledged.json` with the same identity and:

```json
{
  "id": "evt-001",
  "subject_id": "subject-001",
  "fusion_result_id": 123,
  "forecast_result_id": "fcst-001",
  "event_type": "acute_escalation_forecast",
  "severity": "high",
  "reason": "Forecast crossed versioned escalation policy",
  "forecast_horizon": 10,
  "status": "ACKNOWLEDGED",
  "created_at": "2026-09-15T10:01:00Z",
  "acknowledged_at": "2026-09-15T10:03:00Z",
  "acknowledged_by": "DR001",
  "resolved_at": null,
  "resolved_by": null,
  "policy_version": "escalation-v1"
}
```

Create `attention_event_resolved.json` with:

```json
{
  "id": "evt-001",
  "subject_id": "subject-001",
  "fusion_result_id": 123,
  "forecast_result_id": "fcst-001",
  "event_type": "acute_escalation_forecast",
  "severity": "high",
  "reason": "Forecast crossed versioned escalation policy",
  "forecast_horizon": 10,
  "status": "RESOLVED",
  "created_at": "2026-09-15T10:01:00Z",
  "acknowledged_at": "2026-09-15T10:03:00Z",
  "acknowledged_by": "DR001",
  "resolved_at": "2026-09-15T10:15:00Z",
  "resolved_by": "DR001",
  "policy_version": "escalation-v1"
}
```

- [ ] **Step 2: Add a patient-summary fixture using the handbook dashboard shape**

Create `test/fixtures/contracts/patient_summary.json`:

```json
{
  "subject_id": "subject-001",
  "display_id": "Patient A",
  "fusion_result_id": 123,
  "current": {"score": 0.58, "tier": "Medium", "band": "AMBER"},
  "forecast": {
    "forecast_result_id": "fcst-001",
    "scope": "physiological",
    "horizon_minutes": 10,
    "score": 0.84,
    "tier": "High",
    "escalation_predicted": true,
    "generated_at": "2026-09-15T10:00:00Z",
    "valid_until": "2026-09-15T10:10:00Z"
  },
  "assessment_status": "complete",
  "last_updated": "2026-09-15T10:00:00Z",
  "open_event_count": 1
}
```

- [ ] **Step 3: Write failing tests**

Create `test/contracts/attention_event_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

import 'fixture_loader.dart';

void main() {
  test('OPEN event has no fabricated acknowledgement', () {
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_open.json'),
    );
    expect(event.status, AttentionEventStatus.open);
    expect(event.severity, AttentionSeverity.high);
    expect(event.acknowledgedAt, isNull);
    expect(event.resolvedAt, isNull);
  });

  test('ACKNOWLEDGED event preserves server actor and time', () {
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_acknowledged.json'),
    );
    expect(event.status, AttentionEventStatus.acknowledged);
    expect(event.acknowledgedBy, 'DR001');
    expect(event.acknowledgedAt, isNotNull);
    expect(event.resolvedAt, isNull);
  });

  test('RESOLVED event preserves the same event and source result ids', () {
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_resolved.json'),
    );
    expect(event.id, 'evt-001');
    expect(event.fusionResultId, 123);
    expect(event.forecastResultId, 'fcst-001');
    expect(event.status, AttentionEventStatus.resolved);
    expect(event.resolvedBy, 'DR001');
  });
}
```

Create `test/contracts/patient_summary_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/contracts/patient_summary.dart';

import 'fixture_loader.dart';

void main() {
  test('dashboard patient summary keeps current and forecast separate', () {
    final patient = PatientSummary.fromJson(
      loadContractFixture('patient_summary.json'),
    );
    expect(patient.subjectId, 'subject-001');
    expect(patient.fusionResultId, 123);
    expect(patient.currentAssessment!.score, 0.58);
    expect(patient.forecast!.score, 0.84);
    expect(patient.forecast!.scope, ForecastScope.physiological);
    expect(patient.openEventCount, 1);
  });
}
```

- [ ] **Step 4: Run the tests and verify they fail**

```bash
flutter test test/contracts/attention_event_test.dart \
  test/contracts/patient_summary_test.dart
```

Expected: FAIL because the production contract files do not exist.

- [ ] **Step 5: Implement `AttentionEvent`**

Create `lib/domain/contracts/attention_event.dart`:

```dart
import 'contract_enums.dart';
import 'contract_parsing.dart';

class AttentionEvent {
  final String id;
  final String subjectId;
  final int? fusionResultId;
  final String? forecastResultId;
  final String? eventType;
  final AttentionSeverity severity;
  final String? reason;
  final int? forecastHorizon;
  final AttentionEventStatus status;
  final DateTime? createdAt;
  final DateTime? acknowledgedAt;
  final String? acknowledgedBy;
  final DateTime? resolvedAt;
  final String? resolvedBy;
  final String? policyVersion;

  const AttentionEvent({
    required this.id,
    required this.subjectId,
    required this.fusionResultId,
    required this.forecastResultId,
    required this.eventType,
    required this.severity,
    required this.reason,
    required this.forecastHorizon,
    required this.status,
    required this.createdAt,
    required this.acknowledgedAt,
    required this.acknowledgedBy,
    required this.resolvedAt,
    required this.resolvedBy,
    required this.policyVersion,
  });

  factory AttentionEvent.fromJson(Map<String, dynamic> json) => AttentionEvent(
        id: contractString(json['id']) ?? '',
        subjectId: contractString(json['subject_id']) ?? '',
        fusionResultId: contractInt(json['fusion_result_id']),
        forecastResultId: contractString(json['forecast_result_id']),
        eventType: contractString(json['event_type']),
        severity: AttentionSeverity.fromWire(json['severity']),
        reason: contractString(json['reason']),
        forecastHorizon: contractInt(json['forecast_horizon']),
        status: AttentionEventStatus.fromWire(json['status']),
        createdAt: contractDateTime(json['created_at']),
        acknowledgedAt: contractDateTime(json['acknowledged_at']),
        acknowledgedBy: contractString(json['acknowledged_by']),
        resolvedAt: contractDateTime(json['resolved_at']),
        resolvedBy: contractString(json['resolved_by']),
        policyVersion: contractString(json['policy_version']),
      );
}
```

- [ ] **Step 6: Implement `PatientSummary`**

Create `lib/domain/contracts/patient_summary.dart`:

```dart
import 'assessment_summary.dart';
import 'contract_enums.dart';
import 'contract_parsing.dart';

class PatientSummary {
  final String subjectId;
  final String? displayId;
  final int? fusionResultId;
  final CurrentAssessment? currentAssessment;
  final ForecastResult? forecast;
  final AssessmentStatus assessmentStatus;
  final DateTime? lastUpdated;
  final int? openEventCount;

  const PatientSummary({
    required this.subjectId,
    required this.displayId,
    required this.fusionResultId,
    required this.currentAssessment,
    required this.forecast,
    required this.assessmentStatus,
    required this.lastUpdated,
    required this.openEventCount,
  });

  factory PatientSummary.fromJson(Map<String, dynamic> json) {
    final current = contractMap(json['current'] ?? json['current_assessment']);
    final forecast = contractMap(json['forecast']);
    return PatientSummary(
      subjectId: contractString(json['subject_id']) ?? '',
      displayId: contractString(json['display_id']),
      fusionResultId: contractInt(json['fusion_result_id']),
      currentAssessment:
          current == null ? null : CurrentAssessment.fromJson(current),
      forecast: forecast == null ? null : ForecastResult.fromJson(forecast),
      assessmentStatus: AssessmentStatus.fromWire(json['assessment_status']),
      lastUpdated: contractDateTime(json['last_updated']),
      openEventCount: contractInt(json['open_event_count']),
    );
  }
}
```

- [ ] **Step 7: Run tests**

```bash
flutter test test/contracts/attention_event_test.dart \
  test/contracts/patient_summary_test.dart
flutter analyze lib/domain/contracts test/contracts
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add lib/domain/contracts/attention_event.dart \
  lib/domain/contracts/patient_summary.dart \
  test/contracts/attention_event_test.dart \
  test/contracts/patient_summary_test.dart \
  test/fixtures/contracts/attention_event_*.json \
  test/fixtures/contracts/patient_summary.json
git commit -m "feat: add attention event contracts"
```

---

### Task 4: Remove the non-existent `/v1/subjects/attach` dependency

**Files:**
- Modify: `lib/data/api/gateways.dart` enrolment section
- Modify: `lib/state/controllers.dart` `ChartController.ensureEnrolled`
- Modify: `test/gateway_contract_test.dart`

**Interfaces:**
- Consumes existing backend contract: `GET /v1/subjects/resolve?app_user_id=...`.
- Produces `Future<String?> CentralBackendGateway.resolveAppUserId(String appUserId)`.
- Removes `CentralBackendGateway.attach(...)`.
- `ChartController.ensureEnrolled` resolves `P_[A-F0-9]{16}` identifiers through `app_user_id` instead of POSTing to a route the backend does not have.

- [ ] **Step 1: Write a transport test for `app_user_id` resolution**

Add this test inside the existing enrolment/transport group in `test/gateway_contract_test.dart`:

```dart
test('Aura participant id resolves through the existing backend alias route',
    () async {
  final g = _gateway(
    (_) async => http.Response(
      '{"subject_id":"subject-aura-001"}',
      200,
    ),
  );

  final id = await g.resolveAppUserId('P_65DC4002E7863773');

  expect(id, 'subject-aura-001');
  expect(
    sent.single.url.toString(),
    '$_base/v1/subjects/resolve?app_user_id=P_65DC4002E7863773',
  );
  expect(sent.single.method, 'GET');
});

test('unknown Aura participant id resolves to null', () async {
  expect(
    await _failing(404).resolveAppUserId('P_65DC4002E7863773'),
    isNull,
  );
});
```

- [ ] **Step 2: Add a structural regression test that rejects `/attach`**

Add to `test/gateway_contract_test.dart`:

```dart
test('ClinAnx does not depend on the non-existent subjects/attach route', () {
  final gatewaySource =
      File('lib/data/api/gateways.dart').readAsStringSync();
  final controllerSource =
      File('lib/state/controllers.dart').readAsStringSync();

  expect(gatewaySource, isNot(contains('/v1/subjects/attach')));
  expect(controllerSource, isNot(contains('_backend.attach(')));
  expect(controllerSource, contains('resolveAppUserId'));
});
```

The file already imports `dart:io`, so no new test import is needed.

- [ ] **Step 3: Run the focused tests and verify they fail**

```bash
flutter test test/gateway_contract_test.dart
```

Expected: FAIL because `resolveAppUserId` is missing and `/v1/subjects/attach` still exists.

- [ ] **Step 4: Replace `attach` with `resolveAppUserId` in the gateway**

Delete the `attach(...)` method and add this method beside `resolveMrn`:

```dart
Future<String?> resolveAppUserId(String appUserId) async {
  try {
    final json = await _api.get(
      '/v1/subjects/resolve?app_user_id='
      '${Uri.encodeQueryComponent(appUserId)}',
      timeout: Env.quickTimeout,
    );
    final id = json['subject_id'];
    return id == null ? null : '$id';
  } on ApiException catch (e) {
    if (e.kind == ApiFailure.notFound) return null;
    rethrow;
  }
}
```

Do not create a replacement POST route.

- [ ] **Step 5: Update `ChartController.ensureEnrolled`**

Replace the current resolve/attach block with this order:

```dart
Future<String> ensureEnrolled({String? clinicianId}) async {
  if (isEnrolled) return _subjectId!;

  final candidate = mrn.trim().toUpperCase();

  if (_participantIdPattern.hasMatch(candidate)) {
    final resolvedByAppId = await _backend.resolveAppUserId(candidate);
    if (resolvedByAppId != null) {
      _subjectId = resolvedByAppId;
      await _linkBehaviouralId(resolvedByAppId);
      await RecordStore.saveSubjectId(mrn, resolvedByAppId);
      notifyListeners();
      return resolvedByAppId;
    }
  }

  final resolved = await _backend.resolveMrn(mrn);
  if (resolved != null) {
    _subjectId = resolved;
    await _linkBehaviouralId(resolved);
    await RecordStore.saveSubjectId(mrn, resolved);
    notifyListeners();
    return resolved;
  }

  final enrolment = await _backend.enrol(
    mrn: mrn,
    enrolledBy: clinicianId,
  );
  _subjectId = enrolment.subjectId;
  await _linkBehaviouralId(enrolment.subjectId);
  _pendingPairing = enrolment;
  await RecordStore.saveSubjectId(mrn, enrolment.subjectId);
  notifyListeners();
  return enrolment.subjectId;
}
```

Update the doc comment above the method so its resolution order is accurate:

```text
1. locally stored subject_id
2. Aura participant id -> app_user_id alias resolve
3. MRN -> mrn_hash alias resolve
4. enrol if unknown
```

- [ ] **Step 6: Run focused tests**

```bash
flutter test test/gateway_contract_test.dart
flutter analyze lib/data/api/gateways.dart lib/state/controllers.dart test/gateway_contract_test.dart
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add lib/data/api/gateways.dart \
  lib/state/controllers.dart \
  test/gateway_contract_test.dart
git commit -m "fix: resolve patient identity without attach route"
```

---

### Task 5: Add cross-contract safety invariants and run the complete verification gate

**Files:**
- Create: `test/contracts/p0_invariants_test.dart`
- No production file changes expected unless verification reveals a defect in Tasks 1-4.

**Interfaces:**
- Consumes all P0 contract classes and frozen fixtures.
- Produces a regression gate that prevents future changes from conflating current assessment, forecast, excluded modalities or attention-event source IDs.

- [ ] **Step 1: Write the invariant suite**

Create `test/contracts/p0_invariants_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

import 'fixture_loader.dart';

void main() {
  test('current assessment and forecast cannot collapse into one risk field', () {
    final assessment = AssessmentSummary.fromJson(
      loadContractFixture('assessment_complete.json'),
    );

    expect(assessment.currentAssessment.score, 0.58);
    expect(assessment.forecast!.score, 0.84);
    expect(assessment.forecast!.scope, ForecastScope.physiological);
  });

  test('same canonical fusion row is linkable from assessment and event', () {
    final assessment = AssessmentSummary.fromJson(
      loadContractFixture('assessment_complete.json'),
    );
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_open.json'),
    );

    expect(assessment.fusionResultId, isNotNull);
    expect(event.fusionResultId, assessment.fusionResultId);
    expect(event.subjectId, assessment.subjectId);
  });

  test('experimental C2 is visible but never silently included', () {
    final assessment = AssessmentSummary.fromJson(
      loadContractFixture('assessment_complete.json'),
    );
    final c2 = assessment.modalities.singleWhere(
      (item) => item.componentId == 'c2_behavioral',
    );

    expect(c2.state, ModalityState.notValidated);
    expect(c2.available, isFalse);
    expect(c2.includedInFusion, isFalse);
    expect(c2.contribution, isNull);
  });

  test('unavailable assessment has no fabricated current score', () {
    final assessment = AssessmentSummary.fromJson(
      loadContractFixture('assessment_unavailable_c3.json'),
    );

    expect(assessment.assessmentStatus, AssessmentStatus.unavailable);
    expect(assessment.currentAssessment.score, isNull);
    expect(assessment.currentAssessment.tier, RiskTier.unknown);
  });

  test('unknown server vocabulary is conservative', () {
    final assessment = AssessmentSummary.fromJson(
      loadContractFixture('assessment_unknown_enum.json'),
    );

    expect(assessment.assessmentStatus, AssessmentStatus.unknown);
    expect(assessment.currentAssessment.tier, RiskTier.unknown);
    expect(assessment.forecast!.scope, ForecastScope.unknown);
    expect(assessment.currentAssessment.score, isNull);
  });
}
```

- [ ] **Step 2: Run all contract tests**

```bash
flutter test test/contracts
```

Expected: PASS.

- [ ] **Step 3: Run the existing application test suite**

```bash
flutter test
```

Expected: every existing test plus the new contract tests passes. Existing timeline parser tests must remain unchanged in behavior.

- [ ] **Step 4: Run static analysis**

```bash
flutter analyze
```

Expected: no analyzer errors. Warnings that already existed before the branch must be documented in the PR rather than hidden by unrelated lint changes.

- [ ] **Step 5: Search for forbidden contract regressions**

Run:

```bash
git grep -n "/v1/subjects/attach" -- lib test || true
git grep -n "FusionResult.local" -- lib test || true
git grep -n "TcwpnGateway" -- lib test || true
```

Expected:

```text
/v1/subjects/attach -> no matches
FusionResult.local   -> comments/tests may mention the removed symbol, but no executable call site
TcwpnGateway         -> comments may mention the retired class, but no executable call site
```

Inspect any match manually; do not delete historical safety comments solely to make `git grep` empty.

- [ ] **Step 6: Review diff against the approved design**

Run:

```bash
git diff main...HEAD -- \
  lib/domain/contracts \
  lib/data/api/gateways.dart \
  lib/state/controllers.dart \
  test/contracts \
  test/fixtures/contracts \
  test/gateway_contract_test.dart
```

Confirm all of the following before opening the PR:

```text
- no fusion formula or weight changes
- no direct /predict call added
- no target dashboard/attention HTTP routes invented
- no final UI redesign
- no local alert removal
- no C2 inclusion in fusion
- current and forecast fields remain separate
- missing/unknown scores stay nullable
- identity uses existing /v1/subjects/resolve
```

- [ ] **Step 7: Commit the invariant suite**

```bash
git add test/contracts/p0_invariants_test.dart
git commit -m "test: lock ClinAnx P0 integration invariants"
```

- [ ] **Step 8: Final verification after the last commit**

Run again from a clean working tree:

```bash
flutter test
flutter analyze
git status --short
```

Expected:

```text
all tests pass
no analyzer errors
empty git status
```

---

## PR Gate

Open the PR only after Task 5 succeeds.

Target:

```text
head: integration/clinanx-p0-contract-first
base: main
```

Recommended title:

```text
ClinAnx P0: lock integration contracts and canonical identity resolution
```

Recommended PR body:

```markdown
## Purpose

Establish the ClinAnx P0 contract foundation from the approved system-integration and clinician-app specifications without pretending unfinished backend APIs already exist.

## What changed

- added typed canonical assessment/forecast/modality contracts
- added typed patient-summary and AttentionEvent contracts
- added frozen JSON fixtures for complete/partial/unavailable/unknown states
- added explicit unknown enum handling
- preserved nullable unavailable scores instead of defaulting to low/zero
- removed the client dependency on non-existent `POST /v1/subjects/attach`
- resolve Aura participant IDs through the backend's existing `/v1/subjects/resolve?app_user_id=...` alias contract

## Intentionally unchanged

- fusion mathematics and weights
- authoritative clinical-note flow (`ClinAnx -> central backend -> C3 -> fusion`)
- current legacy timeline rendering
- local ClinicalAlert path, pending a real server AttentionEvent subsystem
- final dashboard/patient-overview UI
- teammate-owned repositories

## Verification

- `flutter test`
- `flutter analyze`
- P0 invariant tests for shared `fusion_result_id`, current-vs-forecast separation, C2 exclusion, C3 unavailable behavior and unknown-enum safety

## Backend dependency

The next integration stage still requires the central backend owner to freeze/implement clinician principals, assignments, latest AssessmentSummary, ForecastResult persistence, AttentionEvent persistence, and acknowledge/resolve APIs.
```

Do not merge the PR merely because it opens successfully. Review the changed-file list and CI status first.

---

## Plan Self-Review

### Spec coverage

- typed enums: Task 1
- safe nullable parsing: Task 1
- `AssessmentSummary`: Task 2
- `CurrentAssessment`: Task 2
- `ForecastResult`: Task 2
- `ModalityStatus`: Task 2
- complete/partial/unavailable/stale/C2/C3/unknown fixtures: Task 2
- `AttentionEvent`: Task 3
- `PatientSummary`: Task 3
- OPEN/ACKNOWLEDGED/RESOLVED fixtures: Task 3
- `/subjects/attach` mismatch: Task 4
- no client fusion/current-vs-forecast/C2/C3 invariants: Task 5
- existing ClinAnx regression suite and analyzer gate: Task 5

### Scope boundary

This plan intentionally stops before assignment-scoped dashboard networking and server attention-event networking because those target endpoints are not implemented in the verified central backend snapshot. Adding those calls now would violate the approved rule against presenting TARGET behavior as CURRENT.

### Type consistency

The plan uses exactly these shared names across tasks:

```text
RiskTier
AssessmentStatus
ForecastScope
ModalityState
AttentionEventStatus
AttentionSeverity
CurrentAssessment
ForecastResult
ModalityStatus
AssessmentSummary
AttentionEvent
PatientSummary
CentralBackendGateway.resolveAppUserId
```

No alias names are introduced for the same concept.
