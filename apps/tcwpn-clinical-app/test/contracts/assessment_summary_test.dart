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
