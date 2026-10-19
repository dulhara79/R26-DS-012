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
