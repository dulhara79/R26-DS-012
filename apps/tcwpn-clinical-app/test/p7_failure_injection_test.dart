import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/evidence.dart';
import 'package:r26_ds012_app/state/async_data_state.dart';

void main() {
  test('stale C1, unavailable C3 and missing C4 never become zero or low', () {
    final assessment = AssessmentSummary.fromJson({
      'subject_id': 'subject-001',
      'fusion_result_id': 77,
      'assessment_status': 'partial',
      'current_assessment': {
        'score': null,
        'tier': null,
        'band': null,
      },
      'modalities': [
        {
          'component_id': 'c1_physiological',
          'score': null,
          'available': false,
          'included_in_fusion': false,
          'status': 'stale',
        },
        {
          'component_id': 'c3_clinical_nlp',
          'score': null,
          'available': false,
          'included_in_fusion': false,
          'status': 'unavailable',
        },
        {
          'component_id': 'c4_demographic',
          'score': null,
          'available': false,
          'included_in_fusion': false,
          'status': 'unavailable',
        },
      ],
    });

    expect(assessment.assessmentStatus, AssessmentStatus.partial);
    expect(assessment.currentAssessment.score, isNull);
    expect(assessment.currentAssessment.tier, RiskTier.unknown);
    expect(assessment.currentAssessment.band, isNull);
    for (final modality in assessment.modalities) {
      expect(modality.score, isNull);
      expect(modality.includedInFusion, isFalse);
    }
  });

  test('network loss without a cache is offline without fabricated data', () {
    const state = AsyncDataState<AssessmentSummary>.offline(
      message: 'No network connection.',
    );

    expect(state.status, AsyncDataStatus.offline);
    expect(state.data, isNull);
  });

  test('RAG timeout/unavailable is distinct from abstention and has no answer', () {
    const unavailable = EvidenceResult(
      available: false,
      error: 'timeout',
    );
    const abstained = EvidenceResult(
      available: true,
      abstained: true,
      abstentionReason: 'Insufficient grounded evidence.',
    );

    expect(unavailable.state, EvidenceState.unavailable);
    expect(unavailable.hasAnswer, isFalse);
    expect(abstained.state, EvidenceState.abstained);
    expect(abstained.hasAnswer, isFalse);
  });
}
