import 'package:anxiety_mobile_app/services/fusion_risk_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('parses a provisional two-score assessment', () {
    final risk = FusionRisk.fromJson({
      'composite': 0.61,
      'band': 'AMBER',
      'assessment_status': 'provisional',
      'missing_modalities': ['c3_clinical_nlp'],
    });

    expect(risk.assessmentStatus, 'provisional');
    expect(risk.assessmentLabel, 'Provisional assessment');
    expect(risk.missingModalities, ['c3_clinical_nlp']);
  });

  test('parses a complete three-score assessment', () {
    final risk = FusionRisk.fromJson({
      'composite': 0.61,
      'band': 'AMBER',
      'assessment_status': 'complete',
      'missing_modalities': <String>[],
    });

    expect(risk.assessmentStatus, 'complete');
    expect(risk.assessmentLabel, 'Complete assessment');
    expect(risk.missingModalities, isEmpty);
  });
}
