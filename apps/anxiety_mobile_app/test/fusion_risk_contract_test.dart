import 'package:anxiety_mobile_app/services/fusion_risk_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('preserves the server fusion assessment ID', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 'assessment-123',
      'composite': 0.42,
      'band': 'AMBER',
      'message': 'Server assessment',
      'updated_at': '2026-09-25T10:00:00Z',
    });

    expect(risk.fusionResultId, 'assessment-123');
    expect(risk.scoreOutOf100, 42.0);
    expect(officialOverallRisk(risk), 42.0);
  });

  test('GREY fusion results remain unavailable', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 'assessment-grey',
      'composite': 0.05,
      'band': 'GREY',
    });

    expect(risk.hasScore, isFalse);
    expect(risk.scoreOutOf100, 5.0);
    expect(officialOverallRisk(risk), isNull);
  });

  test('missing composite remains unavailable', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 'assessment-missing',
      'band': 'AMBER',
    });

    expect(risk.hasScore, isFalse);
    expect(officialOverallRisk(risk), isNull);
  });

  test('missing fusion result remains unavailable', () {
    expect(officialOverallRisk(null), isNull);
  });
}
