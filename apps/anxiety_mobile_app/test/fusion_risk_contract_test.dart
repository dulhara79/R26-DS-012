import 'package:anxiety_mobile_app/services/fusion_risk_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('preserves the server fusion assessment ID', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 123,
      'composite': 0.42,
      'band': 'AMBER',
      'tier': 'Medium',
      'message': 'Server assessment',
      'updated_at': '2026-09-25T10:00:00Z',
    });

    expect(risk.fusionResultId, 123);
    expect(risk.scoreOutOf100, 42.0);
    expect(officialOverallRisk(risk), 42.0);
    expect(risk.displayTier, 'Medium');
  });

  test('GREY fusion results remain unavailable', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 124,
      'composite': 0.05,
      'band': 'GREY',
    });

    expect(risk.hasScore, isFalse);
    expect(risk.scoreOutOf100, isNull);
    expect(officialOverallRisk(risk), isNull);
  });

  test('missing composite remains unavailable', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 125,
      'band': 'AMBER',
    });

    expect(risk.hasScore, isFalse);
    expect(officialOverallRisk(risk), isNull);
  });

  test('missing fusion result remains unavailable', () {
    expect(officialOverallRisk(null), isNull);
  });

  test('separate server forecast remains physiological and expires', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 128,
      'composite': 0.58,
      'tier': 'Medium',
      'band': 'AMBER',
      'forecast': {
        'scope': 'physiological',
        'horizon_minutes': 10,
        'score': 0.84,
        'tier': 'High',
        'predicted': true,
        'generated_at': '2026-09-29T10:00:00Z',
        'valid_until': '2026-09-29T10:10:00Z',
      },
    });
    expect(risk.scoreOutOf100, closeTo(58.0, 1e-9));
    expect(risk.forecast?.scope, 'physiological');
    expect(risk.forecast?.scoreOutOf100, closeTo(84.0, 1e-9));
    expect(risk.forecast?.isValidAt(DateTime.utc(2026, 9, 29, 10, 9)), isTrue);
    expect(
      risk.forecast?.isValidAt(DateTime.utc(2026, 9, 29, 10, 11)),
      isFalse,
    );
  });

  test('missing server tier or invalid forecast is unavailable', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 129,
      'composite': 0.58,
      'band': 'AMBER',
      'forecast': {'scope': 'physiological', 'score': 0.84},
    });
    expect(risk.displayTier, 'Unavailable');
    expect(risk.forecast, isNull);
  });
}
