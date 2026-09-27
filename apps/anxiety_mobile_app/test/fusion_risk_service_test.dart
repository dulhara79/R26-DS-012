import 'package:anxiety_mobile_app/services/fusion_risk_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('preserves authoritative fusion identity and scored assessment', () {
    final risk = FusionRisk.fromJson({
      'fusion_result_id': 123,
      'composite': 0.58,
      'band': 'AMBER',
      'updated_at': '2026-09-18T12:00:00Z',
    });

    expect(risk.fusionResultId, 123);
    expect(risk.hasScore, isTrue);
    expect(risk.scoreOutOf100, closeTo(58.0, 1e-9));
  });

  test('older response without fusion identity remains readable', () {
    final risk = FusionRisk.fromJson({'composite': 0.58, 'band': 'AMBER'});
    expect(risk.fusionResultId, isNull);
  });

  test('missing assessment never becomes zero or low', () {
    final risk = FusionRisk.fromJson({'band': 'GREY'});
    expect(risk.hasScore, isFalse);
    expect(risk.scoreOutOf100, isNull);
  });

  test('numeric composite with grey band remains unavailable', () {
    final risk = FusionRisk.fromJson({'composite': 0, 'band': 'grey'});
    expect(risk.band, 'GREY');
    expect(risk.hasScore, isFalse);
    expect(risk.scoreOutOf100, isNull);
  });
}
