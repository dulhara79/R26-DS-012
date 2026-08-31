import 'dart:io';

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

  group('officialOverallRisk', () {
    test('returns the backend fusion score', () {
      const risk = FusionRisk(composite: 0.61, band: 'AMBER');

      expect(officialOverallRisk(risk), 61.0);
    });

    test('returns null when no backend fusion result exists', () {
      expect(officialOverallRisk(null), isNull);
    });

    test('returns null when the backend refused to produce a score', () {
      const risk = FusionRisk(composite: 0.61, band: 'GREY');

      expect(officialOverallRisk(risk), isNull);
    });
  });

  test('home fusion polling refreshes within the investor demo window', () {
    final source = File(
      'lib/services/fusion_risk_service.dart',
    ).readAsStringSync();

    expect(source, contains('Duration(seconds: 5)'));
    expect(source, isNot(contains('Duration(minutes: 5)')));
  });
}
