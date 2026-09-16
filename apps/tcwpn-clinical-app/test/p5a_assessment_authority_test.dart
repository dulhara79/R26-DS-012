import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  const signalsPath =
      'lib/features/patients/signals_contributions_screen.dart';
  const qualityPath = 'lib/features/patients/data_quality_screen.dart';
  const overviewPath = 'lib/features/patients/patient_overview_screen.dart';
  const presentationPath =
      'lib/features/patients/assessment_detail_presentation.dart';

  test('P5A deep views use AssessmentSummary and not legacy FusionResult authority',
      () {
    final signals = File(signalsPath).readAsStringSync();
    final quality = File(qualityPath).readAsStringSync();

    expect(signals, contains('AssessmentSummary'));
    expect(quality, contains('AssessmentSummary'));
    expect(signals, isNot(contains('FusionResult')));
    expect(quality, isNot(contains('FusionResult')));
  });

  test('P5A deep views contain no direct model or guessed deep-view route', () {
    final deepViews = [
      File(signalsPath).readAsStringSync(),
      File(qualityPath).readAsStringSync(),
    ].join('\n');

    expect(deepViews, isNot(contains("'/predict'")));
    expect(deepViews, isNot(contains('"/predict"')));
    expect(deepViews, isNot(contains('/contributions')));
    expect(deepViews, isNot(contains('/data-quality')));
    expect(deepViews, isNot(contains('CentralBackendGateway')));
    expect(deepViews, isNot(contains('AssessmentRepository')));
  });

  test('Patient Overview deep-view navigation does not add another data boundary',
      () {
    final overview = File(overviewPath).readAsStringSync();

    expect(overview, contains('SignalsContributionsScreen('));
    expect(overview, contains('DataQualityScreen('));
    expect(overview, contains('assessment: assessment'));
    expect(overview, isNot(contains('/contributions')));
    expect(overview, isNot(contains('/data-quality')));
    expect(overview, isNot(contains("'/predict'")));
  });

  test('P5A does not locally derive risk or missing contribution', () {
    final source = [
      File(signalsPath).readAsStringSync(),
      File(qualityPath).readAsStringSync(),
      File(presentationPath).readAsStringSync(),
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
      File(presentationPath).readAsStringSync(),
    ].join('\n');

    expect(source, contains('Clinical NLP / TC-WPN'));
    expect(source, contains('Not reported'));
    expect(source, isNot(contains('Overall TC-WPN risk')));
    expect(source, isNot(contains('TC-WPN Risk')));
  });
}
