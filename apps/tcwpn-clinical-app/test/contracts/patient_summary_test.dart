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
