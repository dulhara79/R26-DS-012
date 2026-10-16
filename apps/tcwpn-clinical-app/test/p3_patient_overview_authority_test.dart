import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('P3 Patient Overview does not derive authoritative risk locally', () {
    final source = File('lib/features/patients/patient_overview_screen.dart')
        .readAsStringSync();

    expect(source, isNot(contains('AlertBandX.fromScore')));
    expect(source, isNot(contains('compositeScore =')));
    expect(source, isNot(contains('weight * score')));
    expect(source, isNot(contains("'/predict'")));
    expect(source, isNot(contains('"/predict"')));
  });

  test('Phase 5 latest-assessment route is verified and live-wired', () {
    final source = File('lib/data/repositories/central_backend_repositories.dart')
        .readAsStringSync();

    expect(source, contains('/assessment/latest'));
    expect(source, contains('AssessmentSummary.fromJson'));
    expect(source, isNot(contains('latest-assessment-contract')));
  });

  test('Patients screen consumes the assigned server roster', () {
    final source =
        File('lib/features/patients/patients_screen.dart').readAsStringSync();

    expect(source, contains('DashboardController'));
    expect(source, contains('assignedPatients'));
    expect(source, contains('PatientSummary'));
    expect(source, isNot(contains('RosterController')));
    expect(source, isNot(contains('resolveMrn')));
    expect(source, isNot(contains('resolveAppUserId')));
    expect(source, isNot(contains('FusionResult')));
    expect(source, isNot(contains('AlertBandX.fromScore')));
  });
}
