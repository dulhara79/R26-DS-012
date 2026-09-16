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

  test('P3 keeps unverified latest-assessment route out of production adapter',
      () {
    final source = File('lib/data/repositories/central_backend_repositories.dart')
        .readAsStringSync();

    expect(source, isNot(contains('/assessment/latest')));
    expect(source, contains('ApiFailure.notConfigured'));
  });

  test('legacy Patients resolves identity rather than treating MRN as subject id',
      () {
    final source =
        File('lib/features/patients/patients_screen.dart').readAsStringSync();

    expect(source, contains('resolveAppUserId'));
    expect(source, contains('resolveMrn'));
    expect(source, isNot(contains('subjectId: patient.mrn')));
  });
}
