import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/clinician_principal.dart';

void main() {
  test('preserves clinician identity from the server contract', () {
    final principal = ClinicianPrincipal.fromJson({
      'clinician_id': 'DR001',
      'display_name': 'Clinician One',
      'role': 'clinician',
      'expires_at': '2026-09-16T10:00:00Z',
    });

    expect(principal.clinicianId, 'DR001');
    expect(principal.displayName, 'Clinician One');
    expect(principal.role, 'clinician');
    expect(principal.isValid, isTrue);
  });

  test('accepts sub as an alternate principal identifier', () {
    final principal = ClinicianPrincipal.fromJson({'sub': 'DR002'});
    expect(principal.clinicianId, 'DR002');
    expect(principal.isValid, isTrue);
  });

  test('does not invent a clinician identifier when absent', () {
    final principal = ClinicianPrincipal.fromJson(const {});
    expect(principal.clinicianId, isEmpty);
    expect(principal.isValid, isFalse);
  });
}
