import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/features/patients/patient_invitation_qr.dart';

void main() {
  const token = 'A1234567890123456789012345678901234567890_Z'; // 43 chars

  test('raw patient-issued invitation is accepted without case changes', () {
    final result = parsePatientInvitationQr(token);
    expect(token.length, 43);
    expect(result.kind, PatientQrKind.invitation);
    expect(result.value, token);
  });

  test('explicit invitation deep link accepts same exact token', () {
    final result = parsePatientInvitationQr('clinanx://invite/$token');
    expect(result.kind, PatientQrKind.invitation);
    expect(result.value, token);
  });

  test('existing Aura QR encodes identity only, not consent', () {
    for (final data in [
      'P_0123456789ABCDEF',
      'clinanx://patient/P_0123456789ABCDEF',
    ]) {
      final result = parsePatientInvitationQr(data);
      expect(result.kind, PatientQrKind.participantId);
      expect(result.value, 'P_0123456789ABCDEF');
    }
  });

  test('wrong, modified, or web URL QR values are rejected', () {
    for (final data in [
      '',
      'aB_123-Z',
      'https://example.org/claim/$token',
      'clinanx://invite/$token?forward=true',
      'clinanx://invite/$token/unexpected',
      'clinanx://unknown/$token',
      'clinanx://invite/$token#fragment',
      'P_0123456789ABCDEG',
      'x' * 513,
    ]) {
      expect(parsePatientInvitationQr(data).kind, PatientQrKind.unsupported,
          reason: data);
    }
  });
}
