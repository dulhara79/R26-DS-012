import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// A clinician-entered display label, not an identity key or risk signal.
abstract class PatientLabelStore {
  Future<String?> read({
    required String clinicianId,
    required String subjectId,
  });

  Future<void> write({
    required String clinicianId,
    required String subjectId,
    required String label,
  });

  Future<void> delete({
    required String clinicianId,
    required String subjectId,
  });

  static String normalizeLabel(String value) =>
      value.trim().replaceAll(RegExp(r'\s+'), ' ');

  static String? validateLabel(String? value) {
    final raw = value ?? '';
    if (RegExp(r'[\x00-\x1F\x7F]').hasMatch(raw)) {
      return 'The label cannot contain control characters.';
    }
    final normalized = normalizeLabel(raw);
    if (normalized.isEmpty) return 'Enter a patient label.';
    if (normalized.length > 80) return 'Use 80 characters or fewer.';
    return null;
  }
}

/// Stored only in OS-backed encrypted storage. The existing SecureStore.signOut
/// calls deleteAll(), so labels are erased on sign-out on this device.
/// Never include names in SharedPreferences snapshots, network requests or QR.
class SecurePatientLabelStore implements PatientLabelStore {
  const SecurePatientLabelStore();

  static const _storage = FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
    iOptions: IOSOptions(
      accessibility: KeychainAccessibility.first_unlock_this_device,
    ),
  );

  static String _key(String clinicianId, String subjectId) {
    final clinician = clinicianId.trim().toUpperCase();
    final subject = subjectId.trim();
    if (clinician.isEmpty || subject.isEmpty) {
      throw ArgumentError('Authenticated clinician and subject are required.');
    }
    final digest = sha256.convert(utf8.encode('$clinician\u0000$subject'));
    return 'clinanx.patient_label.v1.$digest';
  }

  @override
  Future<String?> read({
    required String clinicianId,
    required String subjectId,
  }) => _storage.read(key: _key(clinicianId, subjectId));

  @override
  Future<void> write({
    required String clinicianId,
    required String subjectId,
    required String label,
  }) async {
    final error = PatientLabelStore.validateLabel(label);
    if (error != null) throw ArgumentError(error);
    await _storage.write(
      key: _key(clinicianId, subjectId),
      value: PatientLabelStore.normalizeLabel(label),
    );
  }

  @override
  Future<void> delete({
    required String clinicianId,
    required String subjectId,
  }) => _storage.delete(key: _key(clinicianId, subjectId));
}
