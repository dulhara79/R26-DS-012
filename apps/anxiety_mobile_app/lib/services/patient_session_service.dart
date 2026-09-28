import 'dart:math';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

abstract class PatientSecretStore {
  Future<String?> read(String key);
  Future<void> write(String key, String value);
  Future<void> delete(String key);
}

class FlutterPatientSecretStore implements PatientSecretStore {
  FlutterPatientSecretStore({FlutterSecureStorage? storage})
    : _storage = storage ?? const FlutterSecureStorage();

  final FlutterSecureStorage _storage;

  @override
  Future<String?> read(String key) => _storage.read(key: key);

  @override
  Future<void> write(String key, String value) =>
      _storage.write(key: key, value: value);

  @override
  Future<void> delete(String key) => _storage.delete(key: key);
}

class PatientSession {
  const PatientSession({
    required this.subjectId,
    required this.accessToken,
    required this.expiresAt,
  });

  final String subjectId;
  final String accessToken;
  final DateTime expiresAt;
}

/// Owns the patient installation proof and short-lived central-backend JWT.
///
/// These values are intentionally kept out of SharedPreferences and build-time
/// defines. The installation proof is stable for this installation, while an
/// expired JWT is discarded and refreshed through `/v1/subjects/self`.
class PatientSessionService {
  PatientSessionService({PatientSecretStore? store, DateTime Function()? clock})
    : _store = store ?? FlutterPatientSecretStore(),
      _clock = clock ?? DateTime.now;

  static final PatientSessionService instance = PatientSessionService();

  static const String installationSecretKey =
      'central_patient_installation_secret';
  static const String subjectIdKey = 'central_patient_subject_id';
  static const String accessTokenKey = 'central_patient_access_token';
  static const String expiresAtKey = 'central_patient_expires_at';

  final PatientSecretStore _store;
  final DateTime Function() _clock;

  Future<String> getOrCreateInstallationSecret() async {
    final existing = await _store.read(installationSecretKey);
    if (existing != null && existing.length >= 32) return existing;

    final random = Random.secure();
    final secret = List<int>.generate(
      32,
      (_) => random.nextInt(256),
    ).map((value) => value.toRadixString(16).padLeft(2, '0')).join();
    await _store.write(installationSecretKey, secret);
    return secret;
  }

  Future<void> saveSession({
    required String subjectId,
    required String accessToken,
    required DateTime expiresAt,
  }) async {
    if (subjectId.trim().isEmpty || accessToken.trim().isEmpty) {
      throw ArgumentError('Patient session values cannot be empty.');
    }
    await _store.write(subjectIdKey, subjectId.trim());
    await _store.write(accessTokenKey, accessToken.trim());
    await _store.write(expiresAtKey, expiresAt.toUtc().toIso8601String());
  }

  Future<PatientSession?> currentSession() async {
    final subjectId = await _store.read(subjectIdKey);
    final accessToken = await _store.read(accessTokenKey);
    final rawExpiry = await _store.read(expiresAtKey);
    final expiresAt = rawExpiry == null ? null : DateTime.tryParse(rawExpiry);
    if (subjectId == null ||
        subjectId.isEmpty ||
        accessToken == null ||
        accessToken.isEmpty ||
        expiresAt == null ||
        !expiresAt.isAfter(_clock().toUtc().add(const Duration(seconds: 30)))) {
      await clearSession();
      return null;
    }
    return PatientSession(
      subjectId: subjectId,
      accessToken: accessToken,
      expiresAt: expiresAt,
    );
  }

  Future<Map<String, String>?> authenticatedHeaders({
    bool includeJsonContentType = true,
  }) async {
    final session = await currentSession();
    if (session == null) return null;
    return {
      if (includeJsonContentType) 'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': 'Bearer ${session.accessToken}',
    };
  }

  Future<void> clearSession() async {
    await _store.delete(subjectIdKey);
    await _store.delete(accessTokenKey);
    await _store.delete(expiresAtKey);
  }

  Future<void> clearAll() async {
    await clearSession();
    await _store.delete(installationSecretKey);
  }
}
