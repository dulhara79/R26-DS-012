import 'package:anxiety_mobile_app/services/patient_session_service.dart';
import 'package:flutter_test/flutter_test.dart';

class MemoryPatientSecretStore implements PatientSecretStore {
  final Map<String, String> values = {};

  @override
  Future<void> delete(String key) async => values.remove(key);

  @override
  Future<String?> read(String key) async => values[key];

  @override
  Future<void> write(String key, String value) async => values[key] = value;
}

void main() {
  test(
    'installation proof is generated once and kept separate from session',
    () async {
      final store = MemoryPatientSecretStore();
      final service = PatientSessionService(store: store);

      final first = await service.getOrCreateInstallationSecret();
      final second = await service.getOrCreateInstallationSecret();

      expect(first, second);
      expect(first.length, 64);
      expect(first, matches(RegExp(r'^[a-f0-9]{64}$')));
      expect(store.values.values, isNot(contains('')));
    },
  );

  test('valid session produces a subject-bound bearer header', () async {
    final now = DateTime.utc(2026, 9, 27, 12);
    final service = PatientSessionService(
      store: MemoryPatientSecretStore(),
      clock: () => now,
    );
    await service.saveSession(
      subjectId: 'subject-1',
      accessToken: 'patient.jwt',
      expiresAt: now.add(const Duration(hours: 1)),
    );

    final session = await service.currentSession();
    final headers = await service.authenticatedHeaders();

    expect(session?.subjectId, 'subject-1');
    expect(headers?['Authorization'], 'Bearer patient.jwt');
    expect(headers?['Content-Type'], 'application/json');
  });

  test(
    'expired session is unavailable and never emits an auth header',
    () async {
      final now = DateTime.utc(2026, 9, 27, 12);
      final service = PatientSessionService(
        store: MemoryPatientSecretStore(),
        clock: () => now,
      );
      await service.saveSession(
        subjectId: 'subject-1',
        accessToken: 'expired.jwt',
        expiresAt: now.subtract(const Duration(seconds: 1)),
      );

      expect(await service.currentSession(), isNull);
      expect(await service.authenticatedHeaders(), isNull);
    },
  );
}
