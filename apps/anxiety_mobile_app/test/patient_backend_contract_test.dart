import 'dart:convert';

import 'package:anxiety_mobile_app/services/api_service.dart';
import 'package:anxiety_mobile_app/services/patient_session_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

class MemoryStore implements PatientSecretStore {
  final values = <String, String>{};
  @override
  Future<String?> read(String key) async => values[key];
  @override
  Future<void> write(String key, String value) async => values[key] = value;
  @override
  Future<void> delete(String key) async => values.remove(key);
}

void main() {
  setUp(() => SharedPreferences.setMockInitialValues({}));
  test(
    'first clinician-created subject claim sends pairing proof to /self',
    () async {
      final session = PatientSessionService(store: MemoryStore());
      final client = MockClient((request) async {
        expect(request.url.path, '/v1/subjects/self');
        final payload = jsonDecode(request.body) as Map<String, dynamic>;
        expect(payload['app_user_id'], 'P_0123456789ABCDEF');
        expect(payload['pairing_code'], 'ABCD-EFGH');
        expect(
          (payload['installation_secret'] as String).length,
          greaterThan(32),
        );
        return http.Response(
          jsonEncode({
            'subject_id': 'subject-a',
            'access_token': 'patient.jwt',
            'expires_at': DateTime.now()
                .toUtc()
                .add(const Duration(hours: 1))
                .toIso8601String(),
          }),
          200,
        );
      });
      final subject = await ApiService.selfEnrol(
        'P_0123456789ABCDEF',
        pairingCode: 'abcd-efgh',
        client: client,
        sessionService: session,
        backendBase: 'https://backend.example',
      );
      expect(subject, 'subject-a');
      expect((await session.currentSession())?.subjectId, subject);
    },
  );

  test(
    'an enrolled patient can create an assignment invite for the clinician',
    () async {
      final session = PatientSessionService(store: MemoryStore());
      await session.saveSession(
        subjectId: 'subject-a',
        accessToken: 'patient.jwt',
        expiresAt: DateTime.now().toUtc().add(const Duration(hours: 1)),
      );
      final client = MockClient((request) async {
        expect(request.url.path, '/v1/patients/me/assignment-invites');
        expect(request.headers['Authorization'], 'Bearer patient.jwt');
        expect(jsonDecode(request.body), <String, dynamic>{});
        return http.Response(
          jsonEncode({
            'invite_code': 'one-time-proof',
            'expires_at': DateTime.now()
                .toUtc()
                .add(const Duration(minutes: 10))
                .toIso8601String(),
          }),
          200,
        );
      });
      final invite = await ApiService.createAssignmentInvite(
        client: client,
        sessionService: session,
        backendBase: 'https://backend.example',
      );
      expect(invite?.code, 'one-time-proof');
    },
  );

  test(
    'a renewal cannot silently switch the canonical patient subject',
    () async {
      final session = PatientSessionService(store: MemoryStore());
      final client = MockClient(
        (request) async => http.Response(
          jsonEncode({
            'subject_id': 'other-subject',
            'access_token': 'wrong.jwt',
            'expires_at': DateTime.now()
                .toUtc()
                .add(const Duration(hours: 1))
                .toIso8601String(),
          }),
          200,
        ),
      );
      final subject = await ApiService.selfEnrol(
        'P_0123456789ABCDEF',
        expectedSubjectId: 'subject-a',
        client: client,
        sessionService: session,
        backendBase: 'https://backend.example',
      );
      expect(subject, isNull);
      expect(await session.currentSession(), isNull);
    },
  );

  test('401 renews the same subject once and retries its assessment', () async {
    final session = PatientSessionService(store: MemoryStore());
    await session.saveSession(
      subjectId: 'subject-a',
      accessToken: 'expired.jwt',
      expiresAt: DateTime.now().toUtc().add(const Duration(hours: 1)),
    );
    var gets = 0;
    var renewals = 0;
    final client = MockClient((request) async {
      if (request.url.path == '/v1/subjects/self') {
        renewals++;
        return http.Response(
          jsonEncode({
            'subject_id': 'subject-a',
            'access_token': 'new.jwt',
            'expires_at': DateTime.now()
                .toUtc()
                .add(const Duration(hours: 1))
                .toIso8601String(),
          }),
          200,
        );
      }
      expect(request.url.path, '/v1/patients/subject-a/risk');
      gets++;
      if (gets == 1) {
        expect(request.headers['Authorization'], 'Bearer expired.jwt');
        return http.Response('Unauthorized', 401);
      }
      expect(request.headers['Authorization'], 'Bearer new.jwt');
      return http.Response(
        jsonEncode({
          'subject_id': 'subject-a',
          'fusion_result_id': 123,
          'composite': 0.58,
          'tier': 'Medium',
          'band': 'AMBER',
        }),
        200,
      );
    });
    final response = await ApiService.getPatientRisk(
      'subject-a',
      client: client,
      sessionService: session,
      participantId: 'P_0123456789ABCDEF',
      backendBase: 'https://backend.example',
    );
    expect(response?['fusion_result_id'], 123);
    expect((await session.currentSession())?.accessToken, 'new.jwt');
    expect(gets, 2);
    expect(renewals, 1);
  });

  test('invitation POST retries only after a rejected expired JWT', () async {
    final session = PatientSessionService(store: MemoryStore());
    await session.saveSession(
      subjectId: 'subject-a',
      accessToken: 'expired.jwt',
      expiresAt: DateTime.now().toUtc().add(const Duration(hours: 1)),
    );
    var posts = 0;
    final client = MockClient((request) async {
      if (request.url.path == '/v1/subjects/self') {
        expect(request.method, 'POST');
        return http.Response(
          jsonEncode({
            'subject_id': 'subject-a',
            'access_token': 'new.jwt',
            'expires_at': DateTime.now()
                .toUtc()
                .add(const Duration(hours: 1))
                .toIso8601String(),
          }),
          200,
        );
      }
      expect(request.url.path, '/v1/patients/me/assignment-invites');
      expect(jsonDecode(request.body), <String, dynamic>{});
      posts++;
      if (posts == 1) return http.Response('Unauthorized', 401);
      expect(request.headers['Authorization'], 'Bearer new.jwt');
      return http.Response(
        jsonEncode({
          'invite_code': 'fresh-code',
          'expires_at': DateTime.now()
              .toUtc()
              .add(const Duration(minutes: 10))
              .toIso8601String(),
        }),
        200,
      );
    });
    final invite = await ApiService.createAssignmentInvite(
      client: client,
      sessionService: session,
      participantId: 'P_0123456789ABCDEF',
      backendBase: 'https://backend.example',
    );
    expect(invite?.code, 'fresh-code');
    expect(posts, 2);
  });
}
