import 'dart:convert';

import 'package:anxiety_mobile_app/services/api_service.dart';
import 'package:anxiety_mobile_app/services/c4_demographics.dart';
import 'package:anxiety_mobile_app/services/patient_session_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _MemoryStore implements PatientSecretStore {
  final values = <String, String>{};

  @override
  Future<String?> read(String key) async => values[key];

  @override
  Future<void> write(String key, String value) async => values[key] = value;

  @override
  Future<void> delete(String key) async => values.remove(key);
}

void main() {
  const participantId = 'P_0123456789ABCDEF';
  const subjectId = 'subject-a';
  const root = 'https://backend.example';

  setUp(() {
    SharedPreferences.setMockInitialValues({
      'participant_id': participantId,
      'central_subject_id': subjectId,
    });
  });

  Future<PatientSessionService> authenticatedSession() async {
    final session = PatientSessionService(store: _MemoryStore());
    await session.saveSession(
      subjectId: subjectId,
      accessToken: 'patient.jwt',
      expiresAt: DateTime.now().toUtc().add(const Duration(hours: 1)),
    );
    return session;
  }

  C4Demographics validDemographics() {
    return C4Demographics.fromSavedProfile({
      'age': '21',
      'gender': 'Female',
      'c4_education': "bachelor's degree",
      'c4_smoke': 'never smokes',
      'c4_drink': 'never drinks',
    })!;
  }

  test(
    'profile submission sends exact C4 fields, no fabricated GAD-7',
    () async {
      final session = await authenticatedSession();
      var requests = 0;
      final client = MockClient((request) async {
        requests++;
        expect(request.url.path, '/v1/ingest/contextual');
        expect(request.method, 'POST');
        expect(request.headers['Authorization'], startsWith('Bearer '));
        final payload = jsonDecode(request.body) as Map<String, dynamic>;
        expect(payload, {
          'subject_id': subjectId,
          'app_user_id': participantId,
          'age': 21,
          'gender': 'female',
          'edu': "bachelor's degree",
          'smoke': 'never smokes',
          'drink': 'never drinks',
        });
        expect(payload.containsKey('gad7_items'), isFalse);
        return http.Response(
          jsonEncode({
            'subject_id': subjectId,
            'status': 'ok',
            'score': 0.018284733498692758,
          }),
          200,
        );
      });

      final result = await ApiService.submitContextualIntake(
        participantId: participantId,
        demographics: validDemographics(),
        client: client,
        sessionService: session,
        backendBase: root,
      );

      expect(requests, 1);
      expect(result.accepted, isTrue, reason: result.reason);
      expect(result.c4Scored, isTrue);
    },
  );

  test(
    'real completed GAD-7 reaches backend even without eligible C4',
    () async {
      final session = await authenticatedSession();
      const answers = <int>[0, 1, 2, 3, 1, 0, 2];
      final client = MockClient((request) async {
        final payload = jsonDecode(request.body) as Map<String, dynamic>;
        expect(payload, {
          'subject_id': subjectId,
          'app_user_id': participantId,
          'gad7_items': answers,
        });
        return http.Response(
          jsonEncode({
            'subject_id': subjectId,
            'status': 'poor_signal',
            'score': null,
            'gad7_total': 9,
          }),
          200,
        );
      });

      final result = await ApiService.submitContextualIntake(
        participantId: participantId,
        gad7Items: answers,
        client: client,
        sessionService: session,
        backendBase: root,
      );

      expect(result.accepted, isTrue);
      expect(result.c4Scored, isFalse);
    },
  );

  test('backend 503 never becomes a false successful profile sync', () async {
    final session = await authenticatedSession();
    final client = MockClient((_) async => http.Response('Unavailable', 503));

    final result = await ApiService.submitContextualIntake(
      participantId: participantId,
      demographics: validDemographics(),
      client: client,
      sessionService: session,
      backendBase: root,
    );

    expect(result.accepted, isFalse);
    expect(result.c4Scored, isFalse);
    expect(result.reason, contains('503'));
  });

  test('200 with wrong subject cannot confirm a C4 sync', () async {
    final session = await authenticatedSession();
    final client = MockClient(
      (_) async => http.Response(
        jsonEncode({
          'subject_id': 'other-subject',
          'status': 'ok',
          'score': 0.8,
        }),
        200,
      ),
    );

    final result = await ApiService.submitContextualIntake(
      participantId: participantId,
      demographics: validDemographics(),
      client: client,
      sessionService: session,
      backendBase: root,
    );

    expect(result.accepted, isFalse);
    expect(result.c4Scored, isFalse);
  });

  test(
    'no completed input or malformed GAD-7 cannot generate requests',
    () async {
      final session = await authenticatedSession();
      var requests = 0;
      final client = MockClient((_) async {
        requests++;
        return http.Response('unexpected', 200);
      });

      final missing = await ApiService.submitContextualIntake(
        participantId: participantId,
        client: client,
        sessionService: session,
        backendBase: root,
      );
      final incomplete = await ApiService.submitContextualIntake(
        participantId: participantId,
        gad7Items: <int>[0, 1, 2],
        client: client,
        sessionService: session,
        backendBase: root,
      );
      final outOfRange = await ApiService.submitContextualIntake(
        participantId: participantId,
        gad7Items: <int>[0, 1, 2, 3, 4, 1, 2],
        client: client,
        sessionService: session,
        backendBase: root,
      );

      expect(missing.accepted, isFalse);
      expect(incomplete.accepted, isFalse);
      expect(outOfRange.accepted, isFalse);
      expect(requests, 0);
    },
  );

  test('a different participant cannot reuse this device session', () async {
    final session = await authenticatedSession();
    var requests = 0;
    final client = MockClient((_) async {
      requests++;
      return http.Response('unexpected', 200);
    });

    final result = await ApiService.submitContextualIntake(
      participantId: 'P_FFFFFFFFFFFFFFFF',
      demographics: validDemographics(),
      client: client,
      sessionService: session,
      backendBase: root,
    );

    expect(result.accepted, isFalse);
    expect(requests, 0);
  });
}
