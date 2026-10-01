import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/p5b_repositories.dart';

void main() {
  test(
    'timeline repository reads canonical assessment history and event markers',
    () async {
      late Uri requested;
      late Map<String, String> headers;
      final client = MockClient((request) async {
        requested = request.url;
        headers = request.headers;
        return http.Response(
          jsonEncode({
            'assessments': [
              {
                'fusion_result_id': 123,
                'current_assessment': {
                  'score': 0.58,
                  'tier': 'Medium',
                  'band': 'AMBER',
                },
                'assessment_status': 'complete',
                'modalities': [
                  {'component_id': 'c2_behavioral', 'available': false},
                ],
                'computed_at': '2026-09-16T08:00:00Z',
                'forecast': {
                  'score': 0.84,
                  'tier': 'High',
                  'scope': 'physiological',
                  'horizon_minutes': 10,
                },
              },
            ],
            'events': [
              {
                'id': 'evt-1',
                'fusion_result_id': 123,
                'event_type': 'acute_escalation_forecast',
                'status': 'OPEN',
                'created_at': '2026-09-16T09:00:00Z',
              },
            ],
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      });
      final api = ApiClient(
        'https://central.example',
        client: client,
        bearer: () => 'prototype-token',
      );
      final repository = CentralBackendTimelineRepository(api);

      final rows = await repository.history('subject-001', limit: 200);

      expect(
        requested.toString(),
        'https://central.example/v1/patients/subject-001/assessments',
      );
      expect(headers['authorization'], 'Bearer prototype-token');
      expect(rows, hasLength(2));
      expect(rows.last.composite, 0.58);
      expect(rows.last.assessmentStatus, 'complete');
      expect(rows.last.forecastScore, 0.84);
      expect(rows.last.forecastScope, 'physiological');
      expect(rows.last.missingModalities, ['c2_behavioral']);
      expect(rows.first.eventId, 'evt-1');
      expect(rows.first.composite, isNull);
      expect(rows.first.fusionResultId, 123);
    },
  );

  test('timeline repository reads assessments only and does not turn current state into history', () async {
    final client = MockClient(
      (request) async => http.Response(
        jsonEncode({
          'composite': 0.91,
          'tier': 'High',
          'band': 'RED',
          'assessments': [],
        }),
        200,
        headers: {'content-type': 'application/json'},
      ),
    );
    final repository = CentralBackendTimelineRepository(
      ApiClient('https://central.example', client: client, bearer: () => ''),
    );

    final rows = await repository.history('subject-001');

    expect(rows, isEmpty);
  });

  test('timeline repository preserves null history values', () async {
    final client = MockClient(
      (request) async => http.Response(
        jsonEncode({
          'assessments': [
            {
              'current_assessment': null,
              'assessment_status': 'insufficient',
              'computed_at': null,
              'trigger': 'manual',
            },
          ],
        }),
        200,
        headers: {'content-type': 'application/json'},
      ),
    );
    final repository = CentralBackendTimelineRepository(
      ApiClient('https://central.example', client: client, bearer: () => ''),
    );

    final row = (await repository.history('subject-001')).single;

    expect(row.composite, isNull);
    expect(row.tier, isNull);
    expect(row.band, isNull);
    expect(row.computedAt, isNull);
  });
}
