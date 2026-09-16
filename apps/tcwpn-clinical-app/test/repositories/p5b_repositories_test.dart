import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/p5b_repositories.dart';

void main() {
  test('timeline repository calls only verified clinician timeline route', () async {
    late Uri requested;
    late Map<String, String> headers;
    final client = MockClient((request) async {
      requested = request.url;
      headers = request.headers;
      return http.Response(
        jsonEncode({
          'trend': [
            {
              'composite': 0.58,
              'tier': 'Medium',
              'band': 'AMBER',
              'assessment_status': 'complete',
              'missing_modalities': ['c2_behavioral'],
              'computed_at': '2026-09-16T08:00:00Z',
              'trigger': 'note-ingest',
            }
          ]
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
      'https://central.example/v1/doctor/patients/subject-001/timeline?limit=200',
    );
    expect(headers['authorization'], 'Bearer prototype-token');
    expect(rows, hasLength(1));
    expect(rows.single.composite, 0.58);
    expect(rows.single.assessmentStatus, 'complete');
  });

  test('timeline repository reads trend only and does not turn current state into history',
      () async {
    final client = MockClient((request) async => http.Response(
          jsonEncode({
            'composite': 0.91,
            'tier': 'High',
            'band': 'RED',
            'trend': [],
          }),
          200,
          headers: {'content-type': 'application/json'},
        ));
    final repository = CentralBackendTimelineRepository(
      ApiClient('https://central.example', client: client, bearer: () => ''),
    );

    final rows = await repository.history('subject-001');

    expect(rows, isEmpty);
  });

  test('timeline repository preserves null history values', () async {
    final client = MockClient((request) async => http.Response(
          jsonEncode({
            'trend': [
              {
                'composite': null,
                'tier': null,
                'band': null,
                'assessment_status': 'insufficient',
                'computed_at': null,
                'trigger': 'manual',
              }
            ]
          }),
          200,
          headers: {'content-type': 'application/json'},
        ));
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
