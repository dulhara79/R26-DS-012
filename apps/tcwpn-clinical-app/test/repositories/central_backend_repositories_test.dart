import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/central_backend_repositories.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

Matcher _notConfigured() => throwsA(
      isA<ApiException>().having(
        (e) => e.kind,
        'kind',
        ApiFailure.notConfigured,
      ),
    );

const _event = <String, dynamic>{
  'id': 'evt-001',
  'subject_id': 'subject-001',
  'fusion_result_id': 123,
  'forecast_result_id': 'fcst-001',
  'event_type': 'acute_escalation_forecast',
  'severity': 'high',
  'reason': 'Forecast crossed versioned escalation policy',
  'forecast_horizon': 10,
  'status': 'OPEN',
  'created_at': '2026-09-16T12:22:00Z',
  'acknowledged_at': null,
  'acknowledged_by': null,
  'resolved_at': null,
  'resolved_by': null,
  'policy_version': 'escalation-v1',
};

void main() {
  ApiClient noNetworkApi() => ApiClient(
        'https://backend.test',
        client: MockClient((request) async {
          throw StateError(
            'Unverified target adapter attempted network call: ${request.url}',
          );
        }),
        bearer: () => 'clinician-jwt',
      );

  test('auth target adapter remains blocked until backend contract is verified',
      () async {
    await expectLater(
      CentralBackendAuthRepository(noNetworkApi()).validateCurrentSession(),
      _notConfigured(),
    );
  });

  test('dashboard target adapter remains blocked until backend contract is verified',
      () async {
    await expectLater(
      CentralBackendDashboardRepository(noNetworkApi()).loadDashboard(),
      _notConfigured(),
    );
  });

  test('latest assessment target adapter remains blocked until backend contract is verified',
      () async {
    await expectLater(
      CentralBackendAssessmentRepository(noNetworkApi())
          .latestAssessment('subject-001'),
      _notConfigured(),
    );
  });

  test('openEvents uses frozen Phase 6 target contract and parses server event',
      () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path, '/v1/attention-events');
        expect(request.url.queryParameters, {'status': 'OPEN'});
        expect(request.headers['Authorization'], 'Bearer clinician-jwt');
        return http.Response(jsonEncode({'events': [_event]}), 200);
      }),
    );

    final events = await CentralBackendAttentionEventRepository(api).openEvents();

    expect(events, hasLength(1));
    expect(events.single.id, 'evt-001');
    expect(events.single.subjectId, 'subject-001');
    expect(events.single.fusionResultId, 123);
    expect(events.single.status, AttentionEventStatus.open);
    expect(events.single.policyVersion, 'escalation-v1');
  });

  test('activity supports server-side subject filter and never filters locally',
      () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path, '/v1/attention-events');
        expect(request.url.queryParameters, {'subject_id': 'subject-001'});
        return http.Response(jsonEncode({'events': [_event]}), 200);
      }),
    );

    final events = await CentralBackendAttentionEventRepository(api)
        .activity(subjectId: 'subject-001');
    expect(events.single.subjectId, 'subject-001');
  });

  test('activity without subject requests assignment-scoped history', () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path, '/v1/attention-events');
        expect(request.url.query, isEmpty);
        return http.Response(jsonEncode({'events': [_event]}), 200);
      }),
    );

    final events = await CentralBackendAttentionEventRepository(api).activity();
    expect(events, hasLength(1));
  });

  test('eventById fetches canonical event by safely encoded id', () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path, '/v1/attention-events/evt-001');
        return http.Response(jsonEncode({'event': _event}), 200);
      }),
    );

    final event = await CentralBackendAttentionEventRepository(api)
        .eventById('evt-001');
    expect(event?.id, 'evt-001');
  });

  test('acknowledge sends no client actor/time and trusts canonical response',
      () async {
    final acknowledged = Map<String, dynamic>.from(_event)
      ..['status'] = 'ACKNOWLEDGED'
      ..['acknowledged_at'] = '2026-09-16T12:24:00Z'
      ..['acknowledged_by'] = 'DR001';
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/v1/attention-events/evt-001/acknowledge');
        expect(jsonDecode(request.body), <String, dynamic>{});
        expect(request.headers['Authorization'], 'Bearer clinician-jwt');
        return http.Response(jsonEncode({'event': acknowledged}), 200);
      }),
    );

    final event = await CentralBackendAttentionEventRepository(api)
        .acknowledge('evt-001');
    expect(event.status, AttentionEventStatus.acknowledged);
    expect(event.acknowledgedBy, 'DR001');
    expect(event.acknowledgedAt?.toUtc(), DateTime.utc(2026, 9, 16, 12, 24));
  });

  test('resolve sends no client actor/time and trusts canonical response',
      () async {
    final resolved = Map<String, dynamic>.from(_event)
      ..['status'] = 'RESOLVED'
      ..['acknowledged_at'] = '2026-09-16T12:24:00Z'
      ..['acknowledged_by'] = 'DR001'
      ..['resolved_at'] = '2026-09-16T12:31:00Z'
      ..['resolved_by'] = 'DR001';
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/v1/attention-events/evt-001/resolve');
        expect(jsonDecode(request.body), <String, dynamic>{});
        return http.Response(jsonEncode({'event': resolved}), 200);
      }),
    );

    final event =
        await CentralBackendAttentionEventRepository(api).resolve('evt-001');
    expect(event.status, AttentionEventStatus.resolved);
    expect(event.resolvedBy, 'DR001');
    expect(event.resolvedAt?.toUtc(), DateTime.utc(2026, 9, 16, 12, 31));
  });

  test('malformed list success wrapper fails explicitly', () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async =>
          http.Response(jsonEncode({'events': 'not-a-list'}), 200)),
    );

    await expectLater(
      CentralBackendAttentionEventRepository(api).openEvents(),
      throwsA(isA<ApiException>().having(
        (e) => e.kind,
        'kind',
        ApiFailure.malformed,
      )),
    );
  });

  test('malformed single-event success wrapper fails explicitly', () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async =>
          http.Response(jsonEncode({'event': 'not-an-object'}), 200)),
    );

    await expectLater(
      CentralBackendAttentionEventRepository(api).eventById('evt-001'),
      throwsA(isA<ApiException>().having(
        (e) => e.kind,
        'kind',
        ApiFailure.malformed,
      )),
    );
  });

  for (final entry in <int, ApiFailure>{
    401: ApiFailure.unauthorized,
    403: ApiFailure.forbidden,
    404: ApiFailure.notFound,
    409: ApiFailure.conflict,
    422: ApiFailure.validation,
    500: ApiFailure.server,
  }.entries) {
    test('AttentionEvent transport maps HTTP ${entry.key} to ${entry.value.name}',
        () async {
      final api = ApiClient(
        'https://backend.test',
        bearer: () => 'clinician-jwt',
        client: MockClient((request) async => http.Response(
              jsonEncode({'detail': 'contract test'}),
              entry.key,
            )),
      );

      await expectLater(
        CentralBackendAttentionEventRepository(api).eventById('evt-001'),
        throwsA(isA<ApiException>().having(
          (e) => e.kind,
          'kind',
          entry.value,
        )),
      );
    });
  }
}
