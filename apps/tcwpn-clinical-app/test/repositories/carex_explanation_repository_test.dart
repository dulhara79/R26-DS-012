import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/carex_explanation_repository.dart';

void main() {
  test('CARE-X uses assigned-patient backend GET with clinician bearer',
      () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-test-jwt',
      client: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path,
            '/v1/doctor/patients/subject-001/explanation');
        expect(request.url.query, isEmpty);
        expect(request.headers['Authorization'],
            startsWith('Bearer '));
        return http.Response(
          jsonEncode({
            'subject_id': 'subject-001',
            'fusion_result_id': 123,
            'assessed': true,
            'headline': 'Stored fusion explanation',
          }),
          200,
        );
      }),
    );

    final result = await CentralBackendCarexExplanationRepository(api)
        .explanation('subject-001');
    expect(result['fusion_result_id'], 123);
    expect(result['headline'], 'Stored fusion explanation');
  });

  test('403 for an unassigned patient is not converted into an explanation',
      () async {
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-test-jwt',
      client: MockClient((_) async => http.Response(
            jsonEncode({'detail': 'Patient not assigned to clinician'}),
            403,
          )),
    );

    await expectLater(
      CentralBackendCarexExplanationRepository(api)
          .explanation('other-subject'),
      throwsA(isA<ApiException>()
          .having((e) => e.kind, 'kind', ApiFailure.forbidden)),
    );
  });

  test('blank patient identity never sends a CARE-X request', () async {
    var calls = 0;
    final api = ApiClient(
      'https://backend.test',
      bearer: () => 'clinician-test-jwt',
      client: MockClient((_) async {
        calls++;
        return http.Response('{}', 200);
      }),
    );
    expect(
      () => CentralBackendCarexExplanationRepository(api).explanation('  '),
      throwsArgumentError,
    );
    expect(calls, 0);
  });
}
