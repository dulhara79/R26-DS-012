import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/api/session.dart';
import 'package:r26_ds012_app/data/repositories/assignment_invite_repository.dart';

void main() {
  tearDown(Session.clear);

  test('redeems only the patient-issued code with clinician bearer', () async {
    Session.set(token: 'clinician-jwt', clinicianId: 'DR001');
    final api = ApiClient(
      'https://central.example',
      bearer: () => 'clinician-jwt',
      client: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/v1/clinicians/me/assignments');
        expect(request.headers['Authorization'], 'Bearer clinician-jwt');
        expect(jsonDecode(request.body), {'invite_code': 'aB_123-Z'});
        return http.Response(
          jsonEncode({
            'clinician_id': 'DR001',
            'subject_id': 'canonical-subject-001',
            'active': true,
          }),
          200,
        );
      }),
    );

    final result = await CentralBackendAssignmentInviteRepository(api)
        .redeem('  aB_123-Z  ');
    expect(result.subjectId, 'canonical-subject-001');
  });

  test(
    'does not accept a different clinician or inactive assignment',
    () async {
      Session.set(token: 'clinician-jwt', clinicianId: 'DR001');
      for (final body in [
        {'clinician_id': 'DR002', 'subject_id': 'S1', 'active': true},
        {'clinician_id': 'DR001', 'subject_id': 'S1', 'active': false},
        {'clinician_id': 'DR001', 'subject_id': '', 'active': true},
      ]) {
        final api = ApiClient(
          'https://central.example',
          client: MockClient((_) async => http.Response(jsonEncode(body), 200)),
        );
        await expectLater(
          CentralBackendAssignmentInviteRepository(api).redeem('valid-code'),
          throwsA(
            isA<ApiException>().having(
              (e) => e.kind,
              'kind',
              ApiFailure.malformed,
            ),
          ),
        );
      }
    },
  );

  test(
    'passes 401, 403, 404, 409 and expired 410 through without retry',
    () async {
      for (final status in [401, 403, 404, 409, 410]) {
        var calls = 0;
        final api = ApiClient(
          'https://central.example',
          client: MockClient((_) async {
            calls++;
            return http.Response(
              '{"detail":"state or access failure"}',
              status,
            );
          }),
        );
        await expectLater(
          CentralBackendAssignmentInviteRepository(api).redeem('valid-code'),
          throwsA(
            isA<ApiException>().having((e) => e.statusCode, 'status', status),
          ),
        );
        expect(calls, 1);
      }
    },
  );

  test('empty or oversized code never reaches the network', () async {
    var calls = 0;
    final api = ApiClient(
      'https://central.example',
      client: MockClient((_) async {
        calls++;
        return http.Response('{}', 200);
      }),
    );
    final repository = CentralBackendAssignmentInviteRepository(api);
    for (final code in ['  ', 'x' * 129]) {
      await expectLater(repository.redeem(code), throwsA(isA<ArgumentError>()));
    }
    expect(calls, 0);
  });
}
