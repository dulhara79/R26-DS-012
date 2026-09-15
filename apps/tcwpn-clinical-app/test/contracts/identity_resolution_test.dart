import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/api/gateways.dart';

const String _base = 'https://backend.test';

CentralBackendGateway _gateway(List<http.Request> sent) =>
    CentralBackendGateway(
      ApiClient(
        _base,
        client: MockClient((request) async {
          sent.add(request);
          return http.Response('{"subject_id":"subject-aura-001"}', 200);
        }),
        bearer: () => 'test-token',
      ),
    );

void main() {
  test('Aura participant id resolves through the existing backend alias route',
      () async {
    final sent = <http.Request>[];
    final gateway = _gateway(sent);

    final id = await gateway.resolveAppUserId('P_65DC4002E7863773');

    expect(id, 'subject-aura-001');
    expect(
      sent.single.url.toString(),
      '$_base/v1/subjects/resolve?app_user_id=P_65DC4002E7863773',
    );
    expect(sent.single.method, 'GET');
  });

  test('unknown Aura participant id resolves to null', () async {
    final gateway = CentralBackendGateway(
      ApiClient(
        _base,
        client: MockClient(
          (_) async => http.Response('{"detail":"not found"}', 404),
        ),
        bearer: () => 'test-token',
      ),
    );

    expect(
      await gateway.resolveAppUserId('P_65DC4002E7863773'),
      isNull,
    );
  });

  test('legacy controller call is bridged to the same canonical GET route',
      () async {
    final sent = <http.Request>[];
    final gateway = _gateway(sent);

    // ignore: deprecated_member_use_from_same_package
    final id = await gateway.attach(
      appUserId: 'P_65DC4002E7863773',
      mrn: 'P_65DC4002E7863773',
      enrolledBy: 'DR001',
    );

    expect(id, 'subject-aura-001');
    expect(sent.single.method, 'GET');
    expect(sent.single.url.path, '/v1/subjects/resolve');
    expect(sent.single.url.queryParameters['app_user_id'], 'P_65DC4002E7863773');
  });

  test('gateway source contains no non-existent attach endpoint', () {
    final gatewaySource =
        File('lib/data/api/gateways.dart').readAsStringSync();

    expect(gatewaySource, isNot(contains('/v1/subjects/attach')));
    expect(gatewaySource, contains('resolveAppUserId'));
  });
}
