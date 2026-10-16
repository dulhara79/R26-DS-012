import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';

const _base = 'https://backend.test';

Future<ApiException> _failureForStatus(int status) async {
  final api = ApiClient(
    _base,
    client: MockClient((_) async => http.Response('{"detail":"test"}', status)),
  );

  try {
    await api.get('/v1/test', retries: 0);
  } on ApiException catch (e) {
    return e;
  }
  throw StateError('Expected ApiException for HTTP $status');
}

void main() {
  test('401 is an expired or invalid session', () async {
    final failure = await _failureForStatus(401);
    expect(failure.kind, ApiFailure.unauthorized);
    expect(failure.message.toLowerCase(), contains('session'));
  });

  test('403 is permission failure without expired-session wording', () async {
    final failure = await _failureForStatus(403);
    expect(failure.kind, ApiFailure.forbidden);
    expect(failure.message.toLowerCase(), contains('permission'));
    expect(failure.message.toLowerCase(), isNot(contains('expired')));
  });

  test('409 remains a distinct server-state conflict', () async {
    final failure = await _failureForStatus(409);
    expect(failure.kind, ApiFailure.conflict);
    expect(failure.message.toLowerCase(), contains('refresh'));
  });

  test('422 remains request validation', () async {
    final failure = await _failureForStatus(422);
    expect(failure.kind, ApiFailure.validation);
  });
}
