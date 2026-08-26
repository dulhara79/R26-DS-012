// lib/data/api/api_client.dart
//
// One HTTP client for every service. Replaces the three divergent clients in the
// previous build (one that decoded before checking status, one that swallowed
// every error into a print, one with no auth header at all).
//
// Guarantees:
//   • status is checked before the body is decoded
//   • non-JSON error bodies produce a readable message, not a FormatException
//   • every request carries auth when a token is configured
//   • cold-start retries are bounded and only applied to idempotent reads

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../../core/security/secure_http.dart';

import '../../core/config/env.dart';
import 'session.dart';

enum ApiFailure {
  offline,
  timeout,
  unauthorized,
  notFound,
  validation,
  server,
  malformed,

  /// No base URL was compiled into this build. Distinct from [notFound]: the
  /// service was never addressed, so nothing was reached and nothing 404'd.
  /// Reporting this as "endpoint not found" sends the clinician to a Settings
  /// screen that cannot fix it — the fix is a rebuild with --dart-define.
  notConfigured,

  /// The server's certificate was not signed by a pinned authority.
  /// Almost always an intercepting proxy, occasionally a rotated CA.
  insecureConnection,
  unknown,
}

class ApiException implements Exception {
  final ApiFailure kind;
  final int statusCode;
  final String detail;
  final String endpoint;

  const ApiException({
    required this.kind,
    this.statusCode = 0,
    this.detail = '',
    this.endpoint = '',
  });

  /// Written for a clinician, not a developer: what happened and what to do.
  String get message => switch (kind) {
        ApiFailure.offline =>
          'No network connection. The note is saved on this device and can be analysed once you are back online.',
        ApiFailure.timeout =>
          'The model did not respond in time. This usually means the service is starting up — try again in a moment.',
        // 401 from the model service after a successful sign-in almost always
        // means the 12-hour session has lapsed, not that the device was never
        // authorised. Telling a clinician to "contact the study administrator"
        // when they simply need to sign in again wastes everyone's time.
        ApiFailure.unauthorized =>
          'Your session has expired. Sign out and sign in again. If that does '
              'not help, contact the study team.',
        // A 404 here is an APP-SIDE or DEPLOYMENT fault: the route this build
        // asks for is not the route the running service serves. A clinician can
        // do nothing about that, and telling them to "check the service address
        // in Settings" invites them to edit a value that is compiled in. Say
        // what is true — it is broken, it is not their doing, and their work is
        // intact — and give the study team the one word they need.
        ApiFailure.notFound =>
          'The clinical service could not be reached at the address this app was '
              'built with. Nothing you entered has been lost. Please report this '
              'to the study team (routing error).',
        ApiFailure.notConfigured =>
          'This build has no clinical service configured, so nothing can be '
              'analysed. Your work is saved on this device. The study team needs '
              'to reinstall a configured build.',
        ApiFailure.validation => 'The service rejected this request. $detail',
        ApiFailure.server =>
          'The clinical service reported an internal error. Nothing was saved on '
              'the server; your note is still safe on this device.',
        ApiFailure.malformed =>
          'The service returned a response this app could not read. Report this with the time it happened.',
        // Deliberately specific. "Check your connection" would send a clinician
        // on a hospital network with an inspecting proxy chasing the wrong
        // problem for an hour.
        ApiFailure.insecureConnection =>
          'The connection was refused because the server\'s security '
              'certificate could not be verified. This network may be '
              'inspecting traffic. Do not submit patient data on it — switch '
              'to mobile data or another network, and tell the study team.',
        ApiFailure.unknown => detail.isEmpty ? 'Something went wrong.' : detail,
      };

  /// Pin failures are never retried. Retrying an intercepted connection just
  /// hands the interceptor more attempts.
  bool get isRetryable =>
      kind == ApiFailure.timeout ||
      kind == ApiFailure.offline ||
      kind == ApiFailure.server;

  @override
  String toString() =>
      'ApiException(${kind.name}, $statusCode, $endpoint): $detail';
}

class ApiClient {
  final String baseUrl;
  final http.Client _http;

  /// Supplies the bearer token for this service, evaluated per request.
  ///
  /// Different services want different credentials, and sending the wrong one
  /// is a silent 401 rather than a visible error:
  ///
  ///   Central Backend — a single shared app token (main.py::_auth compares the
  ///                     header against one static BACKEND_API_TOKEN). The
  ///                     clinician's JWT is NOT what it checks.
  ///   auth service    — the clinician's session JWT.
  ///   TC-WPN /health  — no credential; /health is unauthenticated.
  ///
  /// Defaults to the session JWT when signed in, and to no Authorization header
  /// otherwise.
  final String Function() _bearer;

  /// The client is chosen by base URL: a host in the pin set gets a client
  /// whose trust store contains only the pinned roots, so an intercepting proxy
  /// fails the handshake before any note text is written to the socket.
  ///
  /// Injecting a client (for tests) bypasses pinning, which is correct — a test
  /// double is not a network path.
  ApiClient(this.baseUrl, {http.Client? client, String Function()? bearer})
      : _http = client ?? SecureHttp.clientFor(baseUrl),
        _bearer = bearer ?? _defaultBearer;

  /// No privileged service token ships in the APK. Anything inside an APK is
  /// extractable, and the only call this app makes without a session is the
  /// TC-WPN /health warm-up, which needs no credential at all. A gateway that
  /// genuinely needs a token now passes one explicitly through the `bearer`
  /// constructor argument, which puts every credential at its call site.
  static String _defaultBearer() => Session.isActive ? Session.token! : '';

  /// Omits Authorization entirely when there is no credential, rather than
  /// sending `Bearer ` with an empty value — an empty bearer is a 401 that
  /// reads like a wrong password instead of like a missing session.
  Map<String, String> get _headers {
    final bearer = _bearer();
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      if (bearer.isNotEmpty) 'Authorization': 'Bearer $bearer',
    };
  }

  Future<Map<String, dynamic>> post(
    String path,
    Map<String, dynamic> body, {
    Duration? timeout,
  }) =>
      _send(
        () => _http
            .post(_uri(path), headers: _headers, body: jsonEncode(body))
            .timeout(timeout ?? Env.inferenceTimeout),
        path,
      );

  Future<Map<String, dynamic>> get(
    String path, {
    Duration? timeout,
    int retries = 1,
  }) async {
    ApiException? last;
    for (var attempt = 0; attempt <= retries; attempt++) {
      try {
        return await _send(
          () => _http
              .get(_uri(path), headers: _headers)
              .timeout(timeout ?? Env.quickTimeout),
          path,
        );
      } on ApiException catch (e) {
        last = e;
        if (!e.isRetryable || attempt == retries) rethrow;
        await Future.delayed(Duration(milliseconds: 600 * (attempt + 1)));
      }
    }
    throw last!;
  }

  Uri _uri(String path) =>
      Uri.parse('${baseUrl.replaceAll(RegExp(r'/+$'), '')}$path');

  Future<Map<String, dynamic>> _send(
    Future<http.Response> Function() run,
    String endpoint,
  ) async {
    if (baseUrl.isEmpty) {
      throw ApiException(
        kind: ApiFailure.notConfigured,
        endpoint: endpoint,
        detail: 'No base URL configured for this service.',
      );
    }

    late http.Response res;
    try {
      res = await run();
    } on TimeoutException {
      throw ApiException(kind: ApiFailure.timeout, endpoint: endpoint);
    } on HandshakeException catch (e) {
      // Must precede SocketException: HandshakeException is not a subtype, but
      // treating a pin failure as "you are offline" would hide an active
      // interception attempt behind a routine-looking message.
      throw ApiException(
        kind: ApiFailure.insecureConnection,
        endpoint: endpoint,
        detail: e.message,
      );
    } on PinningNotConfigured catch (e) {
      throw ApiException(
        kind: ApiFailure.insecureConnection,
        endpoint: endpoint,
        detail: e.toString(),
      );
    } on SocketException {
      throw ApiException(kind: ApiFailure.offline, endpoint: endpoint);
    } on http.ClientException catch (e) {
      throw ApiException(
          kind: ApiFailure.offline, endpoint: endpoint, detail: e.message);
    } catch (e) {
      throw ApiException(
          kind: ApiFailure.unknown, endpoint: endpoint, detail: e.toString());
    }

    // Status first, body second. Always.
    if (res.statusCode >= 400) {
      throw ApiException(
        kind: switch (res.statusCode) {
          401 || 403 => ApiFailure.unauthorized,
          404 => ApiFailure.notFound,
          422 || 400 => ApiFailure.validation,
          >= 500 => ApiFailure.server,
          _ => ApiFailure.unknown,
        },
        statusCode: res.statusCode,
        endpoint: endpoint,
        detail: _extractDetail(res.body),
      );
    }

    try {
      final decoded = jsonDecode(res.body);
      if (decoded is Map<String, dynamic>) return decoded;
      return {'data': decoded};
    } catch (_) {
      throw ApiException(
        kind: ApiFailure.malformed,
        statusCode: res.statusCode,
        endpoint: endpoint,
        detail: res.body.substring(0, res.body.length.clamp(0, 200)),
      );
    }
  }

  /// FastAPI puts errors under `detail`; some Spaces use `error`. Fall back to
  /// a truncated raw body rather than showing the user a stack trace.
  String _extractDetail(String body) {
    try {
      final j = jsonDecode(body);
      if (j is Map) {
        final d = j['detail'] ?? j['error'] ?? j['message'];
        if (d != null) return d.toString();
      }
    } catch (_) {}
    return body.substring(0, body.length.clamp(0, 200));
  }

  void close() => _http.close();
}
