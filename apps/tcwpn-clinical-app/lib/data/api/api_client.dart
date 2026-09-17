// lib/data/api/api_client.dart

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../../core/config/env.dart';
import '../../core/security/clinical_endpoint_policy.dart';
import '../../core/security/secure_http.dart';
import 'session.dart';

enum ApiFailure {
  offline,
  timeout,
  unauthorized,
  forbidden,
  notFound,
  conflict,
  validation,
  server,
  malformed,
  notConfigured,
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

  String get message => switch (kind) {
        ApiFailure.offline =>
          'No network connection. The note is saved on this device and can be analysed once you are back online.',
        ApiFailure.timeout =>
          'The service did not respond in time. Try again in a moment.',
        ApiFailure.unauthorized =>
          'Your session has expired. Sign out and sign in again. If that does not help, contact the study team.',
        ApiFailure.forbidden =>
          'You are signed in, but you do not have permission to access this patient or action.',
        ApiFailure.notFound =>
          'The clinical service could not be reached at the address this app was built with. Please report this to the study team.',
        ApiFailure.conflict =>
          'This record changed on the server. Refresh to see the current state before trying again.',
        ApiFailure.notConfigured =>
          'This build has no clinical service configured. The study team needs to install a configured build.',
        ApiFailure.validation => detail.isEmpty
            ? 'The service rejected this request.'
            : 'The service rejected this request. $detail',
        ApiFailure.server =>
          'The clinical service reported an internal error. Your local work has not been replaced by a fabricated result.',
        ApiFailure.malformed =>
          'The service returned a response this app could not read. Report this with the time it happened.',
        ApiFailure.insecureConnection =>
          'The connection was refused because the configured clinical endpoint is not using an approved secure transport or its certificate could not be verified.',
        ApiFailure.unknown => detail.isEmpty ? 'Something went wrong.' : detail,
      };

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
  final String Function() _bearer;

  ApiClient(this.baseUrl, {http.Client? client, String Function()? bearer})
      : _http = client ?? SecureHttp.clientFor(baseUrl),
        _bearer = bearer ?? _defaultBearer;

  static String _defaultBearer() => Session.isActive ? Session.token! : '';

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
    if (!ClinicalEndpointPolicy.isAllowed(baseUrl)) {
      throw ApiException(
        kind: ApiFailure.insecureConnection,
        endpoint: endpoint,
        detail: 'Remote clinical endpoints must use HTTPS.',
      );
    }

    late http.Response res;
    try {
      res = await run();
    } on TimeoutException {
      throw ApiException(kind: ApiFailure.timeout, endpoint: endpoint);
    } on HandshakeException catch (e) {
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
        kind: ApiFailure.offline,
        endpoint: endpoint,
        detail: e.message,
      );
    } catch (e) {
      throw ApiException(
        kind: ApiFailure.unknown,
        endpoint: endpoint,
        detail: 'Unexpected client error.',
      );
    }

    if (res.statusCode >= 400) {
      throw ApiException(
        kind: switch (res.statusCode) {
          401 => ApiFailure.unauthorized,
          403 => ApiFailure.forbidden,
          404 => ApiFailure.notFound,
          409 => ApiFailure.conflict,
          422 || 400 => ApiFailure.validation,
          >= 500 => ApiFailure.server,
          _ => ApiFailure.unknown,
        },
        statusCode: res.statusCode,
        endpoint: endpoint,
        detail: _extractSafeDetail(res.body),
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
        detail: 'Response body was not valid JSON.',
      );
    }
  }

  String _extractSafeDetail(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map) {
        final value = decoded['detail'] ?? decoded['error'] ?? decoded['message'];
        if (value is String) {
          final text = value.trim();
          final lower = text.toLowerCase();
          if (text.length <= 300 &&
              !text.contains('\n') &&
              !lower.contains('traceback') &&
              !lower.contains('stack trace')) {
            return text;
          }
        }
      }
    } catch (_) {}
    return '';
  }

  void close() => _http.close();
}
