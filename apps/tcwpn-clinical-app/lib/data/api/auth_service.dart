// lib/data/api/auth_service.dart
//
// Two modes, chosen at build time by whether AUTH_BASE is set.
//
//   REMOTE — the Space verifies credentials, issues sessions, sends OTP email,
//            and handles registration and password reset. Required before any
//            real patient data.
//
//   LOCAL  — credentials compiled into the build. Development and demos only.
//            Registration and password reset are unavailable, and the app says
//            so rather than showing controls that cannot work.

import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';

import '../../core/config/env.dart';
import 'api_client.dart';

class AuthSession {
  final String clinicianId;
  final String displayName;
  final String token;

  const AuthSession({
    required this.clinicianId,
    required this.displayName,
    required this.token,
  });
}

/// Thrown when the server rejects an action for a reason the clinician needs to
/// read verbatim — "awaiting approval", "code expired", "3 attempts remaining".
/// Flattening these into a generic failure would leave people stuck.
class AuthMessage implements Exception {
  final String message;
  const AuthMessage(this.message);
  @override
  String toString() => message;
}

class AuthService {
  static const String _base = String.fromEnvironment('AUTH_BASE');
  static const String _salt =
      String.fromEnvironment('AUTH_SALT', defaultValue: 'r26-ds012-local-salt');
  static const String _localAccounts = String.fromEnvironment('AUTH_LOCAL');

  static bool get isLocalMode => _base.isEmpty;
  static bool get supportsSelfService => !isLocalMode;
  static bool get shouldWarnInsecure => isLocalMode && kReleaseMode;

  static ApiClient _client() => ApiClient(_base);

  static void _ensureSelfServiceConfigured() {
    if (isLocalMode) {
      throw const AuthMessage(
        'Self-service account actions are disabled in local mode. '
        'Set AUTH_BASE to the ClinAnx auth service to use register/reset.',
      );
    }

    final authHost = Uri.tryParse(_base)?.host.toLowerCase() ?? '';
    final tcwpnHost = Uri.tryParse(Env.tcwpnBase)?.host.toLowerCase() ?? '';
    if (authHost.isNotEmpty && authHost == tcwpnHost) {
      throw AuthMessage(
        'AUTH_BASE is currently set to the TC-WPN model service ($_base). '
        'Password reset and registration must use your auth backend '
        '(for example: https://<org>-clinanx-auth.hf.space).',
      );
    }
  }

  // ── SIGN IN ──────────────────────────────────────────────────────────────

  /// Returns a session, or null when the credentials are simply wrong.
  /// Throws [AuthMessage] when the server has something specific to say.
  static Future<AuthSession?> signIn({
    required String clinicianId,
    required String password,
  }) async {
    if (isLocalMode) return _local(clinicianId, password);

    final api = _client();
    try {
      final json = await api.post(
        '/auth/login',
        {'clinician_id': clinicianId, 'password': password},
        timeout: const Duration(seconds: 30),
      );
      final token = '${json['access_token'] ?? ''}';
      if (token.isEmpty) return null;
      return AuthSession(
        clinicianId: '${json['clinician_id'] ?? clinicianId}',
        displayName: '${json['display_name'] ?? clinicianId}',
        token: token,
      );
    } on ApiException catch (e) {
      // 401 is a wrong password. 403 is "verify your email" or "awaiting
      // approval" — a different situation, and the clinician must see it.
      if (e.statusCode == 401) return null;
      if (e.statusCode == 403) throw AuthMessage(e.detail);
      rethrow;
    } finally {
      api.close();
    }
  }

  // ── REGISTRATION ─────────────────────────────────────────────────────────

  /// Creates an account and triggers the verification email.
  /// Returns the email the code was sent to.
  static Future<String> register({
    required String clinicianId,
    required String displayName,
    required String email,
    required String password,
    required String inviteCode,
  }) async {
    _ensureSelfServiceConfigured();
    final api = _client();
    try {
      final json = await api.post(
        '/auth/register',
        {
          'clinician_id': clinicianId,
          'display_name': displayName,
          'email': email,
          'password': password,
          'invite_code': inviteCode,
        },
        timeout: const Duration(seconds: 45),
      );
      return '${json['email'] ?? email}';
    } on ApiException catch (e) {
      throw AuthMessage(e.detail.isEmpty ? e.message : e.detail);
    } finally {
      api.close();
    }
  }

  /// Confirms the emailed code. Returns the server's guidance on what happens
  /// next — normally that approval is pending.
  static Future<String> verifyEmail({
    required String clinicianId,
    required String code,
  }) async {
    _ensureSelfServiceConfigured();
    final api = _client();
    try {
      final json = await api.post(
        '/auth/verify',
        {'clinician_id': clinicianId, 'code': code},
        timeout: const Duration(seconds: 30),
      );
      return '${json['next'] ?? 'Your email is verified.'}';
    } on ApiException catch (e) {
      throw AuthMessage(e.detail.isEmpty ? e.message : e.detail);
    } finally {
      api.close();
    }
  }

  static Future<void> resendVerification(String clinicianId) async {
    _ensureSelfServiceConfigured();
    final api = _client();
    try {
      await api.post('/auth/resend', {'clinician_id': clinicianId, 'code': ''},
          timeout: const Duration(seconds: 30));
    } on ApiException catch (e) {
      throw AuthMessage(e.detail.isEmpty ? e.message : e.detail);
    } finally {
      api.close();
    }
  }

  // ── PASSWORD RESET ───────────────────────────────────────────────────────

  /// Requests a reset code.
  ///
  /// The server answers identically whether or not the address is registered,
  /// so this cannot be used to discover who is on the study. The UI must not
  /// imply otherwise.
  static Future<void> requestReset(String email) async {
    _ensureSelfServiceConfigured();
    final api = _client();
    try {
      await api.post('/auth/forgot-password', {'email': email},
          timeout: const Duration(seconds: 45));
    } on ApiException catch (e) {
      throw AuthMessage(e.detail.isEmpty ? e.message : e.detail);
    } finally {
      api.close();
    }
  }

  static Future<void> resetPassword({
    required String email,
    required String code,
    required String newPassword,
  }) async {
    _ensureSelfServiceConfigured();
    final api = _client();
    try {
      await api.post(
        '/auth/reset-password',
        {'email': email, 'code': code, 'new_password': newPassword},
        timeout: const Duration(seconds: 30),
      );
    } on ApiException catch (e) {
      throw AuthMessage(e.detail.isEmpty ? e.message : e.detail);
    } finally {
      api.close();
    }
  }

  // ── LOCAL MODE ───────────────────────────────────────────────────────────

  static String digest(String password) =>
      sha256.convert(utf8.encode('$_salt$password')).toString();

  static AuthSession? _local(String id, String password) {
    final entries = _localAccounts.isEmpty
        ? _fallbackAccounts()
        : _localAccounts.split(';');
    final wanted = digest(password);
    for (final e in entries) {
      final parts = e.split('|');
      if (parts.length != 3) continue;
      if (parts[0].toUpperCase() != id.toUpperCase()) continue;
      if (parts[2].trim() != wanted) return null;
      return AuthSession(
        clinicianId: parts[0].toUpperCase(),
        displayName: parts[1],
        token: 'local-session-${DateTime.now().millisecondsSinceEpoch}',
      );
    }
    return null;
  }

  /// Debug-only, so a fresh clone runs with no flags. Empty in release builds,
  /// so a forgotten flag cannot ship an app with a known password.
  static List<String> _fallbackAccounts() {
    if (kReleaseMode) return const [];
    return [
      'DR001|Dr D. Kaushalya|${digest('clinanx-dev')}',
      'DR002|Dr C. Suraweera|${digest('clinanx-dev')}',
    ];
  }

  static const String devPassword = 'clinanx-dev';
}
