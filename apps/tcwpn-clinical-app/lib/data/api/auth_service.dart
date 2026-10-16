// lib/data/api/auth_service.dart
//
// Central Backend issues clinician JWTs. Local accounts are for debug demos only.
// Central Backend does not expose clinician registration or password reset.

import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';

import '../../core/config/env.dart';
import 'api_client.dart';

class AuthSession {
  final String clinicianId;
  final String displayName;
  final String token;
  final String? role;
  final DateTime? expiresAt;

  const AuthSession({
    required this.clinicianId,
    required this.displayName,
    required this.token,
    this.role,
    this.expiresAt,
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
  // Env is the single build-time configuration source. Keeping the auth defines
  // here as a second String.fromEnvironment set previously allowed documentation
  // and production code to drift independently.
  static const String _salt = Env.authSalt;
  static const String _localAccounts = Env.authLocalAccounts;

  static bool get isLocalMode => !kReleaseMode && Env.backendBase.isEmpty;
  static bool get supportsSelfService => false;
  static bool get shouldWarnInsecure => false;

  static Never _unsupportedSelfService() {
    throw const AuthMessage(
      'Clinician account changes are managed by the study team. '
      'Contact the study team for registration or password reset.',
    );
  }

  // ── SIGN IN ──────────────────────────────────────────────────────────────

  /// Returns a session, or null when the credentials are simply wrong.
  /// Throws [AuthMessage] when the server has something specific to say.
  static Future<AuthSession?> signIn({
    required String clinicianId,
    required String password,
  }) async {
    if (isLocalMode) return _local(clinicianId, password);
    if (Env.backendBase.trim().isEmpty) {
      throw const AuthMessage(
        'This build has no Central Backend configured. Contact the study team.',
      );
    }

    final api = ApiClient(Env.backendBase);
    try {
      final json = await api.post('/auth/login', {
        'clinician_id': clinicianId,
        'password': password,
      }, timeout: const Duration(seconds: 30));
      final token = '${json['access_token'] ?? ''}';
      final identity = json['clinician'] is Map
          ? Map<String, dynamic>.from(json['clinician'] as Map)
          : const <String, dynamic>{};
      final expiresAt = DateTime.tryParse('${json['expires_at'] ?? ''}')
          ?.toUtc();
      final canonicalId = '${identity['clinician_id'] ?? ''}'.trim();
      if (token.isEmpty ||
          canonicalId.isEmpty ||
          expiresAt == null ||
          !expiresAt.isAfter(DateTime.now().toUtc())) {
        throw const ApiException(
          kind: ApiFailure.malformed,
          detail: 'Clinician login response is missing identity, token or a valid expiry.',
        );
      }
      return AuthSession(
        clinicianId: canonicalId,
        displayName: '${identity['display_name'] ?? canonicalId}',
        token: token,
        role: identity['role']?.toString(),
        expiresAt: expiresAt,
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

  /// Retained for older screens; the Central Backend has no account-creation contract.
  static Future<String> register({
    required String clinicianId,
    required String displayName,
    required String email,
    required String password,
    required String inviteCode,
  }) async {
    _unsupportedSelfService();
  }

  /// Retained for older screens; disabled with the same account guidance.
  static Future<String> verifyEmail({
    required String clinicianId,
    required String code,
  }) async {
    _unsupportedSelfService();
  }

  static Future<void> resendVerification(String clinicianId) async {
    _unsupportedSelfService();
  }

  // ── PASSWORD RESET ───────────────────────────────────────────────────────

  /// Retained for older screens; disabled until an authenticated contract exists.
  static Future<void> requestReset(String email) async {
    _unsupportedSelfService();
  }

  static Future<void> resetPassword({
    required String email,
    required String code,
    required String newPassword,
  }) async {
    _unsupportedSelfService();
  }

  // ── LOCAL MODE ───────────────────────────────────────────────────────────

  static String digest(String password) =>
      sha256.convert(utf8.encode('$_salt$password')).toString();

  static AuthSession? _local(String id, String password) {
    final entries = _localAccounts.isEmpty
        ? const <String>[]
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
}
