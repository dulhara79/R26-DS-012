import 'dart:convert';
import 'dart:math';

import 'package:crypto/crypto.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'participant_identity_service.dart';

/// Lightweight local-only authentication for demonstration builds.
///
/// This deliberately does not use JWTs, refresh tokens, server sessions, or a
/// remote identity provider. Accounts are stored on this device only.
///
/// Passwords are never stored in plain text: a random salt and SHA-256 digest
/// are persisted instead. This is still a demo mechanism and must not be used
/// as production authentication.
class DemoAuthService {
  static const String _accountsKey = 'demo_auth_accounts_v1';

  static String normalizeEmail(String email) => email.trim().toLowerCase();

  static Future<DemoAuthResult> signUp({
    required String displayName,
    required String email,
    required int age,
    required String password,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final accounts = _readAccounts(prefs);
    final normalizedEmail = normalizeEmail(email);

    if (accounts.any((a) => a.email == normalizedEmail)) {
      return const DemoAuthResult.failure(
        'An account with this email already exists on this device.',
      );
    }
    if (accounts.isNotEmpty) {
      return const DemoAuthResult.failure(
        'This demo keeps one local account per device. Log in with the existing account.',
      );
    }

    final participantId =
        await ParticipantIdentityService.createForDisplayName(displayName);
    final salt = _generateSalt();
    final passwordHash = _hashPassword(password, salt);

    final account = DemoAccount(
      email: normalizedEmail,
      displayName: displayName.trim(),
      age: age,
      participantId: participantId,
      salt: salt,
      passwordHash: passwordHash,
      createdAt: DateTime.now().toUtc(),
    );

    accounts.add(account);
    await _writeAccounts(prefs, accounts);
    await _activateAccount(prefs, account);

    return DemoAuthResult.success(account);
  }

  static Future<DemoAuthResult> login({
    required String email,
    required String password,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final accounts = _readAccounts(prefs);
    final normalizedEmail = normalizeEmail(email);

    DemoAccount? account;
    for (final item in accounts) {
      if (item.email == normalizedEmail) {
        account = item;
        break;
      }
    }

    if (account == null) {
      return const DemoAuthResult.failure(
        'No local demo account was found for this email.',
      );
    }

    final attemptedHash = _hashPassword(password, account.salt);
    if (attemptedHash != account.passwordHash) {
      return const DemoAuthResult.failure('Incorrect password.');
    }

    await _activateAccount(prefs, account);
    return DemoAuthResult.success(account);
  }

  static Future<void> _activateAccount(
    SharedPreferences prefs,
    DemoAccount account,
  ) async {
    await prefs.setString(
      ParticipantIdentityService.participantIdKey,
      account.participantId,
    );
    await prefs.setString(
      ParticipantIdentityService.displayNameKey,
      account.displayName,
    );
    await prefs.setString('user_id', account.participantId);
    await prefs.setString('demo_auth_email', account.email);

    // Prefill the existing research profile form without silently marking it
    // complete. The participant still reviews and submits the full profile.
    Map<String, dynamic> profile = <String, dynamic>{};
    final rawProfile = prefs.getString('user_profile_data');
    if (rawProfile != null && rawProfile.isNotEmpty) {
      try {
        final decoded = jsonDecode(rawProfile);
        if (decoded is Map) {
          profile = Map<String, dynamic>.from(decoded);
        }
      } catch (_) {
        profile = <String, dynamic>{};
      }
    }
    profile['age'] = account.age.toString();
    await prefs.setString('user_profile_data', jsonEncode(profile));
  }

  static List<DemoAccount> _readAccounts(SharedPreferences prefs) {
    final raw = prefs.getString(_accountsKey);
    if (raw == null || raw.isEmpty) return <DemoAccount>[];

    try {
      final decoded = jsonDecode(raw);
      if (decoded is! List) return <DemoAccount>[];
      return decoded
          .whereType<Map>()
          .map((item) => DemoAccount.fromJson(Map<String, dynamic>.from(item)))
          .toList();
    } catch (_) {
      return <DemoAccount>[];
    }
  }

  static Future<void> _writeAccounts(
    SharedPreferences prefs,
    List<DemoAccount> accounts,
  ) async {
    await prefs.setString(
      _accountsKey,
      jsonEncode(accounts.map((a) => a.toJson()).toList()),
    );
  }

  static String _generateSalt() {
    final random = Random.secure();
    final bytes = List<int>.generate(16, (_) => random.nextInt(256));
    return base64UrlEncode(bytes);
  }

  static String _hashPassword(String password, String salt) {
    return sha256.convert(utf8.encode('$salt:$password')).toString();
  }
}

class DemoAccount {
  final String email;
  final String displayName;
  final int age;
  final String participantId;
  final String salt;
  final String passwordHash;
  final DateTime createdAt;

  const DemoAccount({
    required this.email,
    required this.displayName,
    required this.age,
    required this.participantId,
    required this.salt,
    required this.passwordHash,
    required this.createdAt,
  });

  factory DemoAccount.fromJson(Map<String, dynamic> json) {
    return DemoAccount(
      email: json['email']?.toString() ?? '',
      displayName: json['display_name']?.toString() ?? '',
      age: (json['age'] as num?)?.toInt() ?? 0,
      participantId: json['participant_id']?.toString() ?? '',
      salt: json['salt']?.toString() ?? '',
      passwordHash: json['password_hash']?.toString() ?? '',
      createdAt:
          DateTime.tryParse(json['created_at']?.toString() ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
    );
  }

  Map<String, dynamic> toJson() => {
    'email': email,
    'display_name': displayName,
    'age': age,
    'participant_id': participantId,
    'salt': salt,
    'password_hash': passwordHash,
    'created_at': createdAt.toIso8601String(),
  };
}

class DemoAuthResult {
  final DemoAccount? account;
  final String? error;

  const DemoAuthResult._({this.account, this.error});

  const DemoAuthResult.failure(String message)
    : this._(account: null, error: message);

  factory DemoAuthResult.success(DemoAccount account) =>
      DemoAuthResult._(account: account);

  bool get isSuccess => account != null;
}
