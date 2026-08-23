// lib/domain/consent.dart
//
// The consent record is deliberately write-once.
//
// There is no `copyWith`, no setter, and no update path. The only way a
// clinician's recorded acceptance changes is that a NEW version of the
// agreement is published, which forces a fresh acceptance of that new version.
// The prior record is never overwritten — it is retained as evidence of what
// was agreed, and when.
//
// `textSha256` is the important field. It is the hash of the exact agreement
// text that was on screen at the moment of acceptance. Six months later, if
// anyone asks "what did they actually agree to?", the hash answers it. Storing
// a version string alone would not: version strings can be edited after the
// fact, hashes cannot.

import 'dart:convert';

import 'package:crypto/crypto.dart';

/// Bumping this forces every user to accept again on next launch.
/// Never edit published agreement text without bumping this.
const String kAgreementVersion = '1.0.0';

/// When the current version was published. Shown in the viewer.
const String kAgreementEffectiveDate = '1 August 2026';

class ConsentRecord {
  final String agreementVersion;
  final String textSha256;
  final DateTime acceptedAtUtc;
  final String platform;
  final String appVersion;

  /// Recorded when known. Acceptance happens before sign-in, so this is
  /// backfilled on first successful authentication and never altered after.
  final String? clinicianId;

  /// Forward-only. Set when a user withdraws. Withdrawal ends access and stops
  /// further collection; it does not erase this record, and it does not by
  /// itself erase research data already analysed — see clause 10 of the
  /// Privacy Notice.
  final DateTime? withdrawnAtUtc;

  const ConsentRecord({
    required this.agreementVersion,
    required this.textSha256,
    required this.acceptedAtUtc,
    required this.platform,
    required this.appVersion,
    this.clinicianId,
    this.withdrawnAtUtc,
  });

  bool get isWithdrawn => withdrawnAtUtc != null;

  /// Valid only if it is for the current version and has not been withdrawn.
  bool get grantsAccess =>
      agreementVersion == kAgreementVersion && !isWithdrawn;

  /// Computes the anchor hash for a given agreement body.
  static String hashOf(String terms, String privacy, String version) => sha256
      .convert(utf8.encode('$version\u0000$terms\u0000$privacy'))
      .toString();

  Map<String, dynamic> toJson() => {
        'agreement_version': agreementVersion,
        'text_sha256': textSha256,
        'accepted_at_utc': acceptedAtUtc.toIso8601String(),
        'platform': platform,
        'app_version': appVersion,
        'clinician_id': clinicianId,
        'withdrawn_at_utc': withdrawnAtUtc?.toIso8601String(),
      };

  factory ConsentRecord.fromJson(Map<String, dynamic> j) => ConsentRecord(
        agreementVersion: '${j['agreement_version'] ?? ''}',
        textSha256: '${j['text_sha256'] ?? ''}',
        acceptedAtUtc: DateTime.tryParse('${j['accepted_at_utc']}')?.toUtc() ??
            DateTime.fromMillisecondsSinceEpoch(0, isUtc: true),
        platform: '${j['platform'] ?? 'unknown'}',
        appVersion: '${j['app_version'] ?? 'unknown'}',
        clinicianId: j['clinician_id'] == null ? null : '${j['clinician_id']}',
        withdrawnAtUtc: j['withdrawn_at_utc'] == null
            ? null
            : DateTime.tryParse('${j['withdrawn_at_utc']}')?.toUtc(),
      );
}
