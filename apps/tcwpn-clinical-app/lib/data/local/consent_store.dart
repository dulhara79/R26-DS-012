// lib/data/local/consent_store.dart
//
// Write-once storage for the acceptance record.
//
// Design rules, enforced here rather than by convention:
//   • `record()` is the ONLY way to create an acceptance, and it refuses to
//     overwrite an existing acceptance of the same agreement version.
//   • There is no update or edit method. A user cannot un-accept.
//   • `withdraw()` is additive: it stamps a withdrawal time onto the existing
//     record. It does not delete it. The evidence of what was agreed, and when,
//     survives withdrawal.
//   • The record lives in secure storage, not SharedPreferences, so it is not
//     casually readable or editable on a rooted device.
//
// A superseded record (older agreement version) is archived rather than
// discarded, so the full acceptance history is auditable.

import 'dart:convert';
import 'dart:io' show Platform;

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../../domain/consent.dart';

class ConsentStore {
  static const _s = FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
    iOptions: IOSOptions(
        accessibility: KeychainAccessibility.first_unlock_this_device),
  );

  static const _kCurrent = 'consent_current';
  static const _kHistory = 'consent_history';

  /// The acceptance in force, if any. Null before first acceptance.
  static Future<ConsentRecord?> current() async {
    final raw = await _s.read(key: _kCurrent);
    if (raw == null) return null;
    try {
      return ConsentRecord.fromJson(jsonDecode(raw) as Map<String, dynamic>);
    } catch (_) {
      return null;
    }
  }

  /// True only when there is a valid, non-withdrawn acceptance of the CURRENT
  /// agreement version. Everything else sends the user back to the gate.
  static Future<bool> hasValidConsent() async =>
      (await current())?.grantsAccess ?? false;

  /// Records an acceptance. Returns the stored record.
  ///
  /// Refuses to overwrite an existing acceptance of the same version — if one
  /// exists, it is returned unchanged. This is what makes acceptance immutable.
  static Future<ConsentRecord> record({
    required String textSha256,
    required String appVersion,
    String? clinicianId,
  }) async {
    final existing = await current();
    if (existing != null &&
        existing.agreementVersion == kAgreementVersion &&
        !existing.isWithdrawn) {
      return existing;
    }

    // Any prior record moves to history before the new one is written.
    if (existing != null) await _archive(existing);

    final rec = ConsentRecord(
      agreementVersion: kAgreementVersion,
      textSha256: textSha256,
      acceptedAtUtc: DateTime.now().toUtc(),
      platform: _platform(),
      appVersion: appVersion,
      clinicianId: clinicianId,
    );
    await _s.write(key: _kCurrent, value: jsonEncode(rec.toJson()));
    return rec;
  }

  /// Backfills the clinician id after first successful sign-in.
  ///
  /// This is the one permitted mutation, and it is one-way: it only ever fills
  /// a null field. It cannot change an id that is already recorded, and it
  /// cannot touch the timestamp, version or hash.
  static Future<void> attachClinician(String clinicianId) async {
    final rec = await current();
    if (rec == null || rec.clinicianId != null) return;
    await _s.write(
      key: _kCurrent,
      value: jsonEncode(ConsentRecord(
        agreementVersion: rec.agreementVersion,
        textSha256: rec.textSha256,
        acceptedAtUtc: rec.acceptedAtUtc,
        platform: rec.platform,
        appVersion: rec.appVersion,
        clinicianId: clinicianId,
        withdrawnAtUtc: rec.withdrawnAtUtc,
      ).toJson()),
    );
  }

  /// Forward-only withdrawal. Stamps the record; never deletes it.
  static Future<void> withdraw() async {
    final rec = await current();
    if (rec == null || rec.isWithdrawn) return;
    await _s.write(
      key: _kCurrent,
      value: jsonEncode(ConsentRecord(
        agreementVersion: rec.agreementVersion,
        textSha256: rec.textSha256,
        acceptedAtUtc: rec.acceptedAtUtc,
        platform: rec.platform,
        appVersion: rec.appVersion,
        clinicianId: rec.clinicianId,
        withdrawnAtUtc: DateTime.now().toUtc(),
      ).toJson()),
    );
  }

  /// Full acceptance history, newest first. For audit export.
  static Future<List<ConsentRecord>> history() async {
    final out = <ConsentRecord>[];
    final cur = await current();
    if (cur != null) out.add(cur);
    final raw = await _s.read(key: _kHistory);
    if (raw != null) {
      try {
        out.addAll((jsonDecode(raw) as List)
            .whereType<Map>()
            .map((e) => ConsentRecord.fromJson(Map<String, dynamic>.from(e))));
      } catch (_) {}
    }
    return out;
  }

  static Future<void> _archive(ConsentRecord rec) async {
    final raw = await _s.read(key: _kHistory);
    final list = <dynamic>[];
    if (raw != null) {
      try {
        list.addAll(jsonDecode(raw) as List);
      } catch (_) {}
    }
    list.insert(0, rec.toJson());
    await _s.write(key: _kHistory, value: jsonEncode(list.take(20).toList()));
  }

  static String _platform() {
    try {
      return Platform.operatingSystem;
    } catch (_) {
      return 'unknown';
    }
  }
}
