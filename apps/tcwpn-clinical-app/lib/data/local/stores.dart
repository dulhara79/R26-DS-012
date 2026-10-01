// lib/data/local/stores.dart
//
// Two stores, deliberately separated:
//
//   SecureStore  — credentials and session tokens. flutter_secure_storage,
//                  which is Keychain on iOS and EncryptedSharedPreferences on
//                  Android. Never SharedPreferences.
//
//   RecordStore  — clinical records. Every key is namespaced by the currently
//                  authenticated clinician, and patient-specific records are
//                  additionally namespaced by MRN. Legacy unscoped keys are
//                  deliberately not read as a fallback because ownership cannot
//                  be proven after an upgrade.

import 'dart:convert';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../domain/models.dart';
import 'clinician_storage_scope.dart';

// ─────────────────────────────────────────────────────────────────────────────

class SecureStore {
  static const _s = FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
    iOptions: IOSOptions(
        accessibility: KeychainAccessibility.first_unlock_this_device),
  );

  static const _kClinicianId = 'clinician_id';
  static const _kClinicianName = 'clinician_name';
  static const _kSessionToken = 'session_token';
  static const _kSessionExpiresAt = 'session_expires_at';

  static Future<void> saveSession({
    required String clinicianId,
    required String clinicianName,
    required String token,
    DateTime? expiresAt,
  }) async {
    await _s.write(key: _kClinicianId, value: clinicianId);
    await _s.write(key: _kClinicianName, value: clinicianName);
    await _s.write(key: _kSessionToken, value: token);
    if (expiresAt != null) {
      await _s.write(
        key: _kSessionExpiresAt,
        value: expiresAt.toUtc().toIso8601String(),
      );
    }
  }

  static Future<String?> clinicianId() => _s.read(key: _kClinicianId);
  static Future<String?> clinicianName() => _s.read(key: _kClinicianName);
  static Future<String?> token() => _s.read(key: _kSessionToken);
  static Future<DateTime?> expiresAt() async =>
      DateTime.tryParse(await _s.read(key: _kSessionExpiresAt) ?? '')?.toUtc();

  static Future<bool> hasSession() async {
    final present = (await _s.read(key: _kSessionToken))?.isNotEmpty ?? false;
    final expiry = await expiresAt();
    if (present && expiry != null && !DateTime.now().toUtc().isBefore(expiry)) {
      await signOut();
      return false;
    }
    return present;
  }

  static Future<void> signOut() async => _s.deleteAll();
}

// ─────────────────────────────────────────────────────────────────────────────

class RecordStore {
  static Future<SharedPreferences> get _p => SharedPreferences.getInstance();

  static String get _kRoster =>
      ClinicianStorageScope.keyForCurrent('roster_v2');
  static String get _kAlerts =>
      ClinicianStorageScope.keyForCurrent('alerts_v2');
  static String get _kSiteSupport =>
      ClinicianStorageScope.keyForCurrent('support_site_v2');

  static String _notes(String mrn) =>
      ClinicianStorageScope.keyForCurrent('notes_v2::$mrn');
  static String _support(String mrn) =>
      ClinicianStorageScope.keyForCurrent('support_v2::$mrn');

  // v3: the cached fusion payload is now the backend's clinician-timeline shape,
  // not the old fusion-service shape. The key is bumped so a cache written by a
  // previous build is ignored rather than misparsed — an old `composite_score`
  // blob read by the new parser would yield a null composite and look like a
  // blocked gate, which is a different clinical statement entirely.
  static String _fusion(String mrn) =>
      ClinicianStorageScope.keyForCurrent('fusion_v3::$mrn');

  static String _subject(String mrn) =>
      ClinicianStorageScope.keyForCurrent('subject_id_v1::$mrn');

  // ── Backend subject id (per patient) ──────────────────────────────────────

  static Future<String?> subjectId(String mrn) async {
    final v = (await _p).getString(_subject(mrn));
    return (v == null || v.isEmpty) ? null : v;
  }

  static Future<void> saveSubjectId(String mrn, String subjectId) async =>
      (await _p).setString(_subject(mrn), subjectId);

  // ── Roster ────────────────────────────────────────────────────────────────

  static Future<List<Patient>> loadRoster() async {
    final raw = (await _p).getString(_kRoster);
    if (raw == null) return [];
    try {
      return (jsonDecode(raw) as List)
          .whereType<Map>()
          .map((e) => Patient.fromJson(Map<String, dynamic>.from(e)))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Future<void> saveRoster(List<Patient> patients) async =>
      (await _p).setString(
          _kRoster, jsonEncode(patients.map((p) => p.toJson()).toList()));

  // ── Clinical notes (per patient) ──────────────────────────────────────────

  static Future<List<ClinicalNote>> loadNotes(String mrn) async {
    final raw = (await _p).getString(_notes(mrn));
    if (raw == null) return [];
    try {
      return (jsonDecode(raw) as List)
          .whereType<Map>()
          .map((e) => ClinicalNote.fromJson(Map<String, dynamic>.from(e)))
          .toList()
        ..sort((a, b) => b.recordedAt.compareTo(a.recordedAt));
    } catch (_) {
      return [];
    }
  }

  static Future<void> saveNotes(String mrn, List<ClinicalNote> notes) async =>
      (await _p).setString(
          _notes(mrn), jsonEncode(notes.map((n) => n.toJson()).toList()));

  // ── Support set ───────────────────────────────────────────────────────────

  static Future<List<SupportNote>> loadSupport(String mrn) =>
      _loadSupportAt(_support(mrn));

  static Future<List<SupportNote>> loadSiteSupport() =>
      _loadSupportAt(_kSiteSupport);

  static Future<List<SupportNote>> _loadSupportAt(String key) async {
    final raw = (await _p).getString(key);
    if (raw == null) return [];
    try {
      return (jsonDecode(raw) as List)
          .whereType<Map>()
          .map((e) => SupportNote.fromJson(Map<String, dynamic>.from(e)))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Future<void> saveSupport(String mrn, List<SupportNote> notes) async =>
      (await _p).setString(
          _support(mrn), jsonEncode(notes.map((n) => n.toJson()).toList()));

  static Future<void> saveSiteSupport(List<SupportNote> notes) async =>
      (await _p).setString(
          _kSiteSupport, jsonEncode(notes.map((n) => n.toJson()).toList()));

  /// Site notes first, then patient-specific ones. TC-WPN's temporal weighting
  /// handles ordering internally; this just guarantees both tiers are present.
  static Future<List<SupportNote>> effectiveSupportSet(String mrn) async =>
      [...await loadSiteSupport(), ...await loadSupport(mrn)];

  // ── Cached fusion result ──────────────────────────────────────────────────

  /// Caches the server's answer verbatim, via FusionResult.toJson, which emits
  /// exactly the keys FusionResult.fromJson reads.
  static Future<void> cacheFusion(String mrn, FusionResult r) async =>
      (await _p).setString(_fusion(mrn), jsonEncode(r.toJson()));

  static Future<FusionResult?> cachedFusion(String mrn) async {
    final raw = (await _p).getString(_fusion(mrn));
    if (raw == null) return null;
    try {
      return FusionResult.fromJson(
          Map<String, dynamic>.from(jsonDecode(raw)), mrn);
    } catch (_) {
      return null;
    }
  }

  // ── Alerts ────────────────────────────────────────────────────────────────

  static Future<List<ClinicalAlert>> loadAlerts() async {
    final raw = (await _p).getString(_kAlerts);
    if (raw == null) return [];
    try {
      return (jsonDecode(raw) as List)
          .whereType<Map>()
          .map((e) => ClinicalAlert.fromJson(Map<String, dynamic>.from(e)))
          .toList()
        ..sort((a, b) => b.raisedAt.compareTo(a.raisedAt));
    } catch (_) {
      return [];
    }
  }

  static Future<void> saveAlerts(List<ClinicalAlert> alerts) async =>
      (await _p).setString(_kAlerts,
          jsonEncode(alerts.take(200).map((a) => a.toJson()).toList()));

  // ── Deletion ──────────────────────────────────────────────────────────────

  /// Removes every locally cached record belonging to one patient within the
  /// current clinician's namespace. It never deletes another clinician's copy.
  static Future<void> purgePatient(String mrn) async {
    final p = await _p;
    await p.remove(_notes(mrn));
    await p.remove(_support(mrn));
    await p.remove(_fusion(mrn));
    await p.remove(_subject(mrn));

    final alerts = await loadAlerts();
    await saveAlerts(alerts.where((a) => a.patientMrn != mrn).toList());

    final roster = await loadRoster();
    await saveRoster(roster.where((x) => x.mrn != mrn).toList());
  }
}
