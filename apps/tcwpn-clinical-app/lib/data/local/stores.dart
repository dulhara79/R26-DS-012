// lib/data/local/stores.dart
//
// Two stores, deliberately separated:
//
//   SecureStore  — credentials and session tokens. flutter_secure_storage,
//                  which is Keychain on iOS and EncryptedSharedPreferences on
//                  Android. Never SharedPreferences.
//
//   RecordStore  — clinical records. Every read and write takes the patient MRN
//                  as an argument. There is no `activePatientId` static: the
//                  previous build had one, and a stale value meant one patient's
//                  assessment could be written into another patient's chart.
//                  Making the key a required parameter removes that failure mode
//                  at the type level.

import 'dart:convert';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../domain/models.dart';
import '../../core/design/tokens.dart';

// ─────────────────────────────────────────────────────────────────────────────

class SecureStore {
  static const _s = FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
    iOptions: IOSOptions(accessibility: KeychainAccessibility.first_unlock_this_device),
  );

  static const _kClinicianId = 'clinician_id';
  static const _kClinicianName = 'clinician_name';
  static const _kSessionToken = 'session_token';

  static Future<void> saveSession({
    required String clinicianId,
    required String clinicianName,
    required String token,
  }) async {
    await _s.write(key: _kClinicianId, value: clinicianId);
    await _s.write(key: _kClinicianName, value: clinicianName);
    await _s.write(key: _kSessionToken, value: token);
  }

  static Future<String?> clinicianId() => _s.read(key: _kClinicianId);
  static Future<String?> clinicianName() => _s.read(key: _kClinicianName);
  static Future<String?> token() => _s.read(key: _kSessionToken);

  static Future<bool> hasSession() async =>
      (await _s.read(key: _kSessionToken))?.isNotEmpty ?? false;

  static Future<void> signOut() async => _s.deleteAll();
}

// ─────────────────────────────────────────────────────────────────────────────

class RecordStore {
  static Future<SharedPreferences> get _p => SharedPreferences.getInstance();

  // Roster is global; everything else is namespaced by MRN.
  static const _kRoster = 'roster_v2';
  static const _kAlerts = 'alerts_v2';
  static const _kSiteSupport = 'support_site_v2';

  static String _notes(String mrn) => 'notes_v2::$mrn';
  static String _support(String mrn) => 'support_v2::$mrn';
  static String _fusion(String mrn) => 'fusion_v2::$mrn';

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
  //
  // Two tiers, matching how few-shot adaptation is actually used: site-level
  // notes seed every patient, and per-patient notes refine the prototype for
  // one individual. `effectiveSupportSet` merges them.

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

  static Future<void> cacheFusion(String mrn, FusionResult r) async =>
      (await _p).setString(
          _fusion(mrn),
          jsonEncode({
            'composite_score': r.compositeScore,
            'alert_level': r.band.protocolName,
            'computed_at': r.computedAt.toIso8601String(),
            'weights': {for (final c in r.contributions) c.key: c.weight},
            'scores': {for (final c in r.contributions) c.key: c.score},
            'renormalised': r.renormalised,
            'modalities_available': r.modalitiesAvailable,
          }));

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
      (await _p).setString(
          _kAlerts, jsonEncode(alerts.take(200).map((a) => a.toJson()).toList()));

  // ── Deletion ──────────────────────────────────────────────────────────────

  /// Removes every record belonging to one patient across every namespace.
  /// The previous build's delete dialog promised this and cleared only the
  /// roster entry, leaving notes and assessments orphaned on disk.
  static Future<void> purgePatient(String mrn) async {
    final p = await _p;
    await p.remove(_notes(mrn));
    await p.remove(_support(mrn));
    await p.remove(_fusion(mrn));

    final alerts = await loadAlerts();
    await saveAlerts(alerts.where((a) => a.patientMrn != mrn).toList());

    final roster = await loadRoster();
    await saveRoster(roster.where((x) => x.mrn != mrn).toList());
  }
}
