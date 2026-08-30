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

  // v3: the cached fusion payload is now the backend's clinician-timeline shape,
  // not the old fusion-service shape. The key is bumped so a cache written by a
  // previous build is ignored rather than misparsed — an old `composite_score`
  // blob read by the new parser would yield a null composite and look like a
  // blocked gate, which is a different clinical statement entirely.
  static String _fusion(String mrn) => 'fusion_v3::$mrn';

  static String _subject(String mrn) => 'subject_id_v1::$mrn';

  // ── Backend subject id (per patient) ──────────────────────────────────────
  //
  // The opaque id the Central Backend minted for this patient at enrolment.
  // Stored so the raw MRN stops travelling once it has been exchanged once.
  // Namespaced by MRN like every other record, for the same reason: there is no
  // `activePatient` static anywhere in this class.

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

  /// Caches the server's answer verbatim, via FusionResult.toJson, which emits
  /// exactly the keys FusionResult.fromJson reads.
  ///
  /// The previous version hand-built a different, lossy shape here — dropping
  /// status, freshness, the gate decision and the fusion row id — so the cached
  /// copy and the network copy were not the same object. Round-tripping through
  /// one parser means the two paths cannot diverge in how a field is read.
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

  /// Removes every record belonging to one patient across every namespace.
  /// The previous build's delete dialog promised this and cleared only the
  /// roster entry, leaving notes and assessments orphaned on disk.
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
