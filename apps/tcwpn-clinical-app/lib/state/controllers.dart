// lib/state/controllers.dart
//
// Two controllers with a deliberate split:
//
//   RosterController  — the patient list, alerts, and site-level support set.
//                       One instance, provided at the root.
//
//   ChartController   — everything about ONE patient. Created when a chart is
//                       opened, disposed when it closes, and holds its MRN
//                       final. No screen can read or write another patient's
//                       record through it.

import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';

import '../core/config/env.dart';
import '../data/api/api_client.dart';
import '../data/api/gateways.dart';
import '../data/local/stores.dart';
import '../domain/models.dart';
import '../core/design/tokens.dart';

const _uuid = Uuid();

// ─────────────────────────────────────────────────────────────────────────────

class RosterController extends ChangeNotifier {
  /// Warm-up only. Inference goes through the Central Backend.
  final _warmup = TcwpnWarmupGateway();
  final _backend = CentralBackendGateway();

  List<Patient> _patients = [];
  List<ClinicalAlert> _alerts = [];
  List<SupportNote> _siteSupport = [];
  final Map<String, FusionResult> _latestFusion = {};
  bool _loading = true;
  Map<String, dynamic>? _modelInfo;
  Map<String, dynamic>? _backendInfo;

  List<Patient> get patients => List.unmodifiable(_patients);
  List<ClinicalAlert> get alerts => List.unmodifiable(_alerts);
  List<SupportNote> get siteSupport => List.unmodifiable(_siteSupport);
  bool get loading => _loading;
  Map<String, dynamic>? get modelInfo => _modelInfo;

  int get unacknowledgedCount => _alerts.where((a) => !a.acknowledged).length;

  FusionResult? fusionFor(String mrn) => _latestFusion[mrn];

  /// Patients whose most recent composite sits in RED or DARK RED.
  ///
  /// A blocked (GREY) assessment has no composite and is not an escalation. It
  /// is also not a clearance — it appears in the unscored count on the caseload
  /// screen so it stays visible without being ranked.
  List<Patient> get needingReview => _patients.where((p) {
        final f = _latestFusion[p.mrn];
        if (f == null || !f.hasComposite) return false;
        return f.band == AlertBand.red || f.band == AlertBand.darkRed;
      }).toList();

  Future<void> init() async {
    _loading = true;
    notifyListeners();

    _patients = await RecordStore.loadRoster();
    _alerts = await RecordStore.loadAlerts();
    _siteSupport = await RecordStore.loadSiteSupport();

    if (_patients.isEmpty && Env.demoData) _seedDemoPatient();

    for (final p in _patients) {
      final cached = await RecordStore.cachedFusion(p.mrn);
      if (cached != null) _latestFusion[p.mrn] = cached;
    }

    _loading = false;
    notifyListeners();

    // Warm the Space in the background; never blocks first paint.
    _warmup.health().then((info) {
      _modelInfo = info;
      notifyListeners();
    });

    // Report backend reachability separately. A clinician needs to know that
    // the service is unreachable BEFORE writing a note, not after submitting it.
    _backend.health().then((info) {
      _backendInfo = info;
      notifyListeners();
    });
  }

  /// Null until the first health call returns, and null again if it failed.
  /// `backendReachable` is deliberately three-state at the call site: not yet
  /// checked, reachable, unreachable.
  Map<String, dynamic>? get backendInfo => _backendInfo;
  bool get backendReachable => _backendInfo != null;

  void _seedDemoPatient() {
    _patients = [
      Patient(
        mrn: 'DEMO-001',
        name: 'Demonstration Patient',
        age: 24,
        gender: 'Female',
        ward: 'Psychiatry OPD',
        referredOn: DateTime.now().subtract(const Duration(days: 96)),
        educationLevel: 4,
      ),
    ];
    RecordStore.saveRoster(_patients);
  }

  Future<void> refreshFusion(String mrn, FusionResult r) async {
    _latestFusion[mrn] = r;
    await RecordStore.cacheFusion(mrn, r);
    notifyListeners();
  }

  // ── Roster mutations ──────────────────────────────────────────────────────

  Future<String?> addPatient(Patient p) async {
    if (_patients.any((x) => x.mrn == p.mrn)) {
      return 'A patient with MRN ${p.mrn} is already on this device.';
    }
    _patients = [..._patients, p];
    await RecordStore.saveRoster(_patients);
    notifyListeners();
    return null;
  }

  Future<void> updatePatient(Patient p) async {
    final i = _patients.indexWhere((x) => x.mrn == p.mrn);
    if (i < 0) return;
    _patients[i] = p;
    await RecordStore.saveRoster(_patients);
    notifyListeners();
  }

  /// Removes the patient and every record in every namespace. Irreversible.
  Future<void> removePatient(String mrn) async {
    await RecordStore.purgePatient(mrn);
    _patients = _patients.where((p) => p.mrn != mrn).toList();
    _latestFusion.remove(mrn);
    _alerts = await RecordStore.loadAlerts();
    notifyListeners();
  }

  // ── Alerts ────────────────────────────────────────────────────────────────
  // Legacy local alert records remain readable until P4 migration, but no new
  // authoritative risk escalation event is generated from fusion on-device.

  Future<void> raiseAlert(ClinicalAlert a) async {
    _alerts = [a, ..._alerts];
    await RecordStore.saveAlerts(_alerts);
    notifyListeners();
  }

  Future<void> acknowledge(String id, String by) async {
    _alerts = _alerts.map((a) => a.id == id ? a.acknowledge(by) : a).toList();
    await RecordStore.saveAlerts(_alerts);
    notifyListeners();
  }

  // ── Site-level support set ────────────────────────────────────────────────

  Future<void> addSiteSupport(SupportNote n) async {
    _siteSupport = [..._siteSupport, n];
    await RecordStore.saveSiteSupport(_siteSupport);
    notifyListeners();
  }

  Future<void> removeSiteSupport(String id) async {
    _siteSupport = _siteSupport.where((n) => n.id != id).toList();
    await RecordStore.saveSiteSupport(_siteSupport);
    notifyListeners();
  }

  List<Patient> search(String query, {AlertBand? band, String? ward}) {
    var out = _patients;
    if (query.trim().isNotEmpty) {
      final q = query.toLowerCase();
      out = out
          .where((p) =>
              p.name.toLowerCase().contains(q) ||
              p.mrn.toLowerCase().contains(q))
          .toList();
    }
    if (band != null) {
      out = out.where((p) {
        final f = _latestFusion[p.mrn];
        return f != null && f.hasComposite && f.band == band;
      }).toList();
    }
    if (ward != null && ward != 'All') {
      out = out.where((p) => p.ward == ward).toList();
    }
    return out;
  }

  @override
  void dispose() {
    _warmup.dispose();
    _backend.dispose();
    super.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────

enum ChartStatus { idle, working, ready }

class ChartController extends ChangeNotifier {
  /// Final for the controller's whole lifetime. This is the fix for the
  /// cross-patient contamination in the previous build.
  final String mrn;
  final Patient patient;
  final RosterController roster;

  final _backend = CentralBackendGateway();

  ChartController({
    required this.patient,
    required this.roster,
  }) : mrn = patient.mrn;

  ChartStatus _status = ChartStatus.idle;
  List<ClinicalNote> _notes = [];
  List<SupportNote> _patientSupport = [];
  FusionResult? _fusionResult;
  String? _error;

  /// The backend's opaque id for this patient, obtained at enrolment. Every
  /// backend call after enrolment uses this rather than the MRN, so the raw
  /// identifier stops travelling once it has been exchanged once.
  String? _subjectId;

  /// Set when enrolment has just minted a code the patient still has to redeem.
  /// Until it is redeemed the patient app's readings do not join this subject,
  /// and the gate will block fusion for want of a second modality — so this is
  /// surfaced in the chart rather than shown once and forgotten.
  EnrolmentResult? _pendingPairing;

  /// True when the composite shown is a cached copy from a previous session
  /// rather than a fresh read. Never used to fabricate a composite — only to
  /// label one the server did produce, earlier.
  bool _fusionFromCache = false;

  ChartStatus get status => _status;
  List<ClinicalNote> get notes => List.unmodifiable(_notes);
  List<SupportNote> get patientSupport => List.unmodifiable(_patientSupport);
  FusionResult? get fusion => _fusionResult;
  String? get error => _error;
  String? get subjectId => _subjectId;
  bool get isEnrolled => (_subjectId ?? '').isNotEmpty;
  EnrolmentResult? get pendingPairing => _pendingPairing;
  bool get fusionFromCache => _fusionFromCache;

  void dismissPairing() {
    _pendingPairing = null;
    notifyListeners();
  }

  ClinicalNote? get latestAnalysedNote {
    for (final n in _notes) {
      if (n.result != null) return n;
    }
    return null;
  }

  int get visitCount => _notes.length;

  /// The backend's response to the most recent note submission, kept so the
  /// result screen can show the fusion outcome and the score's provenance
  /// alongside the model output.
  ClinicalNoteIngestResult? _lastIngest;
  ClinicalNoteIngestResult? get lastIngest => _lastIngest;

  /// Site notes plus this patient's own. K in the UI always reflects this.
  Future<List<SupportNote>> effectiveSupport() =>
      RecordStore.effectiveSupportSet(mrn);

  int get effectiveSupportCount =>
      roster.siteSupport.length + _patientSupport.length;

  Future<void> load() async {
    _status = ChartStatus.working;
    notifyListeners();

    _notes = await RecordStore.loadNotes(mrn);
    _patientSupport = await RecordStore.loadSupport(mrn);
    _subjectId = await RecordStore.subjectId(mrn);

    // A cached result is the LAST SERVER ANSWER, replayed. It is shown with its
    // age and replaced the moment a fresh read succeeds. It is never computed.
    _fusionResult = await RecordStore.cachedFusion(mrn);
    _fusionFromCache = _fusionResult != null;

    _status = ChartStatus.ready;
    notifyListeners();

    unawaited(refreshFusion());
  }

  // ── Enrolment ─────────────────────────────────────────────────────────────

  /// Ensures this patient exists on the backend and returns their subject_id.
  Future<String> ensureEnrolled({String? clinicianId}) async {
    if (isEnrolled) return _subjectId!;

    final resolved = await _backend.resolveMrn(mrn);
    if (resolved != null) {
      _subjectId = resolved;
      await _linkBehaviouralId(resolved);
      await RecordStore.saveSubjectId(mrn, resolved);
      notifyListeners();
      return resolved;
    }

    // Patient-first: if AURA already created the participant, resolve the
    // canonical app_user_id alias directly. There is no separate attach route.
    final participantId = mrn.trim().toUpperCase();
    if (_participantIdPattern.hasMatch(participantId)) {
      final resolvedByAppId = await _backend.resolveAppUserId(participantId);
      if (resolvedByAppId != null) {
        _subjectId = resolvedByAppId;
        await _linkBehaviouralId(resolvedByAppId);
        await RecordStore.saveSubjectId(mrn, resolvedByAppId);
        notifyListeners();
        return resolvedByAppId;
      }
    }

    final enrolment = await _backend.enrol(mrn: mrn, enrolledBy: clinicianId);
    _subjectId = enrolment.subjectId;
    await _linkBehaviouralId(enrolment.subjectId);
    _pendingPairing = enrolment;
    await RecordStore.saveSubjectId(mrn, enrolment.subjectId);
    notifyListeners();
    return enrolment.subjectId;
  }

  /// The identifier the patient app (Aura) shows as a QR is a C2 PARTICIPANT ID,
  /// in the form `P_` followed by 16 hex digits — the same shape as
  /// `C2_TEST_SUBJECT=P_65DC4002E7863773` in the backend's env.example.
  static final RegExp _participantIdPattern = RegExp(r'^P_[A-F0-9]{16}$');

  Future<void> _linkBehaviouralId(String subjectId) async {
    final candidate = mrn.trim().toUpperCase();
    if (!_participantIdPattern.hasMatch(candidate)) return;
    try {
      await _backend.registerExternalId(
        subjectId: subjectId,
        modality: Modality.c2Behavioral,
        externalId: candidate,
      );
    } on ApiException catch (e) {
      _error = 'Could not link the Aura Participant ID to this patient on the '
          'backend: ${e.message}';
    }
  }

  Future<void> linkExternalId({
    required String modality,
    required String externalId,
  }) async {
    final id = await ensureEnrolled();
    await _backend.registerExternalId(
      subjectId: id,
      modality: modality,
      externalId: externalId,
    );
  }

  // ── Fusion ────────────────────────────────────────────────────────────────

  Future<void> refreshFusion({bool force = false}) async {
    if (!Env.hasBackend) return;
    if (!isEnrolled && !force) return;

    try {
      final id = await ensureEnrolled();
      final state = await _backend.timeline(subjectId: id, mrn: mrn);
      if (state == null) return;

      _fusionResult = state;
      _fusionFromCache = false;
      _error = null;
      await RecordStore.cacheFusion(mrn, state);
      await roster.refreshFusion(mrn, state);
      // Authoritative escalation episodes are created and persisted by the
      // Central Backend. A RED/DARK RED client rendering must never mint a
      // separate local event with a new identity.
      notifyListeners();
    } on ApiException catch (e) {
      _error = e.message;
      notifyListeners();
    }
  }

  Future<void> rerunFusion() async {
    if (!Env.hasBackend) return;
    try {
      final id = await ensureEnrolled();
      await _backend.runFusion(id, trigger: 'manual');
      await refreshFusion();
    } on ApiException catch (e) {
      _error = e.message;
      notifyListeners();
    }
  }

  Future<Map<String, dynamic>?> submitTierVerdict({
    required String tierLabel,
    String? author,
    String? note,
  }) async {
    final id = _fusionResult?.fusionResultId;
    if (id == null) {
      _error = 'No fusion result to judge yet.';
      notifyListeners();
      return null;
    }
    try {
      final res = await _backend.submitVerdict(
        fusionResultId: id,
        tierLabel: tierLabel,
        author: author,
        note: note,
      );
      await refreshFusion();
      return res;
    } on ApiException catch (e) {
      _error = e.message;
      notifyListeners();
      return null;
    }
  }

  Future<Map<String, dynamic>?> askEvidence(String question) async {
    try {
      final id = await ensureEnrolled();
      return await _backend.evidence(subjectId: id, question: question);
    } on ApiException catch (e) {
      _error = e.message;
      notifyListeners();
      return null;
    }
  }

  // ── TC-WPN analysis ───────────────────────────────────────────────────────

  // ── Note lifecycle ────────────────────────────────────────────────────────

  ClinicalNote? noteById(String id) {
    for (final n in _notes) {
      if (n.id == id) return n;
    }
    return null;
  }

  List<ClinicalNote> get drafts => _notes.where((n) => n.isDraft).toList();

  Future<void> _persist() async {
    await RecordStore.saveNotes(mrn, _notes);
    notifyListeners();
  }

  Future<ClinicalNote> saveDraft({
    required String text,
    required String noteType,
    required String clinicianId,
  }) async {
    final note = ClinicalNote(
      id: _uuid.v4(),
      patientMrn: mrn,
      recordedAt: DateTime.now(),
      text: text,
      noteType: noteType,
      clinicianId: clinicianId,
      status: ClinicalNoteStatus.draft,
    );
    _notes = [note, ..._notes];
    await _persist();
    return note;
  }

  Future<ClinicalNote?> updateNote(
    String noteId, {
    String? text,
    String? noteType,
  }) async {
    final existing = noteById(noteId);
    if (existing == null) return null;

    final edited = existing.copyWith(
      text: text,
      noteType: noteType,
      updatedAt: DateTime.now(),
      clearAnalysisError: true,
      status: existing.hasBeenAnalysed
          ? ClinicalNoteStatus.analysed
          : ClinicalNoteStatus.draft,
    );
    _notes = _notes.map((n) => n.id == noteId ? edited : n).toList();
    await _persist();
    return edited;
  }

  Future<bool> deleteNote(String noteId) async {
    final before = _notes.length;
    _notes = _notes.where((n) => n.id != noteId).toList();
    if (_notes.length == before) return false;
    await _persist();
    return true;
  }

  Future<ClinicalNote> analyseStoredNote(String noteId) async {
    final note = noteById(noteId);
    if (note == null) {
      throw StateError('No note $noteId on this chart.');
    }

    _error = null;
    _status = ChartStatus.working;
    notifyListeners();

    try {
      final subject = await ensureEnrolled(clinicianId: note.clinicianId);
      final support = await effectiveSupport();

      final ingest = await _backend.submitNote(
        subjectId: subject,
        noteText: note.text,
        noteType: note.noteType,
        noteDate: note.recordedAt,
        supportSet: support,
        visitCount: visitCount,
        author: note.clinicianId,
      );

      _lastIngest = ingest;

      final analysed = note.copyWith(
        result: ingest.result,
        clearResult: ingest.result == null,
        analysedAt: DateTime.now(),
        status: ingest.result == null
            ? ClinicalNoteStatus.analysisFailed
            : ClinicalNoteStatus.analysed,
        clearAnalysisError: ingest.result != null,
        lastAnalysisError: ingest.result == null
            ? 'The note was submitted, but no assessment was returned.'
            : null,
      );
      _notes = _notes.map((n) => n.id == noteId ? analysed : n).toList();
      await RecordStore.saveNotes(mrn, _notes);

      if (!ingest.scored) {
        _error = ingest.needsSupportSet
            ? 'The note was stored, but no labelled examples exist for this '
                'patient, so no comparison could be made. Add labelled notes to '
                'enable analysis.'
            : 'The note was stored, but the clinical model did not return a '
                'usable score (${ingest.status})'
                '${ingest.scoreProvenance == null ? '' : ': ${ingest.scoreProvenance}'}.';
      }

      _status = ChartStatus.ready;
      notifyListeners();

      await refreshFusion();
      return analysed;
    } on ApiException catch (e) {
      final failed = note.copyWith(
        status: ClinicalNoteStatus.analysisFailed,
        lastAnalysisError: e.message,
      );
      _notes = _notes.map((n) => n.id == noteId ? failed : n).toList();
      await RecordStore.saveNotes(mrn, _notes);

      _error = e.message;
      _status = ChartStatus.ready;
      notifyListeners();
      rethrow;
    }
  }

  Future<ClinicalNote> analyseNote({
    required String text,
    required String noteType,
    required String clinicianId,
    bool skipAnalysis = false,
  }) async {
    final draft = await saveDraft(
      text: text,
      noteType: noteType,
      clinicianId: clinicianId,
    );
    if (skipAnalysis) return draft;
    return analyseStoredNote(draft.id);
  }

  Future<void> recordVerdict(
      String noteId, String verdict, String? comment) async {
    _notes = _notes
        .map((n) => n.id == noteId
            ? n.copyWith(clinicianVerdict: verdict, clinicianComment: comment)
            : n)
        .toList();
    await RecordStore.saveNotes(mrn, _notes);
    notifyListeners();
  }

  // ── Patient-specific support set ──────────────────────────────────────────

  Future<void> addSupport(SupportNote n) async {
    _patientSupport = [..._patientSupport, n];
    await RecordStore.saveSupport(mrn, _patientSupport);
    notifyListeners();
  }

  Future<void> removeSupport(String id) async {
    _patientSupport = _patientSupport.where((n) => n.id != id).toList();
    await RecordStore.saveSupport(mrn, _patientSupport);
    notifyListeners();
  }

  Future<void> promoteNoteToSupport(ClinicalNote note, String label) =>
      addSupport(SupportNote(
        id: _uuid.v4(),
        text: note.text,
        label: label,
        noteDate: note.recordedAt,
        addedAt: DateTime.now(),
        patientMrn: mrn,
        addedByClinician: note.clinicianId,
      ));

  @override
  void dispose() {
    _backend.dispose();
    super.dispose();
  }
}

void unawaited(Future<void> f) {}
