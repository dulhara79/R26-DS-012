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
  Map<String, FusionResult> _latestFusion = {};
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
  ///
  /// Three paths, in order of preference:
  ///   1. already stored locally — no network call;
  ///   2. known to the backend under this MRN — resolve and store;
  ///   3. unknown — enrol, store, and hold the pairing code for display.
  ///
  /// Enrolling an MRN the backend already knows is safe: it returns the existing
  /// subject with a fresh pairing code rather than creating a second patient.
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
  ///
  /// This app uses that value as its patient key, which is fine on its own: the
  /// backend treats whatever it receives as an opaque MRN, hashes it under the
  /// pepper, and mints its own `subject_id`. But the backend then knows this
  /// patient ONLY by that UUID, and `_external_id(db, subject_id,
  /// 'c2_behavioral')` falls back to it when no alias is registered — so the
  /// backend ends up asking C2 about an id C2 has never seen, and the
  /// behavioural reading comes back empty for a patient whose real C2 id we
  /// were holding the whole time.
  ///
  /// Registering the alias is idempotent per modality. A 409 means this
  /// participant id is already mapped to a DIFFERENT subject, which is a real
  /// cross-patient error and is deliberately allowed to surface.
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
      // Surfaced, not swallowed — see the 409 case above.
      _error = 'Could not link the Aura Participant ID to this patient on the '
          'backend: ${e.message}';
    }
  }

  /// Registers the id another component knows this patient by, so the backend
  /// does not ask that service about a subject_id it has never seen.
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

  /// Reads the authoritative clinician view from the Central Backend.
  ///
  /// ClinAnx collects none of the passive modalities. C1 (wearable) and C4
  /// (intake + GAD-7) arrive from the patient app; C2 is recorded and excluded
  /// from the composite by pre-registered rule. Reading the timeline is how this
  /// app learns all of their current values, statuses and capture times.
  ///
  /// A failure leaves the previous result on screen rather than blanking it, but
  /// does NOT substitute a locally computed one — there is no local fusion path
  /// any more.
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
      await _raiseIfEscalated(state);
      notifyListeners();
    } on ApiException catch (e) {
      // Surface it. A stale composite with no explanation is worse than a stale
      // composite the clinician knows is stale.
      _error = e.message;
      notifyListeners();
    }
  }

  /// Re-runs fusion server-side over the readings already stored, then re-reads.
  ///
  /// Calls no component service — it re-derives the composite from persisted
  /// readings, which is the point of the backend keeping them.
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

  /// Records the clinician's tier judgement against the fusion row currently on
  /// screen.
  ///
  /// Two ordering rules the UI must honour, both from the backend's docstring:
  /// the judgement is entered BEFORE the conformal set is revealed, or the label
  /// is contaminated by the prediction it exists to calibrate; and it attaches
  /// to the id of the row that was displayed, not to whatever is latest now.
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

  /// CARE-AnxRAG clinical decision support.
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

  Future<void> _raiseIfEscalated(FusionResult result) async {
    // A blocked fusion is GREY and has no composite. It is not an escalation,
    // and it must not be silently treated as one in either direction.
    if (!result.hasComposite) return;
    if (result.band != AlertBand.red && result.band != AlertBand.darkRed)
      return;

    await roster.raiseAlert(ClinicalAlert(
      id: _uuid.v4(),
      title: '${result.band.protocolName} · ${patient.name}',
      body: 'Composite risk ${result.compositeLabel} across '
          '${result.modalitiesUsed} of ${Modality.all.length} modalities.',
      raisedAt: DateTime.now(),
      kind: AlertKind.riskEscalation,
      band: result.band,
      patientMrn: mrn,
      patientName: patient.name,
    ));
  }

  // ── TC-WPN analysis ───────────────────────────────────────────────────────

  // ── Note lifecycle ────────────────────────────────────────────────────────
  //
  //   draft ──edit──► draft ──analyse──► analysed ──edit──► analysed (stale)
  //     │                       │                                  │
  //     └──delete               └──failure──► analysisFailed        └──re-analyse
  //
  // Every path writes to local storage BEFORE any network call, and no path
  // removes a note because a network call failed. Persistence is LOCAL ONLY —
  // the Central Backend has no draft CRUD contract, and none was invented.

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

  /// Creates a new draft and returns it. No network call.
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

  /// Edits an EXISTING note in place. The previous build had no way to do this,
  /// so every edit minted a second note and the original stayed behind.
  ///
  /// Editing a note that already carries an assessment does not delete the
  /// assessment — it stamps `updatedAt`, which makes `resultIsStale` true, and
  /// the UI says the assessment describes an earlier version of the text.
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
      // An edit clears a stale failure banner; the new text has not failed yet.
      clearAnalysisError: true,
      status: existing.hasBeenAnalysed
          ? ClinicalNoteStatus.analysed
          : ClinicalNoteStatus.draft,
    );
    _notes = _notes.map((n) => n.id == noteId ? edited : n).toList();
    await _persist();
    return edited;
  }

  /// Deletes a note. Local only, irreversible, and the caller is responsible for
  /// having confirmed with the clinician first.
  ///
  /// Returns false when the note was already gone, so a double-tap does not
  /// report a success that did not happen.
  Future<bool> deleteNote(String noteId) async {
    final before = _notes.length;
    _notes = _notes.where((n) => n.id != noteId).toList();
    if (_notes.length == before) return false;
    await _persist();
    return true;
  }

  /// Analyses a note that is ALREADY STORED. Used for first analysis and for
  /// re-analysis alike — the note is never re-created, so its id, its clinical
  /// date and its clinician attribution survive.
  ///
  /// On success the previous assessment is REPLACED. The local store keeps one
  /// assessment per note because the backend's own record is the history: every
  /// submission is a new `ModalityReading` row and appears on the timeline. A
  /// second local copy would be a second version of the truth.
  ///
  /// On failure the note stays exactly where it was, with its text intact, and
  /// moves to `analysisFailed` so the UI can offer Retry rather than pretending
  /// nothing was attempted. The exception is rethrown for the caller to render.
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

      // ONE call. Server-side this runs the clinical model, stores the reading,
      // and triggers fusion. The app does not call the model service, does not
      // decide whether the reading is usable, and does not fuse anything.
      final ingest = await _backend.submitNote(
        subjectId: subject,
        noteText: note.text,
        noteType: note.noteType,
        // The clinical event date, not the retry date. Re-analysing a note must
        // not make it look newer than the encounter it documents — the backend
        // weights notes by recency.
        noteDate: note.recordedAt,
        supportSet: support,
        visitCount: visitCount,
        author: note.clinicianId,
      );

      _lastIngest = ingest;

      // `result` stays null when the component returned no detail. That renders
      // as "no assessment available", which is the truth — it must never render
      // as a model that looked and found nothing.
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

      // Carry the component's own explanation of a non-ok status through to the
      // UI rather than reporting a generic failure.
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

      // Fusion already ran server-side as part of the ingest; re-read to pick up
      // the full clinician view rather than trusting the ingest's summary.
      await refreshFusion();
      return analysed;
    } on ApiException catch (e) {
      // The note is NOT touched beyond its status. This is the whole point.
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

  /// Compose-and-analyse, for the "write a new note and analyse it now" path.
  ///
  /// Saves first, always. If analysis fails the draft is already on disk and the
  /// exception is rethrown for the caller to render — a clinician's typing is
  /// never lost to an unreachable service.
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

  /// Records the clinician's agreement or disagreement with a prediction. This
  /// is the human-in-the-loop audit trail; it is stored with the note and
  /// exported in the PDF.
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

  /// Promotes an analysed note into the support set after the clinician has
  /// labelled it. This is how few-shot adaptation actually accumulates at a
  /// site: the clinician confirms, and the confirmed case becomes a prototype.
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
