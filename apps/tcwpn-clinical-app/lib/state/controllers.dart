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
  final _tcwpn = TcwpnGateway();

  List<Patient> _patients = [];
  List<ClinicalAlert> _alerts = [];
  List<SupportNote> _siteSupport = [];
  Map<String, FusionResult> _latestFusion = {};
  bool _loading = true;
  Map<String, dynamic>? _modelInfo;

  List<Patient> get patients => List.unmodifiable(_patients);
  List<ClinicalAlert> get alerts => List.unmodifiable(_alerts);
  List<SupportNote> get siteSupport => List.unmodifiable(_siteSupport);
  bool get loading => _loading;
  Map<String, dynamic>? get modelInfo => _modelInfo;

  int get unacknowledgedCount => _alerts.where((a) => !a.acknowledged).length;

  FusionResult? fusionFor(String mrn) => _latestFusion[mrn];

  /// Patients whose most recent composite sits in RED or DARK RED.
  List<Patient> get needingReview => _patients.where((p) {
        final band = _latestFusion[p.mrn]?.band;
        return band == AlertBand.red || band == AlertBand.darkRed;
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
    _tcwpn.health().then((info) {
      _modelInfo = info;
      notifyListeners();
    });
  }

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
      out = out.where((p) => _latestFusion[p.mrn]?.band == band).toList();
    }
    if (ward != null && ward != 'All') {
      out = out.where((p) => p.ward == ward).toList();
    }
    return out;
  }

  @override
  void dispose() {
    _tcwpn.dispose();
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

  final _tcwpn = TcwpnGateway();
  final _fusion = FusionGateway();

  ChartController({
    required this.patient,
    required this.roster,
  }) : mrn = patient.mrn;

  ChartStatus _status = ChartStatus.idle;
  List<ClinicalNote> _notes = [];
  List<SupportNote> _patientSupport = [];
  FusionResult? _fusionResult;
  String? _error;

  ChartStatus get status => _status;
  List<ClinicalNote> get notes => List.unmodifiable(_notes);
  List<SupportNote> get patientSupport => List.unmodifiable(_patientSupport);
  FusionResult? get fusion => _fusionResult;
  String? get error => _error;

  ClinicalNote? get latestAnalysedNote {
    for (final n in _notes) {
      if (n.result != null) return n;
    }
    return null;
  }

  int get visitCount => _notes.length;

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
    _fusionResult = await RecordStore.cachedFusion(mrn);

    _status = ChartStatus.ready;
    notifyListeners();

    // Passive modalities and fusion refresh in the background.
    unawaited(refreshFusion());
  }

  /// Refreshes the fused composite from the fusion service.
  ///
  /// ClinAnx does not collect C1 or C2 and does not contact their services.
  /// Those modalities are gathered by the patient-facing app and pushed to
  /// fusion independently. Reading the fused state is how this app learns
  /// their current values — and their capture timestamps, so the chart can say
  /// how fresh they are.
  Future<void> refreshFusion() async {
    final state = await _fusion.state(mrn);
    if (state == null) return;
    _fusionResult = state;
    await roster.refreshFusion(mrn, state);
    await _raiseIfEscalated(state);
    notifyListeners();
  }

  /// Sends Component 4's assessment into the framework and takes back the
  /// recomputed composite. This is the single point at which TC-WPN output
  /// enters the fusion layer.
  Future<void> contributeAndRefresh(ClinicalNote note) async {
    final r = note.result;
    if (r == null) return;
    final fused = await _fusion.contributeClinicalNlp(
      mrn: mrn,
      result: r,
      noteId: note.id,
      noteDate: note.recordedAt,
    );
    _fusionResult = fused;
    await roster.refreshFusion(mrn, fused);
    await _raiseIfEscalated(fused);
    notifyListeners();
  }

  Future<void> _raiseIfEscalated(FusionResult result) async {
    if (result.band != AlertBand.red && result.band != AlertBand.darkRed) return;
    await roster.raiseAlert(ClinicalAlert(
      id: _uuid.v4(),
      title: '${result.band.protocolName} · ${patient.name}',
      body: 'Composite risk ${result.compositeScore.toStringAsFixed(3)} across '
          '${result.modalitiesAvailable} of 4 modalities.',
      raisedAt: DateTime.now(),
      kind: AlertKind.riskEscalation,
      band: result.band,
      patientMrn: mrn,
      patientName: patient.name,
    ));
  }

  // ── TC-WPN analysis ───────────────────────────────────────────────────────

  /// Saves the note first, then analyses. If analysis fails the note survives as
  /// a draft — a clinician's typing is never lost to a sleeping Space.
  Future<ClinicalNote> analyseNote({
    required String text,
    required String noteType,
    required String clinicianId,
    bool skipAnalysis = false,
  }) async {
    _error = null;
    _status = ChartStatus.working;
    notifyListeners();

    final note = ClinicalNote(
      id: _uuid.v4(),
      patientMrn: mrn,
      recordedAt: DateTime.now(),
      text: text,
      noteType: noteType,
      clinicianId: clinicianId,
    );
    _notes = [note, ..._notes];
    await RecordStore.saveNotes(mrn, _notes);
    notifyListeners();

    if (skipAnalysis) {
      _status = ChartStatus.ready;
      notifyListeners();
      return note;
    }

    try {
      final support = await effectiveSupport();
      final result = await _tcwpn.analyse(
        patientMrn: mrn,
        noteText: text,
        noteType: noteType,
        noteDate: note.recordedAt,
        supportSet: support,
        visitCount: visitCount,
      );
      final analysed = note.copyWith(result: result);
      _notes = _notes.map((n) => n.id == note.id ? analysed : n).toList();
      await RecordStore.saveNotes(mrn, _notes);
      _status = ChartStatus.ready;
      notifyListeners();

      await contributeAndRefresh(analysed);
      return analysed;
    } on ApiException catch (e) {
      _error = e.message;
      _status = ChartStatus.ready;
      notifyListeners();
      rethrow;
    }
  }

  /// Records the clinician's agreement or disagreement with a prediction. This
  /// is the human-in-the-loop audit trail; it is stored with the note and
  /// exported in the PDF.
  Future<void> recordVerdict(String noteId, String verdict, String? comment) async {
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
    _tcwpn.dispose();
    _fusion.dispose();
    super.dispose();
  }
}

void unawaited(Future<void> f) {}