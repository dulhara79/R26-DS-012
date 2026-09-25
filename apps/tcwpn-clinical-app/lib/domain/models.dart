// lib/domain/models.dart
//
// One model layer for the whole framework. Every `fromJson` is defensive: a
// missing or wrongly-typed field degrades to a sensible default rather than
// throwing, because these objects are parsed both from network responses and
// from on-device storage written by older builds.

import '../core/design/tokens.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Safe primitives
// ─────────────────────────────────────────────────────────────────────────────

double _d(dynamic v, [double fallback = 0]) =>
    v is num ? v.toDouble() : double.tryParse('$v') ?? fallback;

int _i(dynamic v, [int fallback = 0]) =>
    v is num ? v.toInt() : int.tryParse('$v') ?? fallback;

String _s(dynamic v, [String fallback = '']) =>
    v == null ? fallback : (v is String ? v : '$v');

bool _b(dynamic v, [bool fallback = false]) => v is bool ? v : fallback;

DateTime _dt(dynamic v) =>
    DateTime.tryParse(_s(v))?.toLocal() ?? DateTime.now();

/// Parses a SERVER timestamp, treating an offset-less string as UTC.
///
/// DateTime.parse treats "2026-08-23T08:39:04" as LOCAL time and
/// "2026-08-23T08:39:04+00:00" as UTC. The backend now emits the aware form
/// everywhere, but two sources still produce the naive form:
///   • caches written by an earlier build of this app;
///   • any deployment running a backend older than the timestamp fix.
///
/// On a device at UTC+5:30 the difference is 5.5 hours, which is enough to make
/// "computed 6 hours ago" read as "computed just now", and enough to invert the
/// ordering between a composite and the readings it was built from. Assuming
/// UTC is right for both sources: the backend works in UTC throughout.
DateTime? _serverDt(dynamic v) {
  final raw = _s(v);
  if (raw.isEmpty) return null;
  final hasOffset =
      raw.endsWith('Z') || RegExp(r'[+-]\d{2}:?\d{2}$').hasMatch(raw);
  final parsed = DateTime.tryParse(hasOffset ? raw : '${raw}Z');
  return parsed?.toLocal();
}

List<Map<String, dynamic>> _objs(dynamic v) => v is List
    ? v.whereType<Map>().map((e) => Map<String, dynamic>.from(e)).toList()
    : const [];

List<String> _strs(dynamic v) =>
    v is List ? v.map((e) => '$e').toList() : const [];

// ─────────────────────────────────────────────────────────────────────────────
// Modality vocabulary
// ─────────────────────────────────────────────────────────────────────────────
//
// THE ONE THING TO READ BEFORE EDITING THIS FILE.
//
// The wire keys below are the CENTRAL BACKEND's, and they do not line up with
// the component numbering in the paper. Both are correct in their own frame;
// they must simply never be confused.
//
//   wire key            backend / fusion service     paper
//   ------------------  ---------------------------  ---------------------------
//   c1_physiological    wearable biosensors          Component 1
//   c2_behavioral       phone sensing (EXCLUDED)     Component 2
//   c3_clinical_nlp     TC-WPN, clinical notes       Component 4
//   c4_demographic      DCAR demographic prior       Component 4's contextual arm
//
// This app previously used `c4_clinical_nlp` and `c3_intervention`, i.e. 3 and 4
// swapped relative to the backend, plus an `intervention` modality the backend
// does not have. A real timeline response therefore produced four contributions
// with weight 0 and score null: the composite rendered, the breakdown was empty.
//
// RULE: wire identifiers follow the backend. Human-readable labels follow the
// paper. Never mix them.
//
// The Personalised Intervention Framework is NOT a fusion modality. It has no
// slot here. When C3_BASE is configured it is called directly and shown as its
// own section of the chart.

class Modality {
  const Modality._();

  static const String c1Physiological = 'c1_physiological';
  static const String c2Behavioral = 'c2_behavioral';
  static const String c3ClinicalNlp = 'c3_clinical_nlp';
  static const String c4Demographic = 'c4_demographic';

  /// Display order, matching main.py's ALL_MODALITIES.
  static const List<String> all = [
    c1Physiological,
    c2Behavioral,
    c3ClinicalNlp,
    c4Demographic,
  ];

  static const Map<String, String> labels = {
    c1Physiological: 'Physiological',
    c2Behavioral: 'Behavioural',
    c3ClinicalNlp: 'Clinical notes — TC-WPN',
    c4Demographic: 'Demographic prior',
  };

  /// The paper's component number, so the UI can speak the language of the
  /// dissertation while the wire speaks the language of the backend.
  static const Map<String, String> paperComponent = {
    c1Physiological: 'Component 1',
    c2Behavioral: 'Component 2',
    c3ClinicalNlp: 'Component 4',
    c4Demographic: 'Component 4 · contextual',
  };

  /// Which app supplies this modality. Shown so a clinician understands that
  /// three of the four signals do not originate in ClinAnx.
  static const Map<String, String> origin = {
    c1Physiological: 'Patient app · wearable',
    c2Behavioral: 'Patient app · phone sensing',
    c3ClinicalNlp: 'ClinAnx · this app',
    c4Demographic: 'Patient app · intake + GAD-7',
  };

  static String labelFor(String key) => labels[key] ?? key;
  static String originFor(String key) => origin[key] ?? 'Unknown';
}

// ─────────────────────────────────────────────────────────────────────────────
// Patient
// ─────────────────────────────────────────────────────────────────────────────

/// The canonical patient. `mrn` is the single key that joins a TC-WPN note
/// analysis, a C3 assessment, a wearable session, and a behavioural graph to the
/// same person across every service.
class Patient {
  final String mrn;
  final String name;
  final int age;
  final String gender;
  final String ward;
  final DateTime referredOn;

  /// C3 requires these for its 13-dimensional feature vector.
  final String maritalStatus;
  final int educationLevel;
  final double incomePir;

  const Patient({
    required this.mrn,
    required this.name,
    required this.age,
    required this.gender,
    this.ward = 'Psychiatry OPD',
    required this.referredOn,
    this.maritalStatus = 'Never',
    this.educationLevel = 3,
    this.incomePir = 2.5,
  });

  /// Splits on any whitespace run — a double space in a name must not crash.
  String get initials {
    final parts =
        name.trim().split(RegExp(r'\s+')).where((p) => p.isNotEmpty).toList();
    if (parts.isEmpty) return '··';
    if (parts.length == 1) {
      final p = parts.first;
      return (p.length > 1 ? p.substring(0, 2) : p).toUpperCase();
    }
    return '${parts.first[0]}${parts.last[0]}'.toUpperCase();
  }

  Patient copyWith({
    String? name,
    int? age,
    String? gender,
    String? ward,
    String? maritalStatus,
    int? educationLevel,
    double? incomePir,
  }) =>
      Patient(
        mrn: mrn,
        name: name ?? this.name,
        age: age ?? this.age,
        gender: gender ?? this.gender,
        ward: ward ?? this.ward,
        referredOn: referredOn,
        maritalStatus: maritalStatus ?? this.maritalStatus,
        educationLevel: educationLevel ?? this.educationLevel,
        incomePir: incomePir ?? this.incomePir,
      );

  Map<String, dynamic> toJson() => {
        'mrn': mrn,
        'name': name,
        'age': age,
        'gender': gender,
        'ward': ward,
        'referred_on': referredOn.toIso8601String(),
        'marital_status': maritalStatus,
        'education_level': educationLevel,
        'income_pir': incomePir,
      };

  factory Patient.fromJson(Map<String, dynamic> j) => Patient(
        mrn: _s(j['mrn'] ?? j['id'], 'UNKNOWN'),
        name: _s(j['name'], 'Unnamed patient'),
        age: _i(j['age'], 24),
        gender: _s(j['gender'], 'Prefer not to say'),
        ward: _s(j['ward'], 'Psychiatry OPD'),
        referredOn: _dt(j['referred_on'] ?? j['referralDate']),
        maritalStatus: _s(j['marital_status'], 'Never'),
        educationLevel: _i(j['education_level'], 3),
        incomePir: _d(j['income_pir'], 2.5),
      );

  /// C3's 13-dimensional feature vector. Encodings kept beside the model they
  /// describe so they can't drift out of sync with the training pipeline.
  Map<String, dynamic> c3Demographics() => {
        'age_norm': ((age - 18) / 17.0).clamp(0.0, 1.0),
        'gender_enc': switch (gender.toLowerCase()) {
          'male' => 1,
          'female' => 2,
          _ => 3,
        },
        'marital_enc': switch (maritalStatus) {
          'Married' => 1,
          'Separated' => 2,
          _ => 3,
        },
        'education_enc': educationLevel,
        'income_enc': (incomePir / 5.0).clamp(0.0, 1.0),
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 4 — TC-WPN
// ─────────────────────────────────────────────────────────────────────────────

/// One span of the submitted note that contributed to the prediction, with the
/// model's own attention weight. `weight` must come from the service — the UI
/// never fabricates prominence from list position.
class AttentionSpan {
  final String text;
  final double weight; // 0..1, normalised attention mass
  final int? start;
  final int? end;

  const AttentionSpan({
    required this.text,
    required this.weight,
    this.start,
    this.end,
  });

  factory AttentionSpan.fromJson(Map<String, dynamic> j) => AttentionSpan(
        text: _s(j['text'] ?? j['phrase']),
        weight: _d(j['weight'] ?? j['attention'], double.nan),
        start: j['start'] == null ? null : _i(j['start']),
        end: j['end'] == null ? null : _i(j['end']),
      );

  /// True when the service returned a real weight. When false the UI renders
  /// the phrase as a plain chip with no intensity ramp and labels the section
  /// "key phrases" rather than "attention-weighted".
  bool get hasWeight => !weight.isNaN;

  Map<String, dynamic> toJson() => {
        'text': text,
        'weight': hasWeight ? weight : null,
        'start': start,
        'end': end
      };
}

/// Per-support-note weighting, so a clinician can see which of their labelled
/// examples actually shaped the prototype. This is the TC-WPN contribution made
/// visible — proposal §4.5.1.
class SupportContribution {
  final String noteId;
  final String label; // 'anxiety' | 'control'
  final String excerpt;
  final double temporalWeight; // exp(-λ · Δt/365), λ = 0.5
  final double confidenceWeight; // 1 / (1 + β·H(x)), β = 1.0
  final double combinedWeight;
  final DateTime? noteDate;

  const SupportContribution({
    required this.noteId,
    required this.label,
    required this.excerpt,
    required this.temporalWeight,
    required this.confidenceWeight,
    required this.combinedWeight,
    this.noteDate,
  });

  factory SupportContribution.fromJson(Map<String, dynamic> j) =>
      SupportContribution(
        noteId: _s(j['note_id'] ?? j['id']),
        label: _s(j['label'], 'anxiety'),
        excerpt: _s(j['excerpt'] ?? j['text']),
        temporalWeight: _d(j['temporal_weight'], 1),
        confidenceWeight: _d(j['confidence_weight'], 1),
        combinedWeight: _d(j['combined_weight'] ?? j['weight'], 1),
        noteDate: j['note_date'] == null ? null : _dt(j['note_date']),
      );

  /// Round-trips through [fromJson]. Without this the support-influence panel
  /// was present the first time an assessment was shown and gone the moment the
  /// note was reopened from history, because `TcwpnResult.toJson` had nothing to
  /// write these into.
  Map<String, dynamic> toJson() => {
        'note_id': noteId,
        'label': label,
        'excerpt': excerpt,
        'temporal_weight': temporalWeight,
        'confidence_weight': confidenceWeight,
        'combined_weight': combinedWeight,
        'note_date': noteDate?.toIso8601String(),
      };
}

/// TC-WPN inference output.
class TcwpnResult {
  final String prediction; // 'ANXIETY' | 'NO ANXIETY'
  final double riskScore; // raw prototype-distance score, 0..1
  final double calibratedProbability; // post-calibration P(anxiety)
  final double confidence;
  final double entropy; // Shannon entropy of class probabilities
  final double threshold; // locked decision threshold from validation
  final int supportK; // shots actually used
  final double? ece; // expected calibration error, if reported
  final List<AttentionSpan> spans;
  final List<SupportContribution> supportContributions;
  final double? prototypeDistanceAnxiety;
  final double? prototypeDistanceControl;
  final String temporalContext; // e.g. 'Visit 3 of 5'
  final String modelVersion;
  final int latencyMs;
  final bool usedDefaultSupportSet;

  /// The service's own `calibration_status`, verbatim, or '' when it did not
  /// send one. The live TC-WPN deployment reports `uncalibrated` alongside a
  /// field it names `probability` (see the Central Backend's
  /// modality_clients.py::call_c3). Calling that number a calibrated
  /// probability in the UI would assert a property the model disclaims.
  final String calibrationStatus;

  const TcwpnResult({
    required this.prediction,
    required this.riskScore,
    required this.calibratedProbability,
    required this.confidence,
    required this.entropy,
    required this.threshold,
    required this.supportK,
    this.ece,
    this.spans = const [],
    this.supportContributions = const [],
    this.prototypeDistanceAnxiety,
    this.prototypeDistanceControl,
    this.temporalContext = '',
    this.modelVersion = 'TC-WPN',
    this.latencyMs = 0,
    this.usedDefaultSupportSet = false,
    this.calibrationStatus = '',
  });

  /// True only when the service positively said the probability is calibrated.
  /// Silence is not a claim of calibration.
  bool get isCalibrated => calibrationStatus.toLowerCase() == 'calibrated';

  /// What to call the headline number. Follows [calibrationStatus] rather than
  /// the field name it arrived under.
  String get probabilityLabel => calibrationStatus.isEmpty
      ? 'CLINICAL-NOTE RISK'
      : (isCalibrated
          ? 'CLINICAL-NOTE RISK · CALIBRATED'
          : 'CLINICAL-NOTE RISK · UNCALIBRATED');

  bool get isPositive => riskScore >= threshold;

  /// Low-confidence predictions must be flagged, not quietly rendered. The
  /// proposal's own safety notice sets the bar at 60%.
  bool get needsManualReview => confidence < 0.60;

  /// True when the service reported real, usable attention mass for at least
  /// one span. `isFinite` as well as [AttentionSpan.hasWeight]: an infinite
  /// weight cannot be drawn as a bar, so claiming attribution for it would put
  /// the section heading and the section's contents out of step.
  bool get hasAttribution => spans.any((s) => s.hasWeight && s.weight.isFinite);

  factory TcwpnResult.fromJson(Map<String, dynamic> j, {int? fallbackLatency}) {
    final raw = _d(j['risk_score']);
    return TcwpnResult(
      prediction: _s(j['prediction'],
          raw >= _d(j['threshold'], .5) ? 'ANXIETY' : 'NO ANXIETY'),
      riskScore: raw,
      // Same preference order the Central Backend applies in call_c3:
      // calibrated_probability > probability > risk_score > score. The live
      // Space names the field `probability`; without it in this chain the
      // headline read 0.0000 for a response that did contain a score.
      calibratedProbability: _d(j['calibrated_probability'] ??
          j['probability'] ??
          j['risk_score'] ??
          j['score']),
      confidence: _d(j['confidence']),
      entropy: _d(j['entropy'], double.nan),
      threshold: _d(j['threshold'], 0.5),
      supportK: _i(j['support_k']),
      ece: j['ece'] == null ? null : _d(j['ece']),
      spans: (j['attention_spans'] is List
              ? _objs(j['attention_spans']).map(AttentionSpan.fromJson).toList()
              : _strs(j['key_phrases'])
                  .map((p) => AttentionSpan(text: p, weight: double.nan))
                  .toList())
          .where((s) => s.text.trim().isNotEmpty)
          .toList(),
      supportContributions: _objs(j['support_contributions'])
          .map(SupportContribution.fromJson)
          .toList(),
      prototypeDistanceAnxiety: j['prototype_distance_anxiety'] == null
          ? null
          : _d(j['prototype_distance_anxiety']),
      prototypeDistanceControl: j['prototype_distance_control'] == null
          ? null
          : _d(j['prototype_distance_control']),
      temporalContext: _s(j['temporal_context']),
      modelVersion: _s(j['model_version'], 'TC-WPN'),
      latencyMs: _i(j['latency_ms'], fallbackLatency ?? 0),
      usedDefaultSupportSet: _b(j['used_default_support_set']),
      calibrationStatus: _s(j['calibration_status']),
    );
  }

  Map<String, dynamic> toJson() => {
        'prediction': prediction,
        'risk_score': riskScore,
        'calibrated_probability': calibratedProbability,
        'confidence': confidence,
        'entropy': entropy.isNaN ? null : entropy,
        'threshold': threshold,
        'support_k': supportK,
        'ece': ece,
        'attention_spans': spans.map((s) => s.toJson()).toList(),
        // These three were dropped by the previous version of this method, so
        // the local cache round-tripped an assessment that had LESS in it than
        // the one the clinician was shown a moment earlier: the support-set
        // influence panel and the prototype-distance rows simply vanished when
        // the note was reopened from the chart. They are written now.
        'support_contributions':
            supportContributions.map((s) => s.toJson()).toList(),
        'prototype_distance_anxiety': prototypeDistanceAnxiety,
        'prototype_distance_control': prototypeDistanceControl,
        'temporal_context': temporalContext,
        'model_version': modelVersion,
        'latency_ms': latencyMs,
        'used_default_support_set': usedDefaultSupportSet,
      };
}

/// The Central Backend's response to POST /v1/clinical-notes.
///
/// One request does four things server-side: calls TC-WPN, stores the reading,
/// auto-triggers fusion, and returns all three outcomes. This class keeps them
/// distinguishable, because they fail independently — a note can be stored and
/// scored while fusion is blocked, and that is a normal state, not an error.
class ClinicalNoteIngestResult {
  final String subjectId;
  final int? readingId;

  /// The component's status, verbatim: `ok` · `no_support_set` · `error` …
  final String status;

  /// The raw modality score the backend stored. Null unless status is `ok`.
  final double? score;

  /// The backend's own note on this reading — which score field it used and how
  /// confidence was derived. Surface it rather than asserting calibration:
  /// call_c3 prefers `calibrated_probability`, falls back to `risk_score`, and
  /// records which one it actually got.
  final String? scoreProvenance;

  /// The TC-WPN response in full, parsed. Null when the component did not
  /// return one — in which case the explanation panels must say so rather than
  /// render empty state as if the model had found nothing.
  final TcwpnResult? result;

  final bool fusionTriggered;
  final int? fusionResultId;

  /// Set when fusion ran but failed. The note is still stored — the backend
  /// never lets a fusion failure fail the ingest.
  final String? fusionError;

  /// Set when fusion was deliberately skipped, e.g. physiological debounce.
  final String? fusionSkippedReason;

  const ClinicalNoteIngestResult({
    required this.subjectId,
    this.readingId,
    this.status = 'error',
    this.score,
    this.scoreProvenance,
    this.result,
    this.fusionTriggered = false,
    this.fusionResultId,
    this.fusionError,
    this.fusionSkippedReason,
  });

  bool get scored => status == 'ok' && score != null;

  /// True when the model ran but had no prototypes to compare against. Worth
  /// distinguishing in the UI: the fix is clinician action (label some support
  /// notes), not retrying.
  bool get needsSupportSet => status == 'no_support_set';

  factory ClinicalNoteIngestResult.fromJson(
    Map<String, dynamic> j, {
    int? fallbackLatency,
  }) {
    final detail = j['component_detail'];
    final fusion = j['fusion'] is Map
        ? Map<String, dynamic>.from(j['fusion'] as Map)
        : const <String, dynamic>{};

    return ClinicalNoteIngestResult(
      subjectId: _s(j['subject_id']),
      readingId: j['reading_id'] == null ? null : _i(j['reading_id']),
      status: _s(j['status'], 'error'),
      score: j['score'] == null ? null : _d(j['score']),
      scoreProvenance: j['score_provenance'] == null
          ? (j['note'] == null ? null : _s(j['note']))
          : _s(j['score_provenance']),
      result: detail is Map && detail.isNotEmpty
          ? TcwpnResult.fromJson(Map<String, dynamic>.from(detail),
              fallbackLatency: fallbackLatency)
          : null,
      fusionTriggered: _b(j['fusion_triggered']),
      fusionResultId: j['fusion_result_id'] == null
          ? (fusion['fusion_result_id'] == null
              ? null
              : _i(fusion['fusion_result_id']))
          : _i(j['fusion_result_id']),
      fusionError: j['fusion_error'] == null ? null : _s(j['fusion_error']),
      fusionSkippedReason: j['fusion_skipped_reason'] == null
          ? null
          : _s(j['fusion_skipped_reason']),
    );
  }
}

/// The Central Backend's response to POST /v1/subjects — enrolment.
class EnrolmentResult {
  final String subjectId;

  /// Read aloud to the patient, who types it into the patient app. Until it is
  /// redeemed, that patient's wearable and intake readings never join this
  /// subject, and the gate will block fusion for want of a second modality.
  final String pairingCode;
  final DateTime? expiresAt;

  const EnrolmentResult({
    required this.subjectId,
    required this.pairingCode,
    this.expiresAt,
  });

  factory EnrolmentResult.fromJson(Map<String, dynamic> j) => EnrolmentResult(
        subjectId: _s(j['subject_id']),
        pairingCode: _s(j['pairing_code']),
        expiresAt: _serverDt(j['expires_at']),
      );
}

/// A clinician-labelled note in the per-patient support set. K of these form the
/// prototypes; the app keeps `visitCount` and `noteDate` because TC-WPN's
/// temporal weighting needs both.
class SupportNote {
  final String id;
  final String text;
  final String label; // 'anxiety' | 'control'
  final DateTime noteDate;
  final DateTime addedAt;
  final String? patientMrn; // null = global/site-level support note
  final String addedByClinician;

  const SupportNote({
    required this.id,
    required this.text,
    required this.label,
    required this.noteDate,
    required this.addedAt,
    this.patientMrn,
    this.addedByClinician = '',
  });

  bool get isAnxiety => label == 'anxiety';

  Map<String, dynamic> toJson() => {
        'id': id,
        'text': text,
        'label': label,
        'note_date': noteDate.toIso8601String(),
        'added_at': addedAt.toIso8601String(),
        'patient_mrn': patientMrn,
        'added_by': addedByClinician,
      };

  factory SupportNote.fromJson(Map<String, dynamic> j) => SupportNote(
        id: _s(j['id'], DateTime.now().microsecondsSinceEpoch.toString()),
        text: _s(j['text']),
        label: _s(j['label'], 'anxiety'),
        noteDate: _dt(j['note_date']),
        addedAt: _dt(j['added_at']),
        patientMrn: j['patient_mrn'] == null ? null : _s(j['patient_mrn']),
        addedByClinician: _s(j['added_by']),
      );

  /// Wire shape the TC-WPN service expects for prototype formation.
  Map<String, dynamic> toWire() => {
        'id': id,
        'text': text,
        'label': label,
        'note_date': noteDate.toIso8601String(),
      };
}

/// A clinical note plus its TC-WPN analysis, as filed in the patient chart.
/// Where a note sits in its lifecycle.
///
/// This is stored EXPLICITLY rather than inferred from `result == null`. The
/// previous build inferred it, which made two very different situations
/// indistinguishable: a note the clinician deliberately parked as a draft, and
/// a note that was submitted, reached the backend, and came back with no
/// component detail. The first must offer "Analyse"; the second must say the
/// service returned no assessment. Collapsing them is how a submitted note ends
/// up looking unwritten.
enum ClinicalNoteStatus {
  /// Written and saved locally. Never sent, or sent and not yet answered.
  draft,

  /// Sent, and the backend returned a usable assessment.
  analysed,

  /// Sent, and the attempt failed. The TEXT IS STILL SAFE — this state exists so
  /// the UI can say "saved, not analysed" instead of losing the distinction.
  analysisFailed,
}

class ClinicalNote {
  final String id;
  final String patientMrn;

  /// When the clinician first wrote the note. Stable across edits — this is the
  /// clinical event date, and it is what goes to the backend as `note_date`.
  final DateTime recordedAt;

  /// When the note was last edited on this device. Null if never edited.
  final DateTime? updatedAt;

  final String text;
  final String noteType;
  final String clinicianId;
  final TcwpnResult? result;
  final String? clinicianComment;

  /// Explicit lifecycle state. See [ClinicalNoteStatus].
  final ClinicalNoteStatus status;

  /// The clinician-facing reason the last analysis attempt failed. Cleared on a
  /// successful analysis. Never contains a URL, a status code, or a service name.
  final String? lastAnalysisError;

  /// When the current [result] was produced. Null when there is no result.
  final DateTime? analysedAt;

  /// Set when a clinician explicitly agrees or disagrees with the model.
  /// This is the HITL audit trail the proposal requires (§4.4 / §6).
  final String? clinicianVerdict; // 'agree' | 'disagree' | null

  const ClinicalNote({
    required this.id,
    required this.patientMrn,
    required this.recordedAt,
    required this.text,
    required this.noteType,
    required this.clinicianId,
    this.updatedAt,
    this.result,
    this.clinicianComment,
    this.clinicianVerdict,
    this.status = ClinicalNoteStatus.draft,
    this.lastAnalysisError,
    this.analysedAt,
  });

  bool get isDraft => status == ClinicalNoteStatus.draft;
  bool get isAnalysed =>
      status == ClinicalNoteStatus.analysed && result != null;
  bool get analysisFailed => status == ClinicalNoteStatus.analysisFailed;

  /// True when this note has been analysed before, so the action is
  /// "Re-analyse" rather than "Analyse".
  bool get hasBeenAnalysed => result != null;

  /// True when the text has changed since the assessment on screen was produced,
  /// which makes that assessment stale relative to what the clinician is reading.
  bool get resultIsStale =>
      result != null &&
      analysedAt != null &&
      updatedAt != null &&
      updatedAt!.isAfter(analysedAt!);

  /// Editing a draft must UPDATE it, never mint a second one. `clearResult`
  /// exists because `result ?? this.result` can only ever set a result, so the
  /// old copyWith had no way to express "this note is being rewritten and the
  /// previous assessment no longer describes it".
  ClinicalNote copyWith({
    String? text,
    String? noteType,
    DateTime? updatedAt,
    TcwpnResult? result,
    bool clearResult = false,
    String? clinicianComment,
    String? clinicianVerdict,
    ClinicalNoteStatus? status,
    String? lastAnalysisError,
    bool clearAnalysisError = false,
    DateTime? analysedAt,
  }) =>
      ClinicalNote(
        id: id,
        patientMrn: patientMrn,
        recordedAt: recordedAt,
        updatedAt: updatedAt ?? this.updatedAt,
        text: text ?? this.text,
        noteType: noteType ?? this.noteType,
        clinicianId: clinicianId,
        result: clearResult ? null : (result ?? this.result),
        clinicianComment: clinicianComment ?? this.clinicianComment,
        clinicianVerdict: clinicianVerdict ?? this.clinicianVerdict,
        status: status ?? this.status,
        lastAnalysisError: clearAnalysisError
            ? null
            : (lastAnalysisError ?? this.lastAnalysisError),
        analysedAt: clearResult ? null : (analysedAt ?? this.analysedAt),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'patient_mrn': patientMrn,
        'recorded_at': recordedAt.toIso8601String(),
        'updated_at': updatedAt?.toIso8601String(),
        'text': text,
        'note_type': noteType,
        'clinician_id': clinicianId,
        'result': result?.toJson(),
        'clinician_comment': clinicianComment,
        'clinician_verdict': clinicianVerdict,
        'status': status.name,
        'last_analysis_error': lastAnalysisError,
        'analysed_at': analysedAt?.toIso8601String(),
      };

  factory ClinicalNote.fromJson(Map<String, dynamic> j) {
    final result = j['result'] is Map
        ? TcwpnResult.fromJson(Map<String, dynamic>.from(j['result']))
        : null;

    // Notes written by the previous build carry no `status`. Reconstruct it the
    // way that build meant it — a result present meant analysed — so an upgrade
    // does not silently reopen every historical note as a draft.
    final status = switch (_s(j['status'])) {
      'analysed' => ClinicalNoteStatus.analysed,
      'analysisFailed' => ClinicalNoteStatus.analysisFailed,
      'draft' => ClinicalNoteStatus.draft,
      _ =>
        result != null ? ClinicalNoteStatus.analysed : ClinicalNoteStatus.draft,
    };

    return ClinicalNote(
      id: _s(j['id'], DateTime.now().microsecondsSinceEpoch.toString()),
      patientMrn: _s(j['patient_mrn'] ?? j['patientId']),
      recordedAt: _dt(j['recorded_at'] ?? j['timestamp']),
      updatedAt: j['updated_at'] == null ? null : _dt(j['updated_at']),
      text: _s(j['text'] ?? j['noteText']),
      noteType: _s(j['note_type'] ?? j['noteType'], 'Psychiatry note'),
      clinicianId: _s(j['clinician_id'] ?? j['clinicianId']),
      result: result,
      clinicianComment:
          j['clinician_comment'] == null ? null : _s(j['clinician_comment']),
      clinicianVerdict:
          j['clinician_verdict'] == null ? null : _s(j['clinician_verdict']),
      status: status,
      lastAnalysisError: j['last_analysis_error'] == null
          ? null
          : _s(j['last_analysis_error']),
      analysedAt: j['analysed_at'] == null ? null : _dt(j['analysed_at']),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 3 — intervention engine
// ─────────────────────────────────────────────────────────────────────────────

class ShapFeature {
  final String feature;
  final double value;
  final double contribution; // signed SHAP value

  const ShapFeature({
    required this.feature,
    required this.value,
    required this.contribution,
  });

  factory ShapFeature.fromJson(Map<String, dynamic> j) => ShapFeature(
        feature: _s(j['feature']),
        value: _d(j['value']),
        contribution: _d(j['contribution'] ?? j['shap_value']),
      );

  /// Feature names arrive as training-pipeline identifiers. Clinicians read
  /// English, so translate at the boundary.
  String get readable => switch (feature) {
        'age_norm' => 'Age',
        'gender_enc' => 'Gender',
        'marital_enc' => 'Marital status',
        'education_enc' => 'Education level',
        'income_enc' => 'Income (PIR)',
        'physiological_risk' => 'Physiological risk (C1)',
        'behavioral_risk' => 'Behavioural risk (C2)',
        'textual_risk' => 'Clinical-note risk (C4)',
        'composite_risk' => 'Composite risk',
        'risk_tier_enc' => 'Previous risk tier',
        'interaction_count_norm' => 'Sessions completed',
        'last_reward_norm' => 'Response to last intervention',
        'escalation_count_norm' => 'Prior escalations',
        _ => feature.replaceAll('_', ' '),
      };
}

class Counterfactual {
  final String feature;
  final String from;
  final String to;
  final String resultingTier;

  const Counterfactual({
    required this.feature,
    required this.from,
    required this.to,
    required this.resultingTier,
  });

  factory Counterfactual.fromJson(Map<String, dynamic> j) => Counterfactual(
        feature: _s(j['feature']),
        from: _s(j['from']),
        to: _s(j['to']),
        resultingTier: _s(j['resulting_tier'], 'Low'),
      );
}

class SimilarCase {
  final String caseId;
  final double similarity;
  final String interventionUsed;
  final double outcomeDelta; // GAD-7 change achieved

  const SimilarCase({
    required this.caseId,
    required this.similarity,
    required this.interventionUsed,
    required this.outcomeDelta,
  });

  factory SimilarCase.fromJson(Map<String, dynamic> j) => SimilarCase(
        caseId: _s(j['case_id']),
        similarity: _d(j['similarity']),
        interventionUsed: _s(j['intervention']),
        outcomeDelta: _d(j['outcome_delta']),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Components 1 & 2 — passive modalities
// ─────────────────────────────────────────────────────────────────────────────

/// One modality's latest reading, exactly as the Central Backend's clinician
/// timeline reports it (`modalities[<key>]` in
/// GET /v1/doctor/patients/{subject_id}/timeline).
///
/// The backend's governing rule, stated at the top of modality_clients.py, is
/// that A COMPONENT THAT DOES NOT ANSWER IS MISSING, NOT ZERO. That rule holds
/// server-side; this class is where it either survives or is quietly undone.
/// `status` is therefore kept verbatim rather than collapsed into a boolean.
class ModalityReading {
  final String key;

  /// Backend status vocabulary, passed through unmodified:
  /// `ok` · `absent` · `warming_up` · `insufficient_data` · `poor_signal` ·
  /// `no_support_set` · `not_validated` · `error`.
  final String status;

  final double? score;
  final double? confidence;
  final double? coverage;
  final DateTime? capturedAt;

  /// Age as the SERVER computed it. Preferred over anything derived from
  /// `capturedAt` on this device, because device clocks drift.
  final double? ageMinutes;

  /// The server's freshness verdict, from gate.MAX_AGE_MINUTES — which differs
  /// per modality (15 min physiological, 7 days behavioural, 90 days notes,
  /// never for the demographic prior). Do not re-derive it from a single
  /// client-side threshold.
  final bool fresh;

  final String? modelVersion;

  /// Excluded from the composite by pre-registered rule rather than by absence.
  /// True for c2_behavioral: AUROC 0.5205 against a permutation null of 0.4991,
  /// p = 0.255. The reading is still shown, so the exclusion stays auditable.
  final bool excluded;

  const ModalityReading({
    required this.key,
    this.status = 'absent',
    this.score,
    this.confidence,
    this.coverage,
    this.capturedAt,
    this.ageMinutes,
    this.fresh = false,
    this.modelVersion,
    this.excluded = false,
  });

  /// Usable evidence. Anything other than exactly `ok` with a real score is not,
  /// which is the same test gate.py applies server-side.
  bool get available => status == 'ok' && score != null;

  bool get isStale => available && !fresh;

  String get label => Modality.labelFor(key);
  String get origin => Modality.originFor(key);

  /// Plain-language reason this modality is not contributing. Written for a
  /// clinician: what is true of the data, not what the service returned.
  String? get unavailableReason {
    if (available) return null;
    return switch (status) {
      'absent' => 'No reading recorded for this patient yet.',
      'warming_up' =>
        'The wearable is still learning this patient\u2019s personal baseline.',
      'insufficient_data' => 'Not enough history yet to produce a score.',
      'poor_signal' => 'Signal quality was below the usable threshold.',
      'no_support_set' =>
        'No labelled support notes for this patient, so no prototype could be '
            'formed. Add support notes to enable few-shot analysis.',
      'not_validated' =>
        'Recorded but excluded from the composite by pre-registered rule \u2014 '
            'this model did not clear its permutation null.',
      'error' => 'The service did not respond, or returned an unusable result.',
      _ => 'Not contributing (status: $status).',
    };
  }

  /// Human-readable age, preferring the server's own figure.
  String? get ageLabel {
    final mins = ageMinutes ??
        (capturedAt == null
            ? null
            : DateTime.now().difference(capturedAt!).inMinutes.toDouble());
    if (mins == null) return null;
    if (mins < 60) return '${mins.round()}m ago';
    if (mins < 24 * 60) return '${(mins / 60).round()}h ago';
    if (mins < 30 * 24 * 60) return '${(mins / 1440).round()}d ago';
    return '${(mins / 43200).round()}mo ago';
  }

  factory ModalityReading.fromJson(String key, Map<String, dynamic> j) =>
      ModalityReading(
        key: key,
        status: _s(j['status'], 'absent'),
        score: j['score'] == null ? null : _d(j['score']),
        confidence: j['confidence'] == null ? null : _d(j['confidence']),
        coverage: j['coverage'] == null ? null : _d(j['coverage']),
        capturedAt: _serverDt(j['captured_at']),
        ageMinutes: j['age_minutes'] == null ? null : _d(j['age_minutes']),
        fresh: _b(j['fresh']),
        modelVersion:
            j['model_version'] == null ? null : _s(j['model_version']),
        excluded: _b(j['excluded']),
      );

  Map<String, dynamic> toJson() => {
        'status': status,
        'score': score,
        'confidence': confidence,
        'coverage': coverage,
        'captured_at': capturedAt?.toIso8601String(),
        'age_minutes': ageMinutes,
        'fresh': fresh,
        'model_version': modelVersion,
        'excluded': excluded,
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// Late fusion — as computed by the Central Backend
// ─────────────────────────────────────────────────────────────────────────────

/// One modality's line in the fusion breakdown: the weight the SERVER applied,
/// the score it applied it to, and their product.
///
/// Every number here comes off the wire. Nothing on this class is computed from
/// a locally held weight table, because this app no longer has one.
class ComponentContribution {
  final String key;
  final double weight;
  final double? contribution;
  final ModalityReading reading;

  /// This reading was captured AFTER the fusion row being displayed was
  /// computed, so it is not in this composite even though it is present, `ok`
  /// and fresh.
  ///
  /// This is not an edge case — it is the normal consequence of the backend's
  /// auto-fusion policy. Note and intake ingests fuse immediately, but
  /// physiological ingests are DEBOUNCED (AUTO_FUSION_DEBOUNCE_MIN, default 5
  /// minutes) because the stream arrives every 60 seconds and fusing on every
  /// tick would write ~1,440 rows per patient per day. So a wearable reading
  /// can sit in `modalities` looking perfectly usable while `weights` gives it
  /// zero, purely because the composite predates it.
  ///
  /// Rendering that as "weight 0.00" would tell a clinician the wearable was
  /// considered and discounted. It was not considered at all yet. Re-running
  /// fusion includes it.
  final bool capturedAfterFusion;

  const ComponentContribution({
    required this.key,
    required this.weight,
    required this.contribution,
    required this.reading,
    this.capturedAfterFusion = false,
  });

  /// Present and usable, but not in the composite on screen.
  bool get pendingNextFusion => capturedAfterFusion && reading.available;

  String get label => Modality.labelFor(key);
  String get origin => Modality.originFor(key);
  double? get score => reading.score;
  bool get available => reading.available;
  bool get isStale => reading.isStale;
  bool get excluded => reading.excluded;
  String? get ageLabel => reading.ageLabel;

  String? get note {
    if (pendingNextFusion) {
      return 'Recorded after this assessment was computed \u2014 it will be '
          'included the next time fusion runs.';
    }
    return reading.unavailableReason;
  }
}

/// One point on the composite's history, from the timeline's `trend` array.
class TrendPoint {
  final double? composite;
  final String? tier;
  final AlertBand band;
  final DateTime? computedAt;

  /// What caused this fusion to run: `note-ingest`, `physio-ingest`,
  /// `contextual-ingest`, or `manual`. Useful for explaining why the composite
  /// moved when no new note was written.
  final String? trigger;

  const TrendPoint({
    this.composite,
    this.tier,
    required this.band,
    this.computedAt,
    this.trigger,
  });

  factory TrendPoint.fromJson(Map<String, dynamic> j) => TrendPoint(
        composite: j['composite'] == null ? null : _d(j['composite']),
        tier: j['tier'] == null ? null : _s(j['tier']),
        band: AlertBandX.fromWire(_s(j['band'], 'GREY')),
        computedAt: _serverDt(j['computed_at']),
        trigger: j['trigger'] == null ? null : _s(j['trigger']),
      );

  Map<String, dynamic> toJson() => {
        'composite': composite,
        'tier': tier,
        'band': band.protocolName,
        'computed_at': computedAt?.toIso8601String(),
        'trigger': trigger,
      };
}

/// The authoritative clinical result, as returned by
/// GET /v1/doctor/patients/{subject_id}/timeline.
///
/// TWO DELIBERATE DESIGN POINTS, both of them safety properties:
///
///  1. `compositeScore` IS NULLABLE. The backend returns `composite: null` with
///     `band: "GREY"` whenever the fusion gate blocks. The previous parser ran
///     that through a helper that defaulted to 0, so an assessment the server
///     refused to make displayed as 0.000 in the GREEN band. A null composite
///     is missing evidence and must render as such.
///
///  2. THERE IS NO LOCAL FALLBACK. `FusionResult.local(...)` and
///     `Env.defaultWeights` are gone. When the backend cannot be reached the app
///     shows the last cached server result with its age, or nothing — it does
///     not invent a composite from a different weight vector and label it
///     provisional.
class FusionResult {
  final String patientMrn;
  final String? subjectId;

  /// The id of the fusion row these numbers came from. A clinician verdict must
  /// be attached to THIS id, not to whatever is latest at submit time —
  /// otherwise the label and the score it calibrates drift apart.
  final int? fusionResultId;

  final double? compositeScore;

  /// The server's own tier: `Low` | `Medium` | `High`, or null when blocked.
  final String? tier;

  /// The server's own band. Never re-derived from `compositeScore` here: the
  /// fusion service bands at 0.33/0.66 into three tiers, which is not the same
  /// split as AlertBandX.fromScore.
  final AlertBand band;

  final double? confidence;

  /// The server's explanation, shown verbatim. For a blocked fusion this is the
  /// gate's own reason, e.g. "insufficient evidence: 1 usable modality, need 2".
  final String? reason;

  final List<ComponentContribution> contributions;
  final Map<String, ModalityReading> modalities;

  /// Weights were rescaled across the modalities that actually reported.
  final bool renormalised;
  final int modalitiesUsed;
  final String assessmentStatus;
  final List<String> missingModalities;

  /// The raw gate decision (`passed`, `usable_modalities`, `rejected`, `reason`)
  /// and conformal prediction set, passed through for the detail screen.
  final Map<String, dynamic>? gate;
  final Map<String, dynamic>? conformal;

  final DateTime? updatedAt;
  final List<TrendPoint> trend;

  const FusionResult({
    required this.patientMrn,
    this.subjectId,
    this.fusionResultId,
    this.compositeScore,
    this.tier,
    required this.band,
    this.confidence,
    this.reason,
    this.contributions = const [],
    this.modalities = const {},
    this.renormalised = false,
    this.modalitiesUsed = 0,
    this.assessmentStatus = 'insufficient',
    this.missingModalities = const [],
    this.gate,
    this.conformal,
    this.updatedAt,
    this.trend = const [],
  });

  bool get hasComposite => compositeScore != null;

  String get assessmentLabel => assessmentStatus == 'complete'
      ? 'Complete assessment'
      : assessmentStatus == 'provisional'
      ? 'Provisional assessment'
      : 'Assessment incomplete';

  /// The gate refused to fuse. Distinct from "no data at all".
  bool get blocked => compositeScore == null;

  /// Readings that arrived after this composite was computed and are therefore
  /// not in it. Non-empty means "re-run fusion to include them", not "these
  /// modalities failed".
  List<ComponentContribution> get pendingReadings =>
      contributions.where((c) => c.pendingNextFusion).toList();

  /// Modalities the gate rejected, with the server's reason for each.
  Map<String, String> get rejected {
    final r = gate?['rejected'];
    if (r is! Map) return const {};
    return {for (final e in r.entries) '${e.key}': '${e.value}'};
  }

  List<String> get usableModalities {
    final u = gate?['usable_modalities'];
    return u is List ? u.map((e) => '$e').toList() : const [];
  }

  /// Composite formatted for display, or an em dash. Never "0.000" for a
  /// composite the server declined to compute.
  String get compositeLabel =>
      compositeScore == null ? '\u2014' : compositeScore!.toStringAsFixed(3);

  /// Parses the clinician timeline response.
  ///
  /// Key names differ from the old fusion-service shape throughout, which is
  /// why every one of them is spelled out here rather than guessed:
  ///   composite (not composite_score) · band (not alert_level) ·
  ///   modalities (not scores) · updated_at (not computed_at).
  factory FusionResult.fromJson(Map<String, dynamic> j, String mrn) {
    final weights = <String, double>{};
    if (j['weights'] is Map) {
      (j['weights'] as Map).forEach((k, v) => weights['$k'] = _d(v));
    }
    final contributionValues = <String, double?>{};
    if (j['contributions'] is Map) {
      (j['contributions'] as Map).forEach(
          (k, v) => contributionValues['$k'] = v == null ? null : _d(v));
    }

    final modalities = <String, ModalityReading>{};
    if (j['modalities'] is Map) {
      (j['modalities'] as Map).forEach((k, v) {
        if (v is Map) {
          modalities['$k'] =
              ModalityReading.fromJson('$k', Map<String, dynamic>.from(v));
        }
      });
    }

    final updatedAt = _serverDt(j['updated_at']);
    final modalitiesUsed = _i(j['modalities_used']);
    final assessmentStatus = j['assessment_status'] == null
        ? modalitiesUsed >= 3
            ? 'complete'
            : modalitiesUsed >= 2
            ? 'provisional'
            : 'insufficient'
        : _s(j['assessment_status']);

    final contribs = <ComponentContribution>[];
    for (final key in Modality.all) {
      final reading = modalities[key] ?? ModalityReading(key: key);
      // A tolerance, not an exact comparison: the fusion row's timestamp and the
      // reading's are written by different code paths within the same request,
      // so sub-second ordering is noise rather than signal.
      final capturedAfter = reading.capturedAt != null &&
          updatedAt != null &&
          reading.capturedAt!.difference(updatedAt).inSeconds > 1;
      contribs.add(ComponentContribution(
        key: key,
        weight: weights[key] ?? 0,
        capturedAfterFusion: capturedAfter,
        // Prefer the server's own contribution figure. Falling back to
        // weight × score is only for a server that has not sent one; it is not
        // a substitute for the server's arithmetic.
        contribution: contributionValues[key] ??
            (reading.score == null
                ? null
                : reading.score! * (weights[key] ?? 0)),
        reading: reading,
      ));
    }

    return FusionResult(
      patientMrn: mrn,
      subjectId: j['subject_id'] == null ? null : _s(j['subject_id']),
      fusionResultId:
          j['fusion_result_id'] == null ? null : _i(j['fusion_result_id']),
      // No default. A missing composite stays missing.
      compositeScore: j['composite'] == null ? null : _d(j['composite']),
      tier: j['tier'] == null ? null : _s(j['tier']),
      band: AlertBandX.fromWire(_s(j['band'], 'GREY')),
      confidence: j['confidence'] == null ? null : _d(j['confidence']),
      reason: j['reason'] == null ? null : _s(j['reason']),
      contributions: contribs,
      modalities: modalities,
      renormalised: _b(j['renormalised']),
      modalitiesUsed: modalitiesUsed,
      assessmentStatus: assessmentStatus,
      missingModalities: j['missing_modalities'] is List
          ? (j['missing_modalities'] as List).map((value) => '$value').toList()
          : const <String>[],
      gate: j['gate'] is Map ? Map<String, dynamic>.from(j['gate']) : null,
      conformal: j['conformal'] is Map
          ? Map<String, dynamic>.from(j['conformal'])
          : null,
      updatedAt: updatedAt,
      trend: _objs(j['trend']).map(TrendPoint.fromJson).toList(),
    );
  }

  /// Round-trips through the same parser, so the on-device cache and the network
  /// path can never diverge in how they interpret a field.
  Map<String, dynamic> toJson() => {
        'subject_id': subjectId,
        'fusion_result_id': fusionResultId,
        'composite': compositeScore,
        'tier': tier,
        'band': band.protocolName,
        'confidence': confidence,
        'reason': reason,
        'renormalised': renormalised,
        'modalities_used': modalitiesUsed,
        'assessment_status': assessmentStatus,
        'missing_modalities': missingModalities,
        'weights': {for (final c in contributions) c.key: c.weight},
        'contributions': {for (final c in contributions) c.key: c.contribution},
        'modalities': {
          for (final e in modalities.entries) e.key: e.value.toJson()
        },
        'gate': gate,
        'conformal': conformal,
        'updated_at': updatedAt?.toIso8601String(),
        'trend': trend.map((t) => t.toJson()).toList(),
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// Alerts / audit
// ─────────────────────────────────────────────────────────────────────────────

enum AlertKind { riskEscalation, analysisComplete, system }

class ClinicalAlert {
  final String id;
  final String title;
  final String body;
  final DateTime raisedAt;
  final AlertKind kind;
  final AlertBand? band;
  final String? patientMrn;
  final String? patientName;
  final bool acknowledged;
  final String? acknowledgedBy;
  final DateTime? acknowledgedAt;

  const ClinicalAlert({
    required this.id,
    required this.title,
    required this.body,
    required this.raisedAt,
    this.kind = AlertKind.system,
    this.band,
    this.patientMrn,
    this.patientName,
    this.acknowledged = false,
    this.acknowledgedBy,
    this.acknowledgedAt,
  });

  ClinicalAlert acknowledge(String by) => ClinicalAlert(
        id: id,
        title: title,
        body: body,
        raisedAt: raisedAt,
        kind: kind,
        band: band,
        patientMrn: patientMrn,
        patientName: patientName,
        acknowledged: true,
        acknowledgedBy: by,
        acknowledgedAt: DateTime.now(),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'body': body,
        'raised_at': raisedAt.toIso8601String(),
        'kind': kind.name,
        'band': band?.name,
        'patient_mrn': patientMrn,
        'patient_name': patientName,
        'acknowledged': acknowledged,
        'acknowledged_by': acknowledgedBy,
        'acknowledged_at': acknowledgedAt?.toIso8601String(),
      };

  factory ClinicalAlert.fromJson(Map<String, dynamic> j) => ClinicalAlert(
        id: _s(j['id'], DateTime.now().microsecondsSinceEpoch.toString()),
        title: _s(j['title']),
        body: _s(j['body']),
        raisedAt: _dt(j['raised_at']),
        kind: AlertKind.values.firstWhere(
          (k) => k.name == _s(j['kind']),
          orElse: () => AlertKind.system,
        ),
        band: j['band'] == null
            ? null
            : AlertBand.values.firstWhere(
                (b) => b.name == _s(j['band']),
                orElse: () => AlertBand.green,
              ),
        patientMrn: j['patient_mrn'] == null ? null : _s(j['patient_mrn']),
        patientName: j['patient_name'] == null ? null : _s(j['patient_name']),
        acknowledged: _b(j['acknowledged']),
        acknowledgedBy:
            j['acknowledged_by'] == null ? null : _s(j['acknowledged_by']),
        acknowledgedAt:
            j['acknowledged_at'] == null ? null : _dt(j['acknowledged_at']),
      );
}
