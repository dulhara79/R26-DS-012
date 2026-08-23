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

List<Map<String, dynamic>> _objs(dynamic v) => v is List
    ? v.whereType<Map>().map((e) => Map<String, dynamic>.from(e)).toList()
    : const [];

List<String> _strs(dynamic v) =>
    v is List ? v.map((e) => '$e').toList() : const [];

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
  });

  bool get isPositive => riskScore >= threshold;

  /// Low-confidence predictions must be flagged, not quietly rendered. The
  /// proposal's own safety notice sets the bar at 60%.
  bool get needsManualReview => confidence < 0.60;

  /// True when the service reported real attention mass for at least one span.
  bool get hasAttribution => spans.any((s) => s.hasWeight);

  factory TcwpnResult.fromJson(Map<String, dynamic> j, {int? fallbackLatency}) {
    final raw = _d(j['risk_score']);
    return TcwpnResult(
      prediction: _s(j['prediction'],
          raw >= _d(j['threshold'], .5) ? 'ANXIETY' : 'NO ANXIETY'),
      riskScore: raw,
      calibratedProbability: _d(j['calibrated_probability'] ?? j['risk_score']),
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
        'temporal_context': temporalContext,
        'model_version': modelVersion,
        'latency_ms': latencyMs,
        'used_default_support_set': usedDefaultSupportSet,
      };
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
class ClinicalNote {
  final String id;
  final String patientMrn;
  final DateTime recordedAt;
  final String text;
  final String noteType;
  final String clinicianId;
  final TcwpnResult? result;
  final String? clinicianComment;

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
    this.result,
    this.clinicianComment,
    this.clinicianVerdict,
  });

  bool get isDraft => result == null;

  ClinicalNote copyWith({
    TcwpnResult? result,
    String? clinicianComment,
    String? clinicianVerdict,
  }) =>
      ClinicalNote(
        id: id,
        patientMrn: patientMrn,
        recordedAt: recordedAt,
        text: text,
        noteType: noteType,
        clinicianId: clinicianId,
        result: result ?? this.result,
        clinicianComment: clinicianComment ?? this.clinicianComment,
        clinicianVerdict: clinicianVerdict ?? this.clinicianVerdict,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'patient_mrn': patientMrn,
        'recorded_at': recordedAt.toIso8601String(),
        'text': text,
        'note_type': noteType,
        'clinician_id': clinicianId,
        'result': result?.toJson(),
        'clinician_comment': clinicianComment,
        'clinician_verdict': clinicianVerdict,
      };

  factory ClinicalNote.fromJson(Map<String, dynamic> j) => ClinicalNote(
        id: _s(j['id'], DateTime.now().microsecondsSinceEpoch.toString()),
        patientMrn: _s(j['patient_mrn'] ?? j['patientId']),
        recordedAt: _dt(j['recorded_at'] ?? j['timestamp']),
        text: _s(j['text'] ?? j['noteText']),
        noteType: _s(j['note_type'] ?? j['noteType'], 'Psychiatry note'),
        clinicianId: _s(j['clinician_id'] ?? j['clinicianId']),
        result: j['result'] is Map
            ? TcwpnResult.fromJson(Map<String, dynamic>.from(j['result']))
            : null,
        clinicianComment:
            j['clinician_comment'] == null ? null : _s(j['clinician_comment']),
        clinicianVerdict:
            j['clinician_verdict'] == null ? null : _s(j['clinician_verdict']),
      );
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

/// C3 risk classification with APS conformal uncertainty (proposal §3.2 C3).
class C3Result {
  final int tier; // 0 Low, 1 Medium, 2 High
  final Map<String, double> calibratedProbabilities;
  final double confidence;
  final List<String> conformalSet; // APS prediction set at α = 0.10
  final double alpha;
  final String interventionType;
  final String priority; // P1..P5
  final List<ShapFeature> shap;
  final List<Counterfactual> counterfactuals;
  final List<SimilarCase> similarCases;
  final double? lastRewardNorm; // feature F12

  const C3Result({
    required this.tier,
    required this.calibratedProbabilities,
    required this.confidence,
    required this.conformalSet,
    this.alpha = 0.10,
    this.interventionType = 'routine_monitoring',
    this.priority = 'P5',
    this.shap = const [],
    this.counterfactuals = const [],
    this.similarCases = const [],
    this.lastRewardNorm,
  });

  /// A singleton conformal set means the model is confident enough to commit to
  /// one tier. Anything wider is genuine ambiguity and must be surfaced.
  bool get isSingleton => conformalSet.length == 1;
  bool get isAmbiguous => conformalSet.length > 1;

  String get tierLabel =>
      switch (tier) { 0 => 'Low', 1 => 'Medium', _ => 'High' };

  /// C3 contributes a 0..1 risk score to the fusion layer.
  double get riskScore => switch (tier) { 0 => 0.20, 1 => 0.55, _ => 0.85 };

  factory C3Result.fromJson(Map<String, dynamic> j) {
    final probs = <String, double>{};
    final rawProbs = j['calibrated_probabilities'] ?? j['probabilities'];
    if (rawProbs is Map) {
      rawProbs.forEach((k, v) => probs['$k'] = _d(v));
    }
    return C3Result(
      tier: _i(j['risk_tier']),
      calibratedProbabilities: probs,
      confidence: _d(j['confidence']),
      conformalSet: _strs(j['conformal_set']),
      alpha: _d(j['alpha'], 0.10),
      interventionType: _s(j['intervention_type'], 'routine_monitoring'),
      priority: _s(j['priority'], 'P5'),
      shap: _objs(j['shap']).map(ShapFeature.fromJson).toList(),
      counterfactuals:
          _objs(j['counterfactuals']).map(Counterfactual.fromJson).toList(),
      similarCases:
          _objs(j['similar_cases']).map(SimilarCase.fromJson).toList(),
      lastRewardNorm:
          j['last_reward_norm'] == null ? null : _d(j['last_reward_norm']),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Components 1 & 2 — passive modalities
// ─────────────────────────────────────────────────────────────────────────────

/// A modality reading fed into the fusion layer. C1 and C2 are owned by other
/// team members; this app consumes their scores and must render honestly when a
/// modality has no recent data.
class ModalityReading {
  final String key; // 'c1_physiological' | 'c2_behavioral' | …
  final double? score; // null = no data
  final DateTime? capturedAt;
  final String? note; // e.g. 'Device not worn since 12 Jun'

  const ModalityReading({
    required this.key,
    this.score,
    this.capturedAt,
    this.note,
  });

  bool get available => score != null;

  /// A reading older than this is stale and should not silently carry weight.
  bool get isStale =>
      capturedAt == null || DateTime.now().difference(capturedAt!).inHours > 72;

  factory ModalityReading.fromJson(String key, Map<String, dynamic> j) =>
      ModalityReading(
        key: key,
        score: j['score'] == null ? null : _d(j['score']),
        capturedAt: j['captured_at'] == null ? null : _dt(j['captured_at']),
        note: j['note'] == null ? null : _s(j['note']),
      );

  Map<String, dynamic> toWire() => {
        'score': score,
        'available': available,
        'captured_at': capturedAt?.toIso8601String(),
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// Late fusion (proposal §5.1)
// ─────────────────────────────────────────────────────────────────────────────

class ComponentContribution {
  final String key;
  final String label;
  final double weight; // fusion weight actually applied
  final double? score; // modality score, null when unavailable
  final double contribution; // weight × score

  /// When the underlying modality was captured.
  ///
  /// For C1 and C2 this timestamp was written by the patient-facing app, not by
  /// ClinAnx. It is the only way a clinician can tell whether the wearable
  /// signal beside today's note is from this morning or from three weeks ago.
  final DateTime? capturedAt;

  /// Free text from the fusion service explaining an absent reading, e.g.
  /// "device not worn since 12 Jun". Shown verbatim.
  final String? note;

  const ComponentContribution({
    required this.key,
    required this.label,
    required this.weight,
    required this.score,
    required this.contribution,
    this.capturedAt,
    this.note,
  });

  bool get available => score != null;

  /// Available, but old enough that it should be labelled as such.
  bool get isStale =>
      available &&
      (capturedAt == null ||
          DateTime.now().difference(capturedAt!) > const Duration(hours: 72));

  /// Human-readable age, for the chart. Null when there is no reading.
  String? get ageLabel {
    if (!available || capturedAt == null) return null;
    final d = DateTime.now().difference(capturedAt!);
    if (d.inMinutes < 60) return '${d.inMinutes}m ago';
    if (d.inHours < 24) return '${d.inHours}h ago';
    if (d.inDays < 30) return '${d.inDays}d ago';
    return '${(d.inDays / 30).floor()}mo ago';
  }

  /// Which app supplied this modality. Shown so a clinician understands that
  /// two of the four signals do not originate in ClinAnx.
  String get origin => switch (key) {
        'c1_physiological' => 'Patient app · wearable',
        'c2_behavioral' => 'Patient app · phone sensing',
        'c3_intervention' => 'Intervention engine',
        'c4_clinical_nlp' => 'ClinAnx · this app',
        _ => 'Unknown',
      };
}

/// Output of the late-fusion service. `renormalised` is important: when a
/// modality is missing, the remaining weights are rescaled, and the clinician
/// must be told the composite was computed on partial evidence.
class FusionResult {
  final String patientMrn;
  final double compositeScore;
  final AlertBand band;
  final List<ComponentContribution> contributions;
  final bool renormalised;
  final int modalitiesAvailable;
  final double? confidence;
  final DateTime computedAt;
  final bool computedLocally; // true when the fusion service was unreachable

  const FusionResult({
    required this.patientMrn,
    required this.compositeScore,
    required this.band,
    required this.contributions,
    required this.renormalised,
    required this.modalitiesAvailable,
    this.confidence,
    required this.computedAt,
    this.computedLocally = false,
  });

  static const _labels = {
    'c1_physiological': 'Physiological',
    'c2_behavioral': 'Behavioural',
    'c3_intervention': 'Intervention',
    'c4_clinical_nlp': 'Clinical notes',
  };

  static String labelFor(String key) => _labels[key] ?? key;

  factory FusionResult.fromJson(Map<String, dynamic> j, String mrn) {
    final weights = <String, double>{};
    if (j['weights'] is Map) {
      (j['weights'] as Map).forEach((k, v) => weights['$k'] = _d(v));
    }
    final scores = <String, double?>{};
    if (j['scores'] is Map) {
      (j['scores'] as Map)
          .forEach((k, v) => scores['$k'] = v == null ? null : _d(v));
    }

    // Per-modality metadata. Accepts either a flat `captured_at` map or a
    // richer `components` map of objects, so the fusion service can evolve
    // without breaking the client.
    final capturedAt = <String, DateTime?>{};
    final notes = <String, String?>{};
    if (j['captured_at'] is Map) {
      (j['captured_at'] as Map)
          .forEach((k, v) => capturedAt['$k'] = v == null ? null : _dt(v));
    }
    if (j['components'] is Map) {
      (j['components'] as Map).forEach((k, v) {
        if (v is! Map) return;
        final m = Map<String, dynamic>.from(v);
        if (m['captured_at'] != null) capturedAt['$k'] = _dt(m['captured_at']);
        if (m['note'] != null) notes['$k'] = _s(m['note']);
        if (m.containsKey('score')) {
          scores['$k'] = m['score'] == null ? null : _d(m['score']);
        }
      });
    }

    final contribs = <ComponentContribution>[];
    for (final key in _labels.keys) {
      final w = weights[key] ?? 0;
      final s = scores[key];
      contribs.add(ComponentContribution(
        key: key,
        label: labelFor(key),
        weight: w,
        score: s,
        contribution: (s ?? 0) * w,
        capturedAt: capturedAt[key],
        note: notes[key],
      ));
    }
    final composite = _d(j['composite_score']);
    return FusionResult(
      patientMrn: mrn,
      compositeScore: composite,
      band: j['alert_level'] != null
          ? AlertBandX.fromWire(_s(j['alert_level']))
          : AlertBandX.fromScore(composite),
      contributions: contribs,
      renormalised: _b(j['renormalised']),
      modalitiesAvailable: _i(
          j['modalities_available'], contribs.where((c) => c.available).length),
      confidence: j['confidence'] == null ? null : _d(j['confidence']),
      computedAt: _dt(j['computed_at']),
    );
  }

  /// Provisional client-side fusion, used only when the fusion service cannot
  /// be reached. Renormalises across available modalities and is always
  /// labelled as provisional in the UI — it is never presented as the
  /// framework's composite score.
  factory FusionResult.local({
    required String mrn,
    required Map<String, ModalityReading> readings,
    Map<String, double> weights = const {
      'c1_physiological': 0.25,
      'c2_behavioral': 0.20,
      'c3_intervention': 0.15,
      'c4_clinical_nlp': 0.40,
    },
  }) {
    final present =
        weights.keys.where((k) => readings[k]?.available == true).toList();
    final totalW = present.fold<double>(0, (a, k) => a + (weights[k] ?? 0));
    final renorm = present.length < weights.length && totalW > 0;

    double composite = 0;
    final contribs = <ComponentContribution>[];
    for (final key in weights.keys) {
      final baseW = weights[key] ?? 0;
      final r = readings[key];
      final effW = (r?.available == true && totalW > 0)
          ? (renorm ? baseW / totalW : baseW)
          : 0.0;
      final c = (r?.score ?? 0) * effW;
      composite += c;
      contribs.add(ComponentContribution(
        key: key,
        label: labelFor(key),
        weight: effW,
        score: r?.score,
        contribution: c,
        capturedAt: r?.capturedAt,
        note: r?.note,
      ));
    }

    return FusionResult(
      patientMrn: mrn,
      compositeScore: composite.clamp(0.0, 1.0),
      band: AlertBandX.fromScore(composite),
      contributions: contribs,
      renormalised: renorm,
      modalitiesAvailable: present.length,
      computedAt: DateTime.now(),
      computedLocally: true,
    );
  }
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
