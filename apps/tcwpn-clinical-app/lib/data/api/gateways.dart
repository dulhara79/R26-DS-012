// lib/data/api/gateways.dart
//
// Two gateways, matching the two services ClinAnx talks to.
//
//   TcwpnGateway   — submit a clinical note, get Component 4's assessment.
//   FusionGateway  — contribute that assessment, read the fused composite.
//
//   C3Gateway      — optional third, for the intervention engine.
//
// There is no C1 or C2 gateway, by design. The patient-facing app collects
// wearable and behavioural data and pushes it to the fusion service itself.
// ClinAnx sees those modalities only in the fused state it reads back.

import '../../core/config/env.dart';
import '../../domain/models.dart';
import 'api_client.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Component 4 — TC-WPN
// ─────────────────────────────────────────────────────────────────────────────

class TcwpnGateway {
  final ApiClient _api;
  TcwpnGateway([ApiClient? api]) : _api = api ?? ApiClient(Env.tcwpnBase);

  /// Wakes a sleeping Space and reports model metadata. Called once at launch
  /// so the first real analysis does not pay the full cold-start.
  Future<Map<String, dynamic>?> health() async {
    try {
      return await _api.get('/health', timeout: const Duration(seconds: 30));
    } on ApiException {
      return null;
    }
  }

  /// Analyse one clinical note against this patient's support set.
  ///
  /// The support set is sent with per-note dates because TC-WPN's temporal
  /// weighting — exp(-λ·Δt/365), λ = 0.5 — and its visit-regularity term are
  /// computed server-side from them. The client must not pre-weight anything.
  Future<TcwpnResult> analyse({
    required String patientMrn,
    required String noteText,
    required String noteType,
    required DateTime noteDate,
    required List<SupportNote> supportSet,
    required int visitCount,
  }) async {
    final started = DateTime.now();
    final json = await _api.post('/predict', {
      'patient_id': patientMrn,
      'note_text': noteText,
      'note_type': noteType,
      'note_date': noteDate.toIso8601String(),
      'visit_count': visitCount,
      'support_set': supportSet.map((n) => n.toWire()).toList(),
      // Ask for the full explanation payload. Services that don't implement it
      // simply omit the fields and the UI degrades honestly.
      'return_attention': true,
      'return_support_contributions': true,
    });

    return TcwpnResult.fromJson(
      json,
      fallbackLatency: DateTime.now().difference(started).inMilliseconds,
    );
  }

  void dispose() => _api.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// Late fusion — proposal §5.1
// ─────────────────────────────────────────────────────────────────────────────

class FusionGateway {
  final ApiClient _api;
  FusionGateway([ApiClient? api]) : _api = api ?? ApiClient(Env.fusionBase);

  /// Read the current fused state for a patient.
  ///
  /// The response carries all four modality snapshots, including the two this
  /// app never collects. C1 and C2 arrive with their own capture timestamps,
  /// written by the patient app, so the clinician can see how fresh the passive
  /// signals are relative to the note they just wrote.
  Future<FusionResult?> state(String mrn) async {
    if (!Env.hasFusion) return null;
    try {
      final json = await _api.get('/state/$mrn');
      return FusionResult.fromJson(json, mrn);
    } on ApiException catch (e) {
      // A patient with no fused record yet is a normal state, not an error.
      if (e.kind == ApiFailure.notFound) return null;
      rethrow;
    }
  }

  /// Contribute Component 4's assessment and receive the recomputed composite.
  ///
  /// This is the single point at which TC-WPN's output enters the framework.
  /// The fusion service combines it with whatever C1, C2 and C3 have most
  /// recently supplied, renormalises across the modalities that actually
  /// reported, and returns the composite with its alert band.
  Future<FusionResult> contributeClinicalNlp({
    required String mrn,
    required TcwpnResult result,
    required String noteId,
    required DateTime noteDate,
  }) async {
    if (!Env.hasFusion) {
      // No fusion service configured: compute a provisional composite locally
      // from the one modality we hold. Clearly labelled in the UI.
      return FusionResult.local(
        mrn: mrn,
        readings: {
          'c4_clinical_nlp': ModalityReading(
            key: 'c4_clinical_nlp',
            score: result.calibratedProbability,
            capturedAt: noteDate,
          ),
        },
      );
    }

    try {
      final json = await _api.post('/contribute', {
        'patient_id': mrn,
        'component': 'c4_clinical_nlp',
        'score': result.calibratedProbability,
        'confidence': result.confidence,
        'captured_at': noteDate.toIso8601String(),
        'source_id': noteId,
        'model_version': result.modelVersion,
        // Sent so the fusion layer can down-weight or flag a low-confidence
        // contribution rather than treating every C4 score as equally solid.
        'needs_review': result.needsManualReview,
        'renormalise_on_missing': true,
      });
      return FusionResult.fromJson(json, mrn);
    } on ApiException {
      return FusionResult.local(
        mrn: mrn,
        readings: {
          'c4_clinical_nlp': ModalityReading(
            key: 'c4_clinical_nlp',
            score: result.calibratedProbability,
            capturedAt: noteDate,
          ),
        },
      );
    }
  }

  /// Contribute Component 3's tier when the intervention engine has run.
  Future<FusionResult?> contributeIntervention({
    required String mrn,
    required C3Result result,
  }) async {
    if (!Env.hasFusion) return null;
    try {
      final json = await _api.post('/contribute', {
        'patient_id': mrn,
        'component': 'c3_intervention',
        'score': result.riskScore,
        'confidence': result.confidence,
        'captured_at': DateTime.now().toIso8601String(),
        'renormalise_on_missing': true,
      });
      return FusionResult.fromJson(json, mrn);
    } on ApiException {
      return null;
    }
  }

  void dispose() => _api.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 3 — intervention engine
// ─────────────────────────────────────────────────────────────────────────────

class C3Gateway {
  final ApiClient _api;
  C3Gateway([ApiClient? api]) : _api = api ?? ApiClient(Env.c3Base);

  /// Calibrated XGBoost + APS conformal classification.
  ///
  /// `textualRisk` is TC-WPN's calibrated probability — a genuinely independent
  /// modality, not a transform of the GAD-7 score. Physiological and behavioural
  /// risks are passed through from the fused state when available and as null
  /// otherwise, so the service can impute rather than receive a fabricated
  /// value.
  Future<C3Result> classify({
    required Patient patient,
    required List<int> gad7Answers,
    double? physiologicalRisk,
    double? behavioralRisk,
    double? textualRisk,
    double? lastRewardNorm,
    int interactionCount = 0,
    int escalationCount = 0,
  }) async {
    final gad7 = gad7Answers.fold<int>(0, (a, b) => a + b);
    final json = await _api.post('/v3/risk/classify', {
      'patient_id': patient.mrn,
      'gad7_score': gad7,
      'gad7_answers': gad7Answers,
      'features': {
        ...patient.c3Demographics(),
        'physiological_risk': physiologicalRisk,
        'behavioral_risk': behavioralRisk,
        'textual_risk': textualRisk,
        'interaction_count_norm': (interactionCount / 100.0).clamp(0.0, 1.0),
        'escalation_count_norm': (escalationCount / 10.0).clamp(0.0, 1.0),
        'last_reward_norm': lastRewardNorm,
      },
      'alpha': 0.10,
      'explain': true,
      'retrieve_cases': true,
    });
    return C3Result.fromJson(json);
  }

  void dispose() => _api.close();
}