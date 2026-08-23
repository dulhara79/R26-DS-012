// lib/data/api/gateways.dart
//
// ONE gateway for the clinician workflow: the R26-DS-012 Central Backend.
//
// WHAT WAS REMOVED, AND WHY
// -------------------------
// `FusionGateway` posted to `/contribute` and read `GET /state/{mrn}`. Those
// routes exist in NO service in any repository. The fusion service serves
// /v1/fuse, /v1/fuse/manual, /v1/physio/tick and /v1/patients/{mrn}/state; the
// Central Backend serves none of them. Every fusion call in the previous build
// therefore fell through to a local composite computed from a client-side weight
// table, and was displayed with a "provisional" label that concealed the fact
// that the framework score had never been fetched at all.
//
// `TcwpnGateway.analyse()` posted directly to the Space's /predict. That path is
// gone from the clinician workflow: the backend owns ingestion (its own
// modality_clients.py header says so), and it is the backend that applies the
// gate, the recency and reliability weighting, the harmonisation and the
// conformal calibration. An app calling /predict directly skips all of it.
//
// WHAT REMAINS
// ------------
//   CentralBackendGateway  — enrolment, note ingestion, fusion, timeline,
//                            evidence, verdict.
//   TcwpnWarmupGateway     — /health only, to wake a sleeping Space.
//   C3Gateway              — the Personalised Intervention Framework, called
//                            directly. NOT a fusion modality; see Modality.

import '../../core/config/env.dart';
import '../../domain/models.dart';
import 'api_client.dart';
import 'session.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Central Backend
// ─────────────────────────────────────────────────────────────────────────────

class CentralBackendGateway {
  final ApiClient _api;

  CentralBackendGateway([ApiClient? api])
      : _api = api ??
            ApiClient(
              Env.backendBase,
              // The backend checks a single shared token (main.py::_auth), not
              // the clinician's session JWT. Sending the JWT here yields 401 on
              // every call. Clinician identity travels in the body's `author`
              // field instead — a documented prototype limitation, not a design.
              bearer: () => Env.backendToken,
            );

  Future<Map<String, dynamic>?> health() async {
    try {
      return await _api.get('/health', timeout: const Duration(seconds: 30));
    } on ApiException {
      return null;
    }
  }

  // ── Enrolment ─────────────────────────────────────────────────────────────

  /// Enrols a patient by MRN. The backend HMAC-hashes it on arrival and never
  /// persists the raw value; what comes back is an opaque `subject_id` plus a
  /// pairing code for the patient app.
  ///
  /// Re-enrolling a known MRN is safe: the backend returns the existing subject
  /// with a fresh code rather than creating a duplicate patient.
  Future<EnrolmentResult> enrol({
    required String mrn,
    String? enrolledBy,
  }) async {
    final json = await _api.post('/v1/subjects', {
      'mrn': mrn,
      if (enrolledBy != null && enrolledBy.isNotEmpty) 'enrolled_by': enrolledBy,
    }, timeout: Env.quickTimeout);
    return EnrolmentResult.fromJson(json);
  }

  /// Resolves an already-enrolled MRN to its subject_id.
  ///
  /// Returns null when the backend has never seen this MRN — a normal state for
  /// a patient added to the local roster but not yet enrolled, not an error.
  ///
  /// Note that this sends the raw MRN over the wire so the server can hash it
  /// with its pepper; the app cannot compute the hash itself. Keep it to HTTPS,
  /// and prefer the stored subject_id for everything afterwards.
  Future<String?> resolveMrn(String mrn) async {
    try {
      final json = await _api.get(
        '/v1/subjects/resolve?mrn=${Uri.encodeQueryComponent(mrn)}',
        timeout: Env.quickTimeout,
      );
      final id = json['subject_id'];
      return id == null ? null : '$id';
    } on ApiException catch (e) {
      if (e.kind == ApiFailure.notFound) return null;
      rethrow;
    }
  }

  /// Registers the id a component service knows this patient by, so the backend
  /// does not ask C2 about a UUID C2 has never heard of.
  ///
  /// `modality` must be one of `c1_physiological`, `c2_behavioral`,
  /// `c3_clinical_nlp`. Idempotent per modality; the backend returns 409 if that
  /// external id already belongs to a different subject, which is a real
  /// cross-patient error and must surface, not be swallowed.
  Future<void> registerExternalId({
    required String subjectId,
    required String modality,
    required String externalId,
  }) =>
      _api.post('/v1/subjects/$subjectId/external-ids', {
        'modality': modality,
        'external_id': externalId,
      }, timeout: Env.quickTimeout);

  // ── Clinical note ─────────────────────────────────────────────────────────

  /// Submits one clinical note.
  ///
  /// Server-side this single call runs TC-WPN, stores the reading, and triggers
  /// fusion. The support set is sent with per-note dates because TC-WPN's
  /// temporal weighting and its visit-regularity term are computed from them
  /// server-side; the client must not pre-weight anything.
  ///
  /// `author` is the clinician id. It is the only clinician attribution the
  /// backend receives, because the transport token is a shared app credential.
  Future<ClinicalNoteIngestResult> submitNote({
    required String subjectId,
    required String noteText,
    required String noteType,
    required DateTime noteDate,
    required List<SupportNote> supportSet,
    required int visitCount,
    String? author,
  }) async {
    final started = DateTime.now();
    final json = await _api.post('/v1/clinical-notes', {
      'subject_id': subjectId,
      'note_text': noteText,
      'note_type': noteType,
      'note_date': noteDate.toUtc().toIso8601String(),
      'visit_count': visitCount,
      'support_set': supportSet.map((n) => n.toWire()).toList(),
      // Ask for the explanation payload. A service that does not implement it
      // omits the fields and the UI degrades honestly rather than inventing
      // attention weights.
      'return_attention': true,
      'return_support_contributions': true,
      if (author != null && author.isNotEmpty) 'author': author,
    });

    return ClinicalNoteIngestResult.fromJson(
      json,
      fallbackLatency: DateTime.now().difference(started).inMilliseconds,
    );
  }

  // ── Fusion and egress ─────────────────────────────────────────────────────

  /// Re-runs fusion over the stored readings. Does not call any component
  /// service — it re-derives the composite from what is already persisted.
  Future<void> runFusion(String subjectId, {String trigger = 'manual'}) =>
      _api.post('/v1/fusion/run', {
        'subject_id': subjectId,
        'trigger': trigger,
      });

  /// The clinician view: composite, per-modality readings with freshness and
  /// status, the gate decision, the conformal set, and the trend history.
  ///
  /// Returns null only when the backend has no such subject.
  Future<FusionResult?> timeline({
    required String subjectId,
    required String mrn,
    int limit = 20,
  }) async {
    try {
      final json = await _api.get(
        '/v1/doctor/patients/$subjectId/timeline?limit=$limit',
        timeout: Env.quickTimeout,
      );
      return FusionResult.fromJson(json, mrn);
    } on ApiException catch (e) {
      if (e.kind == ApiFailure.notFound) return null;
      rethrow;
    }
  }

  /// CARE-AnxRAG decision support. The backend forwards no patient data into
  /// the RAG call; subject_id is used for auth and audit only.
  Future<Map<String, dynamic>> evidence({
    required String subjectId,
    required String question,
  }) =>
      _api.post('/v1/doctor/patients/$subjectId/evidence', {
        'question': question,
      });

  /// Records the clinician's tier judgement against a SPECIFIC fusion row.
  ///
  /// Two ordering rules, both from the backend's own docstring, and both the
  /// UI's responsibility to enforce:
  ///   • the verdict must be entered BEFORE the conformal set is shown, or the
  ///     label is contaminated by the prediction it exists to calibrate;
  ///   • `fusionResultId` must be the id of the row the clinician actually
  ///     looked at, not the latest one.
  Future<Map<String, dynamic>> submitVerdict({
    required int fusionResultId,
    required String tierLabel, // 'Low' | 'Medium' | 'High'
    String? author,
    String? note,
  }) =>
      _api.post('/v1/verdict', {
        'fusion_result_id': fusionResultId,
        'tier_label': tierLabel,
        if (author != null && author.isNotEmpty) 'author': author,
        if (note != null && note.isNotEmpty) 'note': note,
      }, timeout: Env.quickTimeout);

  void dispose() => _api.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// TC-WPN — warm-up only
// ─────────────────────────────────────────────────────────────────────────────

/// Wakes a sleeping Space and reports model metadata, so the first real
/// analysis does not pay the full cold-start. This is the ONLY call this app
/// makes to the Space; inference goes through the Central Backend.
class TcwpnWarmupGateway {
  final ApiClient _api;
  TcwpnWarmupGateway([ApiClient? api])
      : _api = api ?? ApiClient(Env.tcwpnBase);

  Future<Map<String, dynamic>?> health() async {
    if (!Env.hasTcwpnWarmup) return null;
    try {
      return await _api.get('/health', timeout: const Duration(seconds: 30));
    } on ApiException {
      return null;
    }
  }

  void dispose() => _api.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 3 — intervention engine
// ─────────────────────────────────────────────────────────────────────────────

/// The Personalised Intervention Framework.
///
/// NOT a fusion modality. The backend's composite is built from four modalities
/// and intervention is not one of them, so nothing this class returns may be
/// rendered as a contribution to the composite. It is shown as its own section.
class C3Gateway {
  final ApiClient _api;
  C3Gateway([ApiClient? api])
      : _api = api ?? ApiClient(Env.c3Base, bearer: () => Session.token ?? '');

  /// Calibrated XGBoost + APS conformal classification.
  ///
  /// `textualRisk` is TC-WPN's score as the backend stored it — a genuinely
  /// independent modality, not a transform of the GAD-7 score.
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