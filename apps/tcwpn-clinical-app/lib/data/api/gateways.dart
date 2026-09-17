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
//
// C3Gateway was DELETED. It posted to /v3/risk/classify, the retired
// intervention route, and nothing under lib/features/ ever called it. Keeping a
// dead client for a dead endpoint is how stale contracts creep back in.

import '../../core/config/env.dart';
import '../../domain/models.dart';
import '../../domain/evidence.dart';
import 'api_client.dart';

class CentralBackendGateway {
  final ApiClient _api;

  CentralBackendGateway([ApiClient? api])
      : _api = api ??
            ApiClient(
              Env.backendBase,
              bearer: () => Env.backendToken,
            );

  Future<Map<String, dynamic>?> health() async {
    try {
      return await _api.get('/health', timeout: const Duration(seconds: 30));
    } on ApiException {
      return null;
    }
  }

  Future<EnrolmentResult> enrol({
    required String mrn,
    String? enrolledBy,
  }) async {
    final json = await _api.post(
        '/v1/subjects',
        {
          'mrn': mrn,
          if (enrolledBy != null && enrolledBy.isNotEmpty)
            'enrolled_by': enrolledBy,
        },
        timeout: Env.quickTimeout);
    return EnrolmentResult.fromJson(json);
  }

  Future<String?> resolveAppUserId(String appUserId) async {
    try {
      final json = await _api.get(
        '/v1/subjects/resolve?app_user_id=${Uri.encodeQueryComponent(appUserId)}',
        timeout: Env.quickTimeout,
      );
      final id = json['subject_id'];
      return id == null ? null : '$id';
    } on ApiException catch (e) {
      if (e.kind == ApiFailure.notFound) return null;
      rethrow;
    }
  }

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

  Future<void> registerExternalId({
    required String subjectId,
    required String modality,
    required String externalId,
  }) =>
      _api.post(
          '/v1/subjects/$subjectId/external-ids',
          {
            'modality': modality,
            'external_id': externalId,
          },
          timeout: Env.quickTimeout);

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
      'return_attention': true,
      'return_support_contributions': true,
      if (author != null && author.isNotEmpty) 'author': author,
    });

    return ClinicalNoteIngestResult.fromJson(
      json,
      fallbackLatency: DateTime.now().difference(started).inMilliseconds,
    );
  }

  Future<void> runFusion(String subjectId, {String trigger = 'manual'}) =>
      _api.post('/v1/fusion/run', {
        'subject_id': subjectId,
        'trigger': trigger,
      });

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

  /// Raw transport retained for compatibility with older gateway-level tests.
  /// P5C production code consumes it only through the typed EvidenceRepository.
  Future<Map<String, dynamic>> evidence({
    required String subjectId,
    required String question,
  }) async {
    final trimmed = question.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError.value(question, 'question', 'must not be blank');
    }
    return _api.post(
      '/v1/doctor/patients/${Uri.encodeComponent(subjectId)}/evidence',
      {'question': trimmed},
      timeout: Env.inferenceTimeout,
    );
  }

  Future<EvidenceResult> askEvidence(String question) async {
    final trimmed = question.trim();
    if (trimmed.isEmpty) {
      return EvidenceResult.failure('Enter a question first.');
    }
    try {
      final json = await _api.post(
        '/v1/evidence/ask',
        {'question': trimmed},
        timeout: Env.inferenceTimeout,
      );
      return EvidenceResult.fromJson(json);
    } on ApiException catch (e) {
      return EvidenceResult.failure(e.toString());
    } catch (e) {
      return EvidenceResult.failure('$e');
    }
  }

  Future<Map<String, dynamic>> submitVerdict({
    required int fusionResultId,
    required String tierLabel,
    String? author,
    String? note,
  }) =>
      _api.post(
          '/v1/verdict',
          {
            'fusion_result_id': fusionResultId,
            'tier_label': tierLabel,
            if (author != null && author.isNotEmpty) 'author': author,
            if (note != null && note.isNotEmpty) 'note': note,
          },
          timeout: Env.quickTimeout);

  void dispose() => _api.close();
}

class TcwpnWarmupGateway {
  final ApiClient _api;
  TcwpnWarmupGateway([ApiClient? api]) : _api = api ?? ApiClient(Env.tcwpnBase);

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
