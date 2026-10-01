import '../../core/config/env.dart';
import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/attention_event.dart';
import '../../domain/contracts/clinician_principal.dart';
import '../../domain/contracts/contract_enums.dart';
import '../../domain/contracts/dashboard_snapshot.dart';
import '../../domain/contracts/patient_summary.dart';
import '../../domain/repositories/assessment_repository.dart';
import '../../domain/repositories/attention_event_repository.dart';
import '../../domain/repositories/auth_repository.dart';
import '../../domain/repositories/dashboard_repository.dart';
import '../../domain/repositories/patient_repository.dart';
import '../api/api_client.dart';
import '../api/session.dart';

class CentralBackendAuthRepository implements AuthRepository {
  CentralBackendAuthRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  final ApiClient _api;

  @override
  Future<void> validateCurrentSession() async {
    final payload = await _api.get('/v1/me');
    final principal = ClinicianPrincipal.fromJson(payload);
    if (!principal.isValid) {
      throw const ApiException(
        kind: ApiFailure.malformed,
        endpoint: '/v1/me',
        detail: 'The clinician principal is missing clinician_id.',
      );
    }
    final role = principal.role?.trim().toLowerCase();
    if (role != null &&
        role.isNotEmpty &&
        role != 'clinician' &&
        role != 'doctor') {
      throw const ApiException(
        kind: ApiFailure.forbidden,
        endpoint: '/v1/me',
        detail: 'The authenticated principal is not a clinician.',
      );
    }

    final localClinicianId = Session.clinicianId?.trim() ?? '';
    if (localClinicianId.isNotEmpty &&
        localClinicianId != principal.clinicianId.trim()) {
      throw const ApiException(
        kind: ApiFailure.forbidden,
        endpoint: '/v1/me',
        detail: 'The authenticated clinician does not match the local session identity.',
      );
    }

    final token = Session.token;
    if (localClinicianId.isEmpty && token != null && token.isNotEmpty) {
      // The Central Backend principal is authoritative. Binding an otherwise
      // unbound in-memory session here prevents clinical cache access before
      // server identity has been established.
      Session.set(token: token, clinicianId: principal.clinicianId);
    }
  }

  @override
  Future<void> expireCurrentSession() => Session.signOut();
}

class CentralBackendAssessmentRepository implements AssessmentRepository {
  CentralBackendAssessmentRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  final ApiClient _api;

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    final id = _requireSubjectId(subjectId);
    final path = '/v1/patients/${Uri.encodeComponent(id)}/assessment/latest';

    Map<String, dynamic> payload;
    try {
      payload = await _api.get(path);
    } on ApiException catch (error) {
      if (error.kind == ApiFailure.notFound) return null;
      rethrow;
    }

    final assessment = AssessmentSummary.fromJson(payload);
    final unavailable =
        assessment.assessmentStatus == AssessmentStatus.unavailable;
    final invalidUnavailable =
        unavailable &&
        (assessment.fusionResultId != null ||
            assessment.currentAssessment.score != null ||
            assessment.currentAssessment.tier != RiskTier.unknown ||
            assessment.currentAssessment.band != null);
    final invalidAssessment =
        !unavailable &&
        (assessment.fusionResultId == null ||
            assessment.fusionResultId! <= 0 ||
            (assessment.modelVersion ?? '').trim().isEmpty);
    if (assessment.subjectId.trim() != id ||
        invalidUnavailable ||
        invalidAssessment) {
      throw ApiException(
        kind: ApiFailure.malformed,
        endpoint: path,
        detail: 'Latest assessment did not preserve the requested subject and authoritative fusion provenance.',
      );
    }
    return assessment;
  }
}

class CentralBackendPatientRepository implements PatientRepository {
  CentralBackendPatientRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  final ApiClient _api;

  @override
  Future<List<PatientSummary>> assignedPatients() async {
    const path = '/v1/clinicians/me/patients';
    final payload = await _api.get(path);
    final rawPatients = payload['patients'];
    if (rawPatients is! List) {
      throw const ApiException(
        kind: ApiFailure.malformed,
        endpoint: path,
        detail: 'Expected a patients array.',
      );
    }

    final summaries = <PatientSummary>[];
    final seen = <String>{};
    for (final raw in rawPatients) {
      if (raw is! Map) {
        throw const ApiException(
          kind: ApiFailure.malformed,
          endpoint: path,
          detail: 'Expected every patients item to be an object.',
        );
      }

      final row = Map<String, dynamic>.from(raw);
      final subjectId = (row['subject_id'] ?? '').toString().trim();
      if (subjectId.isEmpty || !seen.add(subjectId)) {
        throw const ApiException(
          kind: ApiFailure.malformed,
          endpoint: path,
          detail: 'Assigned-patient roster contains a missing or duplicate subject_id.',
        );
      }

      final summary = PatientSummary.fromJson(row);
      if (summary.subjectId != subjectId) {
        throw const ApiException(
          kind: ApiFailure.malformed,
          endpoint: path,
          detail: 'Assigned-patient summary changed subject identity.',
        );
      }
      summaries.add(summary);
    }

    return List.unmodifiable(summaries);
  }
}

class CentralBackendDashboardRepository implements DashboardRepository {
  CentralBackendDashboardRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  final ApiClient _api;

  @override
  Future<DashboardSnapshot> loadDashboard() async {
    const path = '/v1/clinicians/me/dashboard';
    final payload = await _api.get(path);
    final snapshot = DashboardSnapshot.fromJson({
      ...payload,
      // Client fetch time is cache metadata only, never clinical authority.
      'fetched_at': DateTime.timestamp().toUtc().toIso8601String(),
    }, isFromCache: false);
    if (snapshot.assignedCount != snapshot.assignedPatients.length) {
      throw const ApiException(
        kind: ApiFailure.malformed,
        endpoint: path,
        detail: 'Dashboard assigned_count does not match patients.',
      );
    }
    return snapshot;
  }
}

/// Adapter for the frozen server-owned AttentionEvent contract.
///
/// The Central Backend owns event creation, assignment, lifecycle, actor
/// identity and timestamps. ClinAnx only retrieves and mutates the canonical
/// server event through the frozen HTTP operations below.
class CentralBackendAttentionEventRepository
    implements AttentionEventRepository {
  CentralBackendAttentionEventRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  static const String _eventsPath = '/v1/attention-events';

  final ApiClient _api;

  @override
  Future<List<AttentionEvent>> openEvents() async {
    final path = _listPath(status: 'OPEN');
    final payload = await _api.get(path);
    return _parseEventList(payload, path);
  }

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async {
    final normalizedSubject = subjectId?.trim();
    final path = _listPath(
      subjectId: normalizedSubject == null || normalizedSubject.isEmpty
          ? null
          : normalizedSubject,
    );
    final payload = await _api.get(path);
    return _parseEventList(payload, path);
  }

  @override
  Future<AttentionEvent?> eventById(String eventId) async {
    final id = _requireEventId(eventId);
    final path = '$_eventsPath/${Uri.encodeComponent(id)}';
    final payload = await _api.get(path);
    return _parseEvent(payload, path);
  }

  @override
  Future<AttentionEvent> acknowledge(String eventId) async {
    final id = _requireEventId(eventId);
    final path = '$_eventsPath/${Uri.encodeComponent(id)}/acknowledge';
    final payload = await _api.post(path, const <String, dynamic>{});
    return _parseEvent(payload, path);
  }

  @override
  Future<AttentionEvent> resolve(String eventId, {String? note}) async {
    final id = _requireEventId(eventId);
    final path = '$_eventsPath/${Uri.encodeComponent(id)}/resolve';
    final normalizedNote = note?.trim();
    final payload = await _api.post(path, <String, dynamic>{
      if (normalizedNote != null && normalizedNote.isNotEmpty)
        'note': normalizedNote,
    });
    return _parseEvent(payload, path);
  }

  String _listPath({String? status, String? subjectId}) {
    final query = <String, String>{
      if (status != null) 'status': status,
      if (subjectId != null) 'subject_id': subjectId,
    };
    if (query.isEmpty) return _eventsPath;
    return Uri(path: _eventsPath, queryParameters: query).toString();
  }

  String _requireEventId(String eventId) {
    final id = eventId.trim();
    if (id.isEmpty) {
      throw const ApiException(
        kind: ApiFailure.validation,
        endpoint: 'attention-event-id',
        detail: 'event_id must not be blank.',
      );
    }
    return id;
  }

  List<AttentionEvent> _parseEventList(
    Map<String, dynamic> payload,
    String endpoint,
  ) {
    final rawEvents = payload['events'];
    if (rawEvents is! List) {
      throw ApiException(
        kind: ApiFailure.malformed,
        endpoint: endpoint,
        detail: 'Expected an events array.',
      );
    }

    final events = <AttentionEvent>[];
    for (final raw in rawEvents) {
      if (raw is! Map) {
        throw ApiException(
          kind: ApiFailure.malformed,
          endpoint: endpoint,
          detail: 'Expected every events item to be an object.',
        );
      }
      events.add(_eventFromMap(raw, endpoint));
    }
    return events;
  }

  AttentionEvent _parseEvent(Map<String, dynamic> payload, String endpoint) {
    final raw = payload['event'];
    if (raw is! Map) {
      throw ApiException(
        kind: ApiFailure.malformed,
        endpoint: endpoint,
        detail: 'Expected an event object.',
      );
    }
    return _eventFromMap(raw, endpoint);
  }

  AttentionEvent _eventFromMap(Map raw, String endpoint) {
    final event = AttentionEvent.fromJson(Map<String, dynamic>.from(raw));
    if (event.id.trim().isEmpty || event.subjectId.trim().isEmpty) {
      throw ApiException(
        kind: ApiFailure.malformed,
        endpoint: endpoint,
        detail: 'AttentionEvent is missing id or subject_id.',
      );
    }
    return event;
  }
}

String _requireSubjectId(String subjectId) {
  final id = subjectId.trim();
  if (id.isEmpty) {
    throw const ApiException(
      kind: ApiFailure.validation,
      endpoint: 'subject-id',
      detail: 'subject_id must not be blank.',
    );
  }
  return id;
}
