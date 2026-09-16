import '../../core/config/env.dart';
import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/attention_event.dart';
import '../../domain/contracts/dashboard_snapshot.dart';
import '../../domain/repositories/assessment_repository.dart';
import '../../domain/repositories/attention_event_repository.dart';
import '../../domain/repositories/auth_repository.dart';
import '../../domain/repositories/dashboard_repository.dart';
import '../api/api_client.dart';
import '../api/session.dart';

const String _contractGateDetail =
    'Clinician target backend routes are not live-wired because the current '
    'Central Backend contract has not been verified to expose the required '
    'clinician principal, assignment-scoped dashboard, latest-assessment, and '
    'persistent attention-event operations.';

ApiException _contractGate(String operation) => ApiException(
      kind: ApiFailure.notConfigured,
      endpoint: operation,
      detail: _contractGateDetail,
    );

class CentralBackendAuthRepository implements AuthRepository {
  CentralBackendAuthRepository([ApiClient? api]);

  @override
  Future<void> validateCurrentSession() async {
    throw _contractGate('clinician-session-contract');
  }

  @override
  Future<void> expireCurrentSession() => Session.signOut();
}

class CentralBackendDashboardRepository implements DashboardRepository {
  CentralBackendDashboardRepository([ApiClient? api]);

  @override
  Future<DashboardSnapshot> loadDashboard() async {
    throw _contractGate('clinician-dashboard-contract');
  }
}

class CentralBackendAssessmentRepository implements AssessmentRepository {
  CentralBackendAssessmentRepository([ApiClient? api]);

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    throw _contractGate('latest-assessment-contract');
  }
}

/// Client-first Phase 6 adapter for the frozen target AttentionEvent contract.
///
/// The handbook defines these route semantics. The mobile client now implements
/// them exactly so the Central Backend can be brought up to this contract
/// independently. The server remains the authority for event creation,
/// assignment, lifecycle, actor identity and timestamps.
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
  Future<AttentionEvent> resolve(String eventId) async {
    final id = _requireEventId(eventId);
    final path = '$_eventsPath/${Uri.encodeComponent(id)}/resolve';
    final payload = await _api.post(path, const <String, dynamic>{});
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

  AttentionEvent _parseEvent(
    Map<String, dynamic> payload,
    String endpoint,
  ) {
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
