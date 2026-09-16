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

class CentralBackendAttentionEventRepository
    implements AttentionEventRepository {
  CentralBackendAttentionEventRepository([ApiClient? api]);

  @override
  Future<List<AttentionEvent>> openEvents() async {
    throw _contractGate('attention-events-open-contract');
  }

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async {
    throw _contractGate('attention-events-activity-contract');
  }

  @override
  Future<AttentionEvent?> eventById(String eventId) async {
    throw _contractGate('attention-event-detail-contract');
  }

  @override
  Future<AttentionEvent> acknowledge(String eventId) async {
    throw _contractGate('attention-event-acknowledge-contract');
  }

  @override
  Future<AttentionEvent> resolve(String eventId) async {
    throw _contractGate('attention-event-resolve-contract');
  }
}
