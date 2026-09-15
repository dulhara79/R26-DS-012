import '../../core/config/env.dart';
import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/attention_event.dart';
import '../../domain/contracts/clinician_principal.dart';
import '../../domain/contracts/contract_parsing.dart';
import '../../domain/contracts/dashboard_snapshot.dart';
import '../../domain/contracts/patient_summary.dart';
import '../../domain/repositories/assessment_repository.dart';
import '../../domain/repositories/auth_repository.dart';
import '../../domain/repositories/dashboard_repository.dart';
import '../api/api_client.dart';
import '../api/session.dart';

class CentralBackendAuthRepository implements AuthRepository {
  final ApiClient _api;

  CentralBackendAuthRepository([ApiClient? api])
      : _api = api ?? ApiClient(Env.backendBase);

  @override
  Future<void> validateCurrentSession() async {
    final json = await _api.get('/v1/me', timeout: Env.quickTimeout);
    final principal = ClinicianPrincipal.fromJson(json);
    if (principal.clinicianId.trim().isEmpty) {
      throw const ApiException(
        kind: ApiFailure.malformed,
        endpoint: '/v1/me',
        detail: 'Authenticated principal response omitted clinician_id.',
      );
    }
  }

  @override
  Future<void> expireCurrentSession() => Session.signOut();
}

class CentralBackendDashboardRepository implements DashboardRepository {
  final ApiClient _api;

  CentralBackendDashboardRepository([ApiClient? api])
      : _api = api ?? ApiClient(Env.backendBase);

  @override
  Future<DashboardSnapshot> loadDashboard() async {
    final json = await _api.get(
      '/v1/clinicians/me/dashboard',
      timeout: Env.quickTimeout,
    );

    final eventRows = contractMapList(
      json['open_attention_events'] ?? json['attention_events'] ?? json['events'],
    );
    final patientRows = contractMapList(
      json['patients'] ?? json['patient_summaries'],
    );

    return DashboardSnapshot(
      openEvents:
          eventRows.map(AttentionEvent.fromJson).toList(growable: false),
      assignedPatients:
          patientRows.map(PatientSummary.fromJson).toList(growable: false),
      fetchedAt: contractDateTime(json['fetched_at'] ?? json['generated_at']) ??
          DateTime.now().toUtc(),
    );
  }
}

class CentralBackendAssessmentRepository implements AssessmentRepository {
  final ApiClient _api;

  CentralBackendAssessmentRepository([ApiClient? api])
      : _api = api ?? ApiClient(Env.backendBase);

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    try {
      final json = await _api.get(
        '/v1/patients/${Uri.encodeComponent(subjectId)}/assessment/latest',
        timeout: Env.quickTimeout,
      );
      return AssessmentSummary.fromJson(json);
    } on ApiException catch (e) {
      if (e.kind == ApiFailure.notFound) return null;
      rethrow;
    }
  }
}
