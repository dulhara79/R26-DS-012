import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/api/session.dart';
import 'package:r26_ds012_app/data/repositories/central_backend_repositories.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

void main() {
  tearDown(Session.clear);

  test('auth repository validates the clinician principal through /v1/me',
      () async {
    Session.set(token: 'clinician-jwt', clinicianId: 'DR001');
    late http.Request sent;
    final api = ApiClient(
      'https://backend.test',
      client: MockClient((request) async {
        sent = request;
        return http.Response(
          '{"clinician_id":"DR001","display_name":"Dr X","role":"clinician"}',
          200,
        );
      }),
    );

    await CentralBackendAuthRepository(api).validateCurrentSession();

    expect(sent.method, 'GET');
    expect(sent.url.path, '/v1/me');
    expect(sent.headers['authorization'], 'Bearer clinician-jwt');
  });

  test('dashboard repository uses clinician JWT and canonical dashboard route',
      () async {
    Session.set(token: 'clinician-jwt', clinicianId: 'DR001');
    late http.Request sent;
    final api = ApiClient(
      'https://backend.test',
      client: MockClient((request) async {
        sent = request;
        return http.Response('''{
          "clinician":{"clinician_id":"DR001","display_name":"Dr X"},
          "assigned_count":1,
          "open_attention_events":[{
            "id":"evt-001",
            "subject_id":"subject-001",
            "fusion_result_id":123,
            "forecast_result_id":"fcst-001",
            "event_type":"acute_escalation_forecast",
            "severity":"high",
            "reason":"Forecast crossed versioned escalation policy",
            "forecast_horizon":10,
            "status":"OPEN",
            "created_at":"2026-09-16T08:00:00Z",
            "policy_version":"escalation-v1"
          }],
          "patients":[{
            "subject_id":"subject-001",
            "display_id":"Patient A",
            "fusion_result_id":123,
            "current":{"score":0.58,"tier":"Medium"},
            "forecast":{"scope":"physiological","score":0.84,"tier":"High","horizon_minutes":10,"escalation_predicted":true},
            "assessment_status":"complete",
            "last_updated":"2026-09-16T08:00:00Z",
            "open_event_count":1
          }]
        }''', 200);
      }),
    );

    final snapshot = await CentralBackendDashboardRepository(api).loadDashboard();

    expect(sent.method, 'GET');
    expect(sent.url.path, '/v1/clinicians/me/dashboard');
    expect(sent.headers['authorization'], 'Bearer clinician-jwt');
    expect(snapshot.openEvents.single.id, 'evt-001');
    expect(snapshot.assignedPatients.single.fusionResultId, 123);
    expect(snapshot.assignedPatients.single.currentAssessment!.tier,
        RiskTier.medium);
    expect(snapshot.assignedPatients.single.forecast!.scope,
        ForecastScope.physiological);
  });

  test('latest assessment repository returns null only for 404', () async {
    final api = ApiClient(
      'https://backend.test',
      client: MockClient((request) async {
        expect(request.url.path,
            '/v1/patients/subject-001/assessment/latest');
        return http.Response('{"detail":"not found"}', 404);
      }),
      bearer: () => 'clinician-jwt',
    );

    final result = await CentralBackendAssessmentRepository(api)
        .latestAssessment('subject-001');

    expect(result, isNull);
  });
}
