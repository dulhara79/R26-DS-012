import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/assessment_repository.dart';
import 'package:r26_ds012_app/domain/repositories/auth_repository.dart';
import 'package:r26_ds012_app/state/async_data_state.dart';
import 'package:r26_ds012_app/state/patient_overview_controller.dart';

class _AssessmentRepository implements AssessmentRepository {
  AssessmentSummary? value;
  Object? failure;
  String? requestedSubjectId;

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    requestedSubjectId = subjectId;
    if (failure != null) throw failure!;
    return value;
  }
}

class _AuthRepository implements AuthRepository {
  int expirations = 0;

  @override
  Future<void> validateCurrentSession() async {}

  @override
  Future<void> expireCurrentSession() async {
    expirations++;
  }
}

AssessmentSummary _assessment() => AssessmentSummary(
      subjectId: 'subject-001',
      fusionResultId: 123,
      currentAssessment: const CurrentAssessment(
        score: 0.58,
        tier: RiskTier.medium,
        band: 'AMBER',
      ),
      forecast: ForecastResult(
        forecastResultId: 'fcst-001',
        scope: ForecastScope.physiological,
        horizonMinutes: 10,
        score: 0.84,
        tier: RiskTier.high,
        escalationProbability: null,
        escalationPredicted: true,
        generatedAt: DateTime.utc(2026, 9, 16, 8),
        validUntil: DateTime.utc(2026, 9, 16, 8, 10),
      ),
      confidence: 0.71,
      uncertainty: 0.18,
      assessmentStatus: AssessmentStatus.complete,
      modalities: const [],
      computedAt: DateTime.utc(2026, 9, 16, 8),
      modelVersion: 'ragf-v0.4',
    );

void main() {
  test('loads the latest authoritative assessment for the canonical subject',
      () async {
    final repository = _AssessmentRepository()..value = _assessment();
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: repository,
    );

    await controller.load();

    expect(repository.requestedSubjectId, 'subject-001');
    expect(controller.state.status, AsyncDataStatus.data);
    expect(controller.state.data?.fusionResultId, 123);
  });

  test('null latest assessment becomes unavailable, never empty low risk',
      () async {
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: _AssessmentRepository()..value = null,
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.unavailable);
    expect(controller.state.data, isNull);
  });

  test('unverified latest-assessment transport is explicit unavailable',
      () async {
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: _AssessmentRepository()
        ..failure = const ApiException(kind: ApiFailure.notConfigured),
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.unavailable);
  });

  test('401 expires the clinician session', () async {
    final auth = _AuthRepository();
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: _AssessmentRepository()
        ..failure = const ApiException(kind: ApiFailure.unauthorized),
      authRepository: auth,
    );

    await controller.load();

    expect(auth.expirations, 1);
    expect(controller.state.status, AsyncDataStatus.sessionExpired);
  });

  test('403 is forbidden and does not expire the clinician session', () async {
    final auth = _AuthRepository();
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: _AssessmentRepository()
        ..failure = const ApiException(kind: ApiFailure.forbidden),
      authRepository: auth,
    );

    await controller.load();

    expect(auth.expirations, 0);
    expect(controller.state.status, AsyncDataStatus.forbidden);
  });

  test('409 remains an explicit canonical-state conflict', () async {
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: _AssessmentRepository()
        ..failure = const ApiException(kind: ApiFailure.conflict),
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.conflict);
  });

  test('network failure is offline without fabricating assessment data',
      () async {
    final controller = PatientOverviewController(
      subjectId: 'subject-001',
      repository: _AssessmentRepository()
        ..failure = const ApiException(kind: ApiFailure.offline),
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.offline);
    expect(controller.state.data, isNull);
  });
}
