import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:r26_ds012_app/data/local/dashboard_cache.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/contracts/dashboard_snapshot.dart';
import 'package:r26_ds012_app/domain/contracts/patient_summary.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  test('round trip preserves server ids and nullable assessment values', () async {
    const patient = PatientSummary(
      subjectId: 'S1',
      displayId: 'P1',
      fusionResultId: 123,
      currentAssessment: CurrentAssessment(
        score: null,
        tier: RiskTier.unknown,
        band: null,
      ),
      forecast: ForecastResult(
        forecastResultId: 'F1',
        scope: ForecastScope.physiological,
        horizonMinutes: 10,
        score: 0.84,
        tier: RiskTier.high,
        escalationProbability: null,
        escalationPredicted: true,
        generatedAt: null,
        validUntil: null,
      ),
      assessmentStatus: AssessmentStatus.partial,
      lastUpdated: null,
      openEventCount: 1,
    );
    const event = AttentionEvent(
      id: 'E1',
      subjectId: 'S1',
      fusionResultId: 123,
      forecastResultId: 'F1',
      eventType: 'acute_escalation_forecast',
      severity: AttentionSeverity.high,
      reason: null,
      forecastHorizon: 10,
      status: AttentionEventStatus.open,
      createdAt: null,
      acknowledgedAt: null,
      acknowledgedBy: null,
      resolvedAt: null,
      resolvedBy: null,
      policyVersion: 'v1',
    );
    final fetchedAt = DateTime.utc(2026, 9, 16, 1, 2, 3);
    final snapshot = DashboardSnapshot(
      openEvents: const [event],
      assignedPatients: const [patient],
      fetchedAt: fetchedAt,
    );
    const store = DashboardCacheStore();

    await store.save(snapshot);
    final loaded = await store.load();

    expect(loaded, isNotNull);
    expect(loaded!.isFromCache, isTrue);
    expect(loaded.fetchedAt.toUtc(), fetchedAt);
    expect(loaded.openEvents.single.id, 'E1');
    expect(loaded.openEvents.single.fusionResultId, 123);
    expect(loaded.assignedPatients.single.fusionResultId, 123);
    expect(loaded.assignedPatients.single.currentAssessment!.score, isNull);
    expect(loaded.assignedPatients.single.forecast!.forecastResultId, 'F1');
    expect(loaded.assignedPatients.single.forecast!.scope,
        ForecastScope.physiological);
  });
}
