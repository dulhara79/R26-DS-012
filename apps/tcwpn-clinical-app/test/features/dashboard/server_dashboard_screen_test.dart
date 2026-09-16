import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/contracts/dashboard_snapshot.dart';
import 'package:r26_ds012_app/domain/contracts/patient_summary.dart';
import 'package:r26_ds012_app/domain/repositories/dashboard_repository.dart';
import 'package:r26_ds012_app/features/dashboard/server_dashboard_screen.dart';
import 'package:r26_ds012_app/state/dashboard_controller.dart';

class _FakeDashboardRepository implements DashboardRepository {
  final DashboardSnapshot snapshot;
  _FakeDashboardRepository(this.snapshot);

  @override
  Future<DashboardSnapshot> loadDashboard() async => snapshot;
}

DashboardSnapshot _snapshot({bool cached = false}) => DashboardSnapshot(
      openEvents: [
        AttentionEvent(
          id: 'evt-001',
          subjectId: 'subject-001',
          fusionResultId: 123,
          forecastResultId: 'fcst-001',
          eventType: 'acute_escalation_forecast',
          severity: AttentionSeverity.high,
          reason: 'Forecast crossed versioned escalation policy',
          forecastHorizon: 10,
          status: AttentionEventStatus.open,
          createdAt: DateTime.utc(2026, 9, 16, 8),
          acknowledgedAt: null,
          acknowledgedBy: null,
          resolvedAt: null,
          resolvedBy: null,
          policyVersion: 'escalation-v1',
        ),
      ],
      assignedPatients: [
        PatientSummary(
          subjectId: 'subject-001',
          displayId: 'Patient A',
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
          assessmentStatus: AssessmentStatus.complete,
          lastUpdated: DateTime.utc(2026, 9, 16, 8),
          openEventCount: 1,
        ),
      ],
      fetchedAt: DateTime.utc(2026, 9, 16, 8),
      isFromCache: cached,
    );

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  testWidgets('renders server attention events before assigned patients',
      (tester) async {
    final controller = DashboardController(
      repository: _FakeDashboardRepository(_snapshot()),
    );
    await controller.load();

    await tester.pumpWidget(
      MaterialApp(home: ServerDashboardScreen(controller: controller)),
    );

    expect(find.text('NEEDS ATTENTION'), findsOneWidget);
    expect(find.text('ASSIGNED PATIENTS'), findsOneWidget);
    expect(find.text('evt-001'), findsOneWidget);
    expect(find.text('Patient A'), findsOneWidget);

    final attentionY = tester.getTopLeft(find.text('NEEDS ATTENTION')).dy;
    final patientsY = tester.getTopLeft(find.text('ASSIGNED PATIENTS')).dy;
    expect(attentionY, lessThan(patientsY));
  });

  testWidgets('keeps current assessment and physiological forecast separate',
      (tester) async {
    final controller = DashboardController(
      repository: _FakeDashboardRepository(_snapshot()),
    );
    await controller.load();

    await tester.pumpWidget(
      MaterialApp(home: ServerDashboardScreen(controller: controller)),
    );

    expect(find.textContaining('Current assessment'), findsWidgets);
    expect(find.textContaining('Medium'), findsWidgets);
    expect(find.textContaining('Near-term physiological forecast'), findsWidgets);
    expect(find.textContaining('High'), findsWidgets);
  });

  testWidgets('unavailable assessment is never presented as low risk',
      (tester) async {
    final unavailable = DashboardSnapshot(
      openEvents: const [],
      assignedPatients: const [
        PatientSummary(
          subjectId: 'subject-002',
          displayId: 'Patient B',
          fusionResultId: null,
          currentAssessment: CurrentAssessment(
            score: null,
            tier: RiskTier.unknown,
            band: null,
          ),
          forecast: null,
          assessmentStatus: AssessmentStatus.unavailable,
          lastUpdated: null,
          openEventCount: 0,
        ),
      ],
      fetchedAt: DateTime.utc(2026, 9, 16, 8),
    );
    final controller = DashboardController(
      repository: _FakeDashboardRepository(unavailable),
    );
    await controller.load();

    await tester.pumpWidget(
      MaterialApp(home: ServerDashboardScreen(controller: controller)),
    );

    expect(find.textContaining('Assessment unavailable'), findsOneWidget);
    expect(find.text('Low'), findsNothing);
    expect(find.text('0'), findsNothing);
  });

  testWidgets('opens Patient Overview with the server canonical subject id',
      (tester) async {
    final controller = DashboardController(
      repository: _FakeDashboardRepository(_snapshot()),
    );
    await controller.load();
    PatientSummary? opened;

    await tester.pumpWidget(
      MaterialApp(
        home: ServerDashboardScreen(
          controller: controller,
          onOpenPatient: (patient) => opened = patient,
        ),
      ),
    );

    await tester.tap(find.text('Patient A'));
    await tester.pump();

    expect(opened, isNotNull);
    expect(opened!.subjectId, 'subject-001');
    expect(opened!.displayId, 'Patient A');
  });
}
