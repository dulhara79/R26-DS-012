import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/contracts/dashboard_snapshot.dart';
import 'package:r26_ds012_app/domain/contracts/patient_summary.dart';
import 'package:r26_ds012_app/domain/repositories/dashboard_repository.dart';
import 'package:r26_ds012_app/features/patients/patients_screen.dart';
import 'package:r26_ds012_app/state/dashboard_controller.dart';

class _DashboardRepository implements DashboardRepository {
  final DashboardSnapshot snapshot;

  _DashboardRepository(this.snapshot);

  @override
  Future<DashboardSnapshot> loadDashboard() async => snapshot;
}

PatientSummary _patient({
  required String subjectId,
  required String displayId,
  required AssessmentStatus status,
  int? fusionResultId,
}) =>
    PatientSummary(
      subjectId: subjectId,
      displayId: displayId,
      fusionResultId: fusionResultId,
      currentAssessment: status == AssessmentStatus.unavailable
          ? null
          : const CurrentAssessment(
              score: 0.58,
              tier: RiskTier.medium,
              band: 'AMBER',
            ),
      forecast: null,
      assessmentStatus: status,
      lastUpdated: DateTime.utc(2026, 9, 18, 12),
      openEventCount: null,
    );

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  testWidgets('Patients renders only the assignment-scoped server roster',
      (tester) async {
    final snapshot = DashboardSnapshot(
      openEvents: const [],
      assignedPatients: [
        _patient(
          subjectId: 'subject-assigned-1',
          displayId: 'Assigned A',
          status: AssessmentStatus.complete,
          fusionResultId: 123,
        ),
        _patient(
          subjectId: 'subject-assigned-2',
          displayId: 'Assigned B',
          status: AssessmentStatus.unavailable,
        ),
      ],
      fetchedAt: DateTime.utc(2026, 9, 18, 12),
    );
    final controller = DashboardController(
      repository: _DashboardRepository(snapshot),
    );
    await controller.load();

    await tester.pumpWidget(
      MaterialApp(
        home: PatientsScreen(controller: controller),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Assigned A'), findsOneWidget);
    expect(find.text('Assigned B'), findsOneWidget);
    expect(find.textContaining('Current multimodal assessment: Medium'), findsOneWidget);
    expect(find.text('Current assessment: unavailable'), findsOneWidget);
    expect(
      find.text('Assessment unavailable — insufficient current data'),
      findsOneWidget,
    );
  });

  testWidgets('tapping assigned patient preserves canonical subject id',
      (tester) async {
    final patient = _patient(
      subjectId: 'canonical-subject-001',
      displayId: 'Assigned A',
      status: AssessmentStatus.complete,
      fusionResultId: 321,
    );
    final controller = DashboardController(
      repository: _DashboardRepository(
        DashboardSnapshot(
          openEvents: const [],
          assignedPatients: [patient],
          fetchedAt: DateTime.utc(2026, 9, 18, 12),
        ),
      ),
    );
    await controller.load();
    PatientSummary? opened;

    await tester.pumpWidget(
      MaterialApp(
        home: PatientsScreen(
          controller: controller,
          onOpenPatient: (value) => opened = value,
        ),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('Assigned A'));
    await tester.pump();

    expect(opened?.subjectId, 'canonical-subject-001');
    expect(opened?.fusionResultId, 321);
  });

  testWidgets('search cannot reveal a patient absent from server assignments',
      (tester) async {
    final controller = DashboardController(
      repository: _DashboardRepository(
        DashboardSnapshot(
          openEvents: const [],
          assignedPatients: [
            _patient(
              subjectId: 'assigned-only',
              displayId: 'Assigned Only',
              status: AssessmentStatus.complete,
              fusionResultId: 7,
            ),
          ],
          fetchedAt: DateTime.utc(2026, 9, 18, 12),
        ),
      ),
    );
    await controller.load();

    await tester.pumpWidget(
      MaterialApp(home: PatientsScreen(controller: controller)),
    );
    await tester.pumpAndSettle();

    await tester.enterText(
      find.byType(TextField),
      'unassigned-guessed-id',
    );
    await tester.pump();

    expect(find.text('Assigned Only'), findsNothing);
    expect(
      find.text('No assigned patients match the current filters.'),
      findsOneWidget,
    );
  });
}
