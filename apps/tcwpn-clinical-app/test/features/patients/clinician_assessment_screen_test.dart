import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/clinician_assessment.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/clinician_assessment_repository.dart';
import 'package:r26_ds012_app/features/patients/clinician_assessment_screen.dart';
import 'package:r26_ds012_app/state/clinician_assessment_controller.dart';

AssessmentSummary _assessment({int? fusionResultId = 37}) => AssessmentSummary(
      subjectId: 'subject-1',
      fusionResultId: fusionResultId,
      currentAssessment: const CurrentAssessment(
        score: 0.58,
        tier: RiskTier.medium,
        band: 'AMBER',
      ),
      forecast: null,
      confidence: 0.71,
      uncertainty: 0.29,
      assessmentStatus: AssessmentStatus.complete,
      modalities: const [],
      computedAt: DateTime.utc(2026, 9, 16, 10),
      modelVersion: 'ragf-v0.4',
    );

class _FakeAssessmentRepository implements ClinicianAssessmentRepository {
  @override
  Future<ClinicianAssessmentReceipt> submit(ClinicianAssessmentDraft draft) async =>
      ClinicianAssessmentReceipt.fromJson(
        const {
          'verdict_id': 12,
          'subject_id': 'subject-1',
          'agrees_with_model': false,
          'calibration_labels_total': 42,
          'conformal_calibrated': false,
        },
        request: draft,
      );
}

Future<void> _pump(
  WidgetTester tester,
  ClinicianAssessmentController controller,
) async {
  await tester.pumpWidget(
    MaterialApp(
      home: ClinicianAssessmentScreen(
        controller: controller,
        displayId: 'Patient A',
      ),
    ),
  );
  await tester.pump();
}

void main() {
  testWidgets('separates read-only model assessment from clinician judgement',
      (tester) async {
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: 'dr-1',
      repository: _FakeAssessmentRepository(),
    );

    await _pump(tester, controller);

    expect(find.text('CURRENT MODEL ASSESSMENT'), findsOneWidget);
    expect(find.textContaining('Medium · 0.58'), findsOneWidget);
    expect(find.textContaining('Fusion result #37'), findsOneWidget);
    expect(find.textContaining('ragf-v0.4'), findsOneWidget);
    expect(find.text('YOUR CLINICIAN ASSESSMENT'), findsOneWidget);
    expect(find.text('Low'), findsOneWidget);
    expect(find.text('Medium'), findsOneWidget);
    expect(find.text('High'), findsOneWidget);
    expect(controller.selectedTier, isNull,
        reason: 'the model tier must not preselect the clinician label');
    expect(find.textContaining('stored separately from the model'), findsOneWidget);
  });

  testWidgets('save is disabled until a clinician tier is selected', (tester) async {
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: 'dr-1',
      repository: _FakeAssessmentRepository(),
    );

    await _pump(tester, controller);

    final before = tester.widget<FilledButton>(find.byType(FilledButton));
    expect(before.onPressed, isNull);

    controller.selectTier('High');
    await tester.pump();
    final after = tester.widget<FilledButton>(find.byType(FilledButton));
    expect(after.onPressed, isNotNull);
  });

  testWidgets('missing fusion id blocks recording with explicit reason',
      (tester) async {
    final controller = ClinicianAssessmentController(
      assessment: _assessment(fusionResultId: null),
      clinicianId: 'dr-1',
      repository: _FakeAssessmentRepository(),
    )..selectTier('High');

    await _pump(tester, controller);

    expect(find.text('Clinician assessment cannot be recorded'), findsOneWidget);
    expect(find.textContaining('fusion-result identifier'), findsOneWidget);
    expect(tester.widget<FilledButton>(find.byType(FilledButton)).onPressed, isNull);
  });

  testWidgets('missing clinician identity blocks auditable submission',
      (tester) async {
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: '',
      repository: _FakeAssessmentRepository(),
    )..selectTier('Low');

    await _pump(tester, controller);

    expect(find.textContaining('Clinician identity is unavailable'), findsOneWidget);
    expect(tester.widget<FilledButton>(find.byType(FilledButton)).onPressed, isNull);
  });

  testWidgets('success is a server-confirmed receipt and does not invent history',
      (tester) async {
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: 'dr-1',
      repository: _FakeAssessmentRepository(),
    )
      ..selectTier('High')
      ..setNote('Observed during review.');
    await controller.submit();

    await _pump(tester, controller);

    expect(find.text('Clinician assessment recorded'), findsOneWidget);
    expect(find.textContaining('Your assessment: High'), findsOneWidget);
    expect(find.textContaining('Fusion result: #37'), findsOneWidget);
    expect(find.textContaining('Server verdict ID: #12'), findsOneWidget);
    expect(find.textContaining('Model comparison: Different tier'), findsOneWidget);
    expect(find.text('Observed during review.'), findsOneWidget);
    expect(find.textContaining('Created at'), findsNothing);
    expect(find.textContaining('Assessment history'), findsNothing);
  });
}
