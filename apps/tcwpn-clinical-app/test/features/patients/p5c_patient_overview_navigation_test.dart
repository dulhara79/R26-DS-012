import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/assessment_repository.dart';
import 'package:r26_ds012_app/features/patients/patient_overview_screen.dart';
import 'package:r26_ds012_app/state/patient_overview_controller.dart';

class _Repository implements AssessmentRepository {
  final AssessmentSummary assessment;
  int calls = 0;
  _Repository(this.assessment);

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    calls++;
    return assessment;
  }
}

AssessmentSummary _assessment() => AssessmentSummary(
      subjectId: 'subject-p5c',
      fusionResultId: 77,
      currentAssessment: const CurrentAssessment(
        score: 0.62,
        tier: RiskTier.medium,
        band: 'AMBER',
      ),
      forecast: null,
      confidence: 0.7,
      uncertainty: 0.3,
      assessmentStatus: AssessmentStatus.complete,
      modalities: const [],
      computedAt: DateTime.utc(2026, 9, 16, 11),
      modelVersion: 'ragf-v0.4',
    );

Future<_Repository> _pump(
  WidgetTester tester, {
  ValueChanged<AssessmentSummary>? onOpenSupportingEvidence,
  ValueChanged<AssessmentSummary>? onOpenClinicianAssessment,
}) async {
  final assessment = _assessment();
  final repository = _Repository(assessment);
  final controller = PatientOverviewController(
    subjectId: assessment.subjectId,
    repository: repository,
  );
  await controller.load();

  await tester.pumpWidget(
    MaterialApp(
      home: PatientOverviewScreen(
        controller: controller,
        displayId: 'Patient P5C',
        onOpenSupportingEvidence: onOpenSupportingEvidence,
        onOpenClinicianAssessment: onOpenClinicianAssessment,
      ),
    ),
  );
  await tester.pump();
  return repository;
}

Future<void> _scrollTo(WidgetTester tester, String label) async {
  await tester.scrollUntilVisible(
    find.text(label),
    250,
    scrollable: find.byType(Scrollable).first,
  );
  await tester.pump();
}

void main() {
  testWidgets('loaded overview exposes both P5C deep-view actions', (tester) async {
    await _pump(tester);

    await _scrollTo(tester, 'View supporting evidence');
    expect(find.text('View supporting evidence'), findsOneWidget);
    await _scrollTo(tester, 'Record clinician assessment');
    expect(find.text('Record clinician assessment'), findsOneWidget);
  });

  testWidgets('both P5C actions receive the exact loaded assessment without refetch',
      (tester) async {
    AssessmentSummary? evidenceAssessment;
    AssessmentSummary? clinicianAssessment;
    final repository = await _pump(
      tester,
      onOpenSupportingEvidence: (value) => evidenceAssessment = value,
      onOpenClinicianAssessment: (value) => clinicianAssessment = value,
    );
    expect(repository.calls, 1);

    await _scrollTo(tester, 'View supporting evidence');
    await tester.tap(find.text('View supporting evidence'));
    await tester.pump();
    expect(evidenceAssessment?.subjectId, 'subject-p5c');
    expect(evidenceAssessment?.fusionResultId, 77);
    expect(repository.calls, 1);

    await _scrollTo(tester, 'Record clinician assessment');
    await tester.tap(find.text('Record clinician assessment'));
    await tester.pump();
    expect(identical(clinicianAssessment, evidenceAssessment), isTrue);
    expect(repository.calls, 1);
  });
}
