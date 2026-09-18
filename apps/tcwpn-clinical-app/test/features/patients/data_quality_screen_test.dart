import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/features/patients/data_quality_screen.dart';

AssessmentSummary _assessment({
  AssessmentStatus status = AssessmentStatus.complete,
  List<ModalityStatus>? modalities,
  int? fusionResultId = 123,
  DateTime? computedAt,
  String? modelVersion = 'ragf-v0.4',
}) =>
    AssessmentSummary(
      subjectId: 'subject-001',
      fusionResultId: fusionResultId,
      currentAssessment: const CurrentAssessment(
        score: 0.58,
        tier: RiskTier.medium,
        band: 'AMBER',
      ),
      forecast: null,
      confidence: 0.71,
      uncertainty: 0.18,
      assessmentStatus: status,
      modalities: modalities ?? _completeModalities(),
      computedAt: computedAt ?? DateTime.utc(2026, 9, 16, 8),
      modelVersion: modelVersion,
    );

List<ModalityStatus> _completeModalities() => [
      ModalityStatus(
        componentId: 'c1_physiological',
        score: 0.82,
        available: true,
        includedInFusion: true,
        state: ModalityState.ok,
        confidence: 0.50,
        coverage: 0.75,
        capturedAt: DateTime.utc(2026, 9, 16, 7, 59),
        contribution: 0.24,
      ),
      const ModalityStatus(
        componentId: 'c2_behavioral',
        score: 0.40,
        available: true,
        includedInFusion: false,
        state: ModalityState.notValidated,
        confidence: null,
        coverage: null,
        capturedAt: null,
        contribution: null,
      ),
      ModalityStatus(
        componentId: 'c3_clinical_nlp',
        score: 0.67,
        available: true,
        includedInFusion: true,
        state: ModalityState.ok,
        confidence: 0.61,
        coverage: 1.0,
        capturedAt: DateTime.utc(2026, 9, 16, 7, 30),
        contribution: 0.22,
      ),
      ModalityStatus(
        componentId: 'c4_demographic',
        score: 0.43,
        available: true,
        includedInFusion: true,
        state: ModalityState.ok,
        confidence: 0.70,
        coverage: 1.0,
        capturedAt: DateTime.utc(2026, 9, 1, 8),
        contribution: 0.12,
      ),
    ];

Future<void> _pump(
  WidgetTester tester,
  AssessmentSummary assessment,
) async {
  await tester.pumpWidget(
    MaterialApp(
      home: DataQualityScreen(
        assessment: assessment,
        displayId: 'Patient A',
      ),
    ),
  );
  await tester.pump();
}

Future<void> _scrollTo(WidgetTester tester, String text) async {
  final target = find.text(text);
  final scrollable = find.byType(Scrollable).first;

  for (var i = 0; i < 10 && target.evaluate().isEmpty; i++) {
    await tester.drag(scrollable, const Offset(0, -250));
    await tester.pump();
  }

  expect(target, findsOneWidget);
  await tester.ensureVisible(target);
  await tester.pump();
}

void main() {
  testWidgets('shows assessment status exactly as supplied', (tester) async {
    final cases = <AssessmentStatus, String>{
      AssessmentStatus.complete: 'Assessment: Complete assessment',
      AssessmentStatus.partial: 'Assessment: Partial assessment',
      AssessmentStatus.unavailable: 'Assessment: Assessment unavailable',
      AssessmentStatus.unknown: 'Assessment: Assessment status unknown',
    };

    for (final entry in cases.entries) {
      await _pump(tester, _assessment(status: entry.key));
      expect(find.text(entry.value), findsOneWidget);
    }
  });

  testWidgets('shows availability and fusion inclusion as separate fields',
      (tester) async {
    await _pump(tester, _assessment());
    await _scrollTo(tester, 'Behavioural');

    expect(find.text('Availability: Available'), findsWidgets);
    expect(find.text('Status: Experimental — not included in fusion'),
        findsOneWidget);
    expect(find.text('Fusion inclusion: Not included in fusion'), findsOneWidget);
  });

  testWidgets('shows exact confidence coverage and captured time when reported',
      (tester) async {
    await _pump(tester, _assessment());
    await _scrollTo(tester, 'Physiological');

    expect(find.text('Confidence: 0.50'), findsOneWidget);
    expect(find.text('Coverage: 0.75'), findsOneWidget);
    expect(find.textContaining('Captured:'), findsWidgets);
    expect(find.text('Score: 0.82'), findsOneWidget);
  });

  testWidgets('missing values are Not reported and never zero', (tester) async {
    final modalities = _completeModalities();
    modalities[2] = const ModalityStatus(
      componentId: 'c3_clinical_nlp',
      score: null,
      available: false,
      includedInFusion: false,
      state: ModalityState.unavailable,
      confidence: null,
      coverage: null,
      capturedAt: null,
      contribution: null,
    );

    await _pump(tester, _assessment(modalities: modalities));
    await _scrollTo(tester, 'Clinical NLP / TC-WPN');

    expect(find.text('Score: —'), findsOneWidget);
    expect(find.text('Confidence: Not reported'), findsWidgets);
    expect(find.text('Coverage: Not reported'), findsWidgets);
    expect(find.text('Captured: Not reported'), findsWidgets);
    expect(find.text('Reason detail: Not reported by backend'), findsWidgets);
    expect(find.text('Score: 0.00'), findsNothing);
    expect(find.text('Confidence: 0.00'), findsNothing);
  });

  testWidgets('stale unavailable error and unknown remain distinct',
      (tester) async {
    final modalities = <ModalityStatus>[
      const ModalityStatus(
        componentId: 'c1_physiological',
        score: 0.82,
        available: true,
        includedInFusion: false,
        state: ModalityState.stale,
        confidence: null,
        coverage: null,
        capturedAt: null,
        contribution: null,
      ),
      const ModalityStatus(
        componentId: 'c2_behavioral',
        score: null,
        available: false,
        includedInFusion: false,
        state: ModalityState.unavailable,
        confidence: null,
        coverage: null,
        capturedAt: null,
        contribution: null,
      ),
      const ModalityStatus(
        componentId: 'c3_clinical_nlp',
        score: null,
        available: false,
        includedInFusion: false,
        state: ModalityState.error,
        confidence: null,
        coverage: null,
        capturedAt: null,
        contribution: null,
      ),
      const ModalityStatus(
        componentId: 'c4_demographic',
        score: null,
        available: null,
        includedInFusion: null,
        state: ModalityState.unknown,
        confidence: null,
        coverage: null,
        capturedAt: null,
        contribution: null,
      ),
    ];

    await _pump(tester, _assessment(status: AssessmentStatus.partial, modalities: modalities));

    expect(find.text('Status: Stale'), findsOneWidget);
    await _scrollTo(tester, 'Behavioural');
    expect(find.text('Status: Unavailable'), findsOneWidget);
    await _scrollTo(tester, 'Clinical NLP / TC-WPN');
    expect(find.text('Status: Service error'), findsOneWidget);
    await _scrollTo(tester, 'Contextual');
    expect(find.text('Status: Status unknown'), findsOneWidget);
  });

  testWidgets('C2 experimental exclusion stays explicit', (tester) async {
    await _pump(tester, _assessment());
    await _scrollTo(tester, 'Behavioural');

    expect(find.text('Status: Experimental — not included in fusion'),
        findsOneWidget);
    expect(find.text('Fusion inclusion: Not included in fusion'), findsOneWidget);
  });

  testWidgets('missing modality record remains visibly unavailable',
      (tester) async {
    final modalities = _completeModalities()
        .where((m) => m.componentId != 'c4_demographic')
        .toList();

    await _pump(tester, _assessment(status: AssessmentStatus.partial, modalities: modalities));
    await _scrollTo(tester, 'Contextual');

    expect(find.text('Contextual'), findsOneWidget);
    expect(find.text('Availability: Unavailable'), findsWidgets);
    expect(find.text('Status: Unavailable'), findsWidgets);
    expect(find.text('Fusion inclusion: Inclusion not reported'), findsOneWidget);
  });

  testWidgets('reason detail is not fabricated', (tester) async {
    await _pump(tester, _assessment());
    await _scrollTo(tester, 'Behavioural');

    expect(find.text('Reason detail: Not reported by backend'), findsWidgets);
    expect(find.textContaining('buffering'), findsNothing);
    expect(find.textContaining('insufficient data'), findsNothing);
  });
}
