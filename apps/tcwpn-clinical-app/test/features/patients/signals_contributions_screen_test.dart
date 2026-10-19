import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/features/patients/assessment_detail_presentation.dart';
import 'package:r26_ds012_app/features/patients/signals_contributions_screen.dart';

AssessmentSummary _completeAssessment() => AssessmentSummary(
      subjectId: 'subject-001',
      fusionResultId: 123,
      currentAssessment: const CurrentAssessment(
        score: 0.58,
        tier: RiskTier.medium,
        band: 'AMBER',
      ),
      forecast: null,
      confidence: 0.71,
      uncertainty: 0.18,
      assessmentStatus: AssessmentStatus.complete,
      modalities: [
        ModalityStatus(
          componentId: 'c1_physiological',
          score: 0.82,
          available: true,
          includedInFusion: true,
          state: ModalityState.ok,
          confidence: 0.50,
          coverage: 0.50,
          capturedAt: DateTime.utc(2026, 9, 16, 7, 59),
          contribution: 0.24,
        ),
        const ModalityStatus(
          componentId: 'c2_behavioral',
          score: null,
          available: false,
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
      ],
      computedAt: DateTime.utc(2026, 9, 16, 8),
      modelVersion: 'ragf-v0.4',
    );

Future<void> _pump(
  WidgetTester tester,
  AssessmentSummary assessment,
) async {
  await tester.pumpWidget(
    MaterialApp(
      home: SignalsContributionsScreen(
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
  testWidgets('shows canonical assessment provenance', (tester) async {
    await _pump(tester, _completeAssessment());

    expect(find.text('Patient A'), findsOneWidget);
    expect(find.text('subject-001'), findsOneWidget);
    expect(find.text('Fusion result #123'), findsOneWidget);
    expect(find.text('Complete assessment'), findsOneWidget);
    expect(find.text('Model ragf-v0.4'), findsOneWidget);
  });

  testWidgets('renders C1 C2 C3 C4 in canonical order', (tester) async {
    await _pump(tester, _completeAssessment());

    expect(
      p5aComponentOrder,
      const [
        'c1_physiological',
        'c2_behavioral',
        'c3_clinical_nlp',
        'c4_demographic',
      ],
    );

    for (final label in const [
      'Physiological',
      'Behavioural',
      'Clinical NLP / TC-WPN',
      'Contextual',
    ]) {
      await _scrollTo(tester, label);
      expect(find.text(label), findsOneWidget);
    }
  });

  testWidgets('renders exact server scores and contributions', (tester) async {
    await _pump(tester, _completeAssessment());

    expect(find.text('Score: 0.82'), findsOneWidget);
    expect(find.text('Contribution: 0.24'), findsOneWidget);

    await _scrollTo(tester, 'Clinical NLP / TC-WPN');
    expect(find.text('Score: 0.67'), findsOneWidget);
    expect(find.text('Contribution: 0.22'), findsOneWidget);
  });

  testWidgets('missing contribution is Not reported and no weight is invented',
      (tester) async {
    await _pump(tester, _completeAssessment());
    await _scrollTo(tester, 'Behavioural');

    expect(find.text('Contribution: Not reported'), findsOneWidget);
    expect(find.textContaining('weight', findRichText: true), findsNothing);
  });

  testWidgets('C3 remains Clinical NLP signal and never overall risk',
      (tester) async {
    await _pump(tester, _completeAssessment());
    await _scrollTo(tester, 'Clinical NLP / TC-WPN');

    expect(find.text('Clinical NLP / TC-WPN'), findsOneWidget);
    expect(find.text('TC-WPN Risk'), findsNothing);
    expect(find.textContaining('Overall TC-WPN'), findsNothing);
  });

  testWidgets('C2 experimental exclusion is explicit', (tester) async {
    await _pump(tester, _completeAssessment());
    await _scrollTo(tester, 'Behavioural');

    expect(
      find.text('Status: Experimental — not included in fusion'),
      findsOneWidget,
    );
    expect(find.text('Fusion: Not included in fusion'), findsOneWidget);
  });

  testWidgets('missing modality gets an explicit unavailable card',
      (tester) async {
    final complete = _completeAssessment();
    final withoutC4 = AssessmentSummary(
      subjectId: complete.subjectId,
      fusionResultId: complete.fusionResultId,
      currentAssessment: complete.currentAssessment,
      forecast: complete.forecast,
      confidence: complete.confidence,
      uncertainty: complete.uncertainty,
      assessmentStatus: AssessmentStatus.partial,
      modalities: complete.modalities
          .where((m) => m.componentId != 'c4_demographic')
          .toList(),
      computedAt: complete.computedAt,
      modelVersion: complete.modelVersion,
    );

    await _pump(tester, withoutC4);
    await _scrollTo(tester, 'Contextual');

    expect(find.text('Contextual'), findsOneWidget);
    expect(find.text('Score: —'), findsWidgets);
    expect(find.text('Status: Unavailable'), findsWidgets);
    expect(find.text('Contribution: Not reported'), findsWidgets);
  });

  testWidgets('unknown and stale states remain explicit', (tester) async {
    final complete = _completeAssessment();
    final modified = AssessmentSummary(
      subjectId: complete.subjectId,
      fusionResultId: complete.fusionResultId,
      currentAssessment: complete.currentAssessment,
      forecast: complete.forecast,
      confidence: complete.confidence,
      uncertainty: complete.uncertainty,
      assessmentStatus: AssessmentStatus.partial,
      modalities: [
        ModalityStatus(
          componentId: 'c1_physiological',
          score: 0.82,
          available: true,
          includedInFusion: false,
          state: ModalityState.stale,
          confidence: 0.50,
          coverage: 0.50,
          capturedAt: DateTime.utc(2026, 9, 16, 7),
          contribution: null,
        ),
        ...complete.modalities.where(
          (m) =>
              m.componentId != 'c1_physiological' &&
              m.componentId != 'c4_demographic',
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
      ],
      computedAt: complete.computedAt,
      modelVersion: complete.modelVersion,
    );

    await _pump(tester, modified);
    expect(find.text('Status: Stale'), findsOneWidget);
    await _scrollTo(tester, 'Contextual');
    expect(find.text('Status: Status unknown'), findsOneWidget);
  });

  testWidgets('null score never becomes zero', (tester) async {
    await _pump(tester, _completeAssessment());
    await _scrollTo(tester, 'Behavioural');

    expect(find.text('Score: —'), findsOneWidget);
    expect(find.text('Score: 0.00'), findsNothing);
  });
}
