import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/assessment_repository.dart';
import 'package:r26_ds012_app/features/patients/patient_overview_screen.dart';
import 'package:r26_ds012_app/state/patient_overview_controller.dart';

class _Repository implements AssessmentRepository {
  final AssessmentSummary? assessment;
  int latestAssessmentCalls = 0;

  _Repository(this.assessment);

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    latestAssessmentCalls++;
    return assessment;
  }
}

class _OverviewHarness {
  final _Repository repository;
  final PatientOverviewController controller;

  const _OverviewHarness({
    required this.repository,
    required this.controller,
  });
}

AssessmentSummary _completeAssessment() => AssessmentSummary(
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

Future<_OverviewHarness> _pumpOverview(
  WidgetTester tester,
  AssessmentSummary? assessment, {
  ValueChanged<AssessmentSummary>? onOpenSignalsContributions,
  ValueChanged<AssessmentSummary>? onOpenDataQuality,
}) async {
  final repository = _Repository(assessment);
  final controller = PatientOverviewController(
    subjectId: 'subject-001',
    repository: repository,
  );
  await controller.load();

  await tester.pumpWidget(
    MaterialApp(
      home: PatientOverviewScreen(
        displayId: 'Patient A',
        controller: controller,
        onOpenSignalsContributions: onOpenSignalsContributions,
        onOpenDataQuality: onOpenDataQuality,
      ),
    ),
  );
  await tester.pump();

  return _OverviewHarness(repository: repository, controller: controller);
}

Future<void> _scrollTo(WidgetTester tester, String text) async {
  await tester.scrollUntilVisible(
    find.text(text),
    250,
    scrollable: find.byType(Scrollable).first,
  );
  await tester.pump();
}

void main() {
  testWidgets('puts forecast before current multimodal assessment',
      (tester) async {
    await _pumpOverview(tester, _completeAssessment());

    expect(find.text('ACUTE ESCALATION FORECAST'), findsOneWidget);
    expect(find.text('CURRENT MULTIMODAL ASSESSMENT'), findsOneWidget);

    final forecastY =
        tester.getTopLeft(find.text('ACUTE ESCALATION FORECAST')).dy;
    final currentY =
        tester.getTopLeft(find.text('CURRENT MULTIMODAL ASSESSMENT')).dy;
    expect(forecastY, lessThan(currentY));
  });

  testWidgets('labels physiological forecast without claiming multimodal scope',
      (tester) async {
    await _pumpOverview(tester, _completeAssessment());

    expect(find.text('Physiological forecast'), findsOneWidget);
    expect(find.textContaining('10-minute horizon'), findsOneWidget);
    expect(find.text('High · 0.84'), findsOneWidget);
    expect(find.textContaining('Multimodal forecast'), findsNothing);
  });

  testWidgets('shows server current assessment and separate uncertainty fields',
      (tester) async {
    await _pumpOverview(tester, _completeAssessment());

    expect(find.text('Medium · 0.58'), findsOneWidget);
    expect(find.text('Complete assessment'), findsOneWidget);
    expect(find.text('Confidence 0.71'), findsOneWidget);
    expect(find.text('Uncertainty 0.18'), findsOneWidget);
    expect(find.text('Fusion result #123'), findsOneWidget);
    expect(find.text('Model ragf-v0.4'), findsOneWidget);
  });

  testWidgets('shows all four signals with C3 subordinate to fusion',
      (tester) async {
    await _pumpOverview(tester, _completeAssessment());

    await _scrollTo(tester, 'Clinical NLP / TC-WPN');

    expect(find.text('Physiological'), findsOneWidget);
    expect(find.text('Behavioural'), findsOneWidget);
    expect(find.text('Clinical NLP / TC-WPN'), findsOneWidget);
    expect(find.text('Contextual'), findsOneWidget);
    expect(find.text('Experimental — not included in fusion'), findsOneWidget);
    expect(find.text('TC-WPN Risk'), findsNothing);
    expect(find.text('Overall TC-WPN risk'), findsNothing);
  });

  testWidgets('stale physiological signal stays stale', (tester) async {
    final complete = _completeAssessment();
    final stale = AssessmentSummary(
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
        ...complete.modalities.skip(1),
      ],
      computedAt: complete.computedAt,
      modelVersion: complete.modelVersion,
    );

    await _pumpOverview(tester, stale);
    await _scrollTo(tester, 'Physiological');

    expect(find.text('Stale'), findsOneWidget);
    expect(find.text('Partial assessment'), findsOneWidget);
  });

  testWidgets('unavailable C3 is not rendered as zero', (tester) async {
    final complete = _completeAssessment();
    final c3Unavailable = AssessmentSummary(
      subjectId: complete.subjectId,
      fusionResultId: complete.fusionResultId,
      currentAssessment: complete.currentAssessment,
      forecast: complete.forecast,
      confidence: complete.confidence,
      uncertainty: complete.uncertainty,
      assessmentStatus: AssessmentStatus.partial,
      modalities: complete.modalities
          .map((m) => m.componentId == 'c3_clinical_nlp'
              ? const ModalityStatus(
                  componentId: 'c3_clinical_nlp',
                  score: null,
                  available: false,
                  includedInFusion: false,
                  state: ModalityState.unavailable,
                  confidence: null,
                  coverage: null,
                  capturedAt: null,
                  contribution: null,
                )
              : m)
          .toList(),
      computedAt: complete.computedAt,
      modelVersion: complete.modelVersion,
    );

    await _pumpOverview(tester, c3Unavailable);
    await _scrollTo(tester, 'Clinical NLP / TC-WPN');

    expect(find.text('Unavailable'), findsWidgets);
    expect(find.text('0.00'), findsNothing);
    expect(find.text('0'), findsNothing);
  });

  testWidgets('missing assessment is unavailable and never low', (tester) async {
    await _pumpOverview(tester, null);

    expect(find.text('Assessment unavailable'), findsOneWidget);
    expect(find.text('Low'), findsNothing);
    expect(find.text('0.00'), findsNothing);
  });

  testWidgets('shows both P5A deep-view actions for a loaded assessment',
      (tester) async {
    await _pumpOverview(tester, _completeAssessment());

    await _scrollTo(tester, 'View signals & contributions');
    expect(find.text('View signals & contributions'), findsOneWidget);

    await _scrollTo(tester, 'View data quality');
    expect(find.text('View data quality'), findsOneWidget);
  });

  testWidgets('signals action receives the exact loaded AssessmentSummary object',
      (tester) async {
    final assessment = _completeAssessment();
    AssessmentSummary? received;

    final harness = await _pumpOverview(
      tester,
      assessment,
      onOpenSignalsContributions: (value) => received = value,
    );

    await _scrollTo(tester, 'View signals & contributions');
    await tester.tap(find.text('View signals & contributions'));
    await tester.pump();

    expect(identical(received, assessment), isTrue);
    expect(harness.repository.latestAssessmentCalls, 1);
  });

  testWidgets('data quality action receives the exact loaded AssessmentSummary object',
      (tester) async {
    final assessment = _completeAssessment();
    AssessmentSummary? received;

    final harness = await _pumpOverview(
      tester,
      assessment,
      onOpenDataQuality: (value) => received = value,
    );

    await _scrollTo(tester, 'View data quality');
    await tester.tap(find.text('View data quality'));
    await tester.pump();

    expect(identical(received, assessment), isTrue);
    expect(harness.repository.latestAssessmentCalls, 1);
  });

  testWidgets('opening a P5A deep view does not fetch a second assessment',
      (tester) async {
    final assessment = _completeAssessment();
    final harness = await _pumpOverview(
      tester,
      assessment,
      onOpenSignalsContributions: (_) {},
      onOpenDataQuality: (_) {},
    );

    expect(harness.repository.latestAssessmentCalls, 1);

    await _scrollTo(tester, 'View signals & contributions');
    await tester.tap(find.text('View signals & contributions'));
    await tester.pump();
    expect(harness.repository.latestAssessmentCalls, 1);

    await _scrollTo(tester, 'View data quality');
    await tester.tap(find.text('View data quality'));
    await tester.pump();
    expect(harness.repository.latestAssessmentCalls, 1);
  });
}
