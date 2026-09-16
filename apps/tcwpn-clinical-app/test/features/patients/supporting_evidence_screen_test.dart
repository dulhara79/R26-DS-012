import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/evidence.dart';
import 'package:r26_ds012_app/domain/repositories/evidence_repository.dart';
import 'package:r26_ds012_app/features/patients/supporting_evidence_screen.dart';
import 'package:r26_ds012_app/state/supporting_evidence_controller.dart';

class _FakeEvidenceRepository implements EvidenceRepository {
  EvidenceResult result;
  _FakeEvidenceRepository(this.result);

  @override
  Future<EvidenceResult> ask({
    required String subjectId,
    required String question,
  }) async => result;
}

Future<void> _pump(
  WidgetTester tester,
  SupportingEvidenceController controller,
) async {
  await tester.pumpWidget(
    MaterialApp(
      home: SupportingEvidenceScreen(
        controller: controller,
        displayId: 'Patient A',
      ),
    ),
  );
  await tester.pump();
}

void main() {
  testWidgets('states truthful patient-review context without personalised RAG claim',
      (tester) async {
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: _FakeEvidenceRepository(
        const EvidenceResult(available: true, answer: 'Answer'),
      ),
    );

    await _pump(tester, controller);

    expect(find.text('Supporting Evidence'), findsOneWidget);
    expect(find.textContaining('general evidence support'), findsOneWidget);
    expect(find.textContaining('fusion score'), findsOneWidget);
    expect(find.textContaining('clinical notes'), findsOneWidget);
    expect(find.textContaining('not sent to CARE-AnxRAG'), findsOneWidget);
  });

  testWidgets('renders answer citations and retrieval quality without risk wording',
      (tester) async {
    final repo = _FakeEvidenceRepository(
      const EvidenceResult(
        available: true,
        answer: 'Evidence-grounded response.',
        confidence: 0.81,
        conflictScore: 0.12,
        citations: [
          EvidenceCitation(
            citationId: 'c-1',
            title: 'Guideline title',
            sourceName: 'Guideline source',
            excerpt: 'Relevant excerpt.',
            url: 'https://example.test/guideline',
            evidenceLevel: 'guideline',
          ),
        ],
      ),
    );
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: repo,
    );
    await controller.ask('What is the evidence?');

    await _pump(tester, controller);

    expect(find.text('Evidence-grounded response.'), findsOneWidget);
    expect(find.text('Guideline title'), findsOneWidget);
    expect(find.text('Guideline source'), findsOneWidget);
    expect(find.text('Relevant excerpt.'), findsOneWidget);
    expect(find.textContaining('Evidence retrieval confidence: 0.81'), findsOneWidget);
    expect(find.textContaining('Retrieved-source conflict: 0.12'), findsOneWidget);
    expect(find.textContaining('patient risk probability'), findsNothing);
    expect(find.textContaining('Generated'), findsNothing,
        reason: 'the current evidence wire has no generated-at timestamp');
  });

  testWidgets('renders abstention without turning it into guidance', (tester) async {
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: _FakeEvidenceRepository(
        const EvidenceResult(
          available: true,
          abstained: true,
          abstentionReason: 'Insufficient evidence.',
        ),
      ),
    );
    await controller.ask('Question');

    await _pump(tester, controller);

    expect(find.text('CARE-AnxRAG abstained'), findsOneWidget);
    expect(find.text('Insufficient evidence.'), findsOneWidget);
    expect(find.textContaining('recommended treatment'), findsNothing);
  });

  testWidgets('renders unavailable evidence explicitly', (tester) async {
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: _FakeEvidenceRepository(
        const EvidenceResult(
          available: false,
          error: 'RAG service unavailable',
        ),
      ),
    );
    await controller.ask('Question');

    await _pump(tester, controller);

    expect(find.text('Supporting evidence unavailable'), findsOneWidget);
    expect(find.textContaining('RAG service unavailable'), findsOneWidget);
  });

  testWidgets('labels crisis bypass as Central Backend safety pre-screen',
      (tester) async {
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: _FakeEvidenceRepository(
        const EvidenceResult(
          available: true,
          localCrisisBypass: true,
          safetyMessage: 'Use urgent safety support.',
        ),
      ),
    );
    await controller.ask('Question');

    await _pump(tester, controller);

    expect(find.text('Central Backend safety pre-screen'), findsOneWidget);
    expect(find.text('Use urgent safety support.'), findsOneWidget);
  });
}
