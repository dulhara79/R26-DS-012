import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/evidence.dart';
import 'package:r26_ds012_app/domain/repositories/evidence_repository.dart';
import 'package:r26_ds012_app/state/supporting_evidence_controller.dart';

class _FakeEvidenceRepository implements EvidenceRepository {
  String? subjectId;
  String? question;
  EvidenceResult result;
  Object? failure;

  _FakeEvidenceRepository(this.result);

  @override
  Future<EvidenceResult> ask({
    required String subjectId,
    required String question,
  }) async {
    this.subjectId = subjectId;
    this.question = question;
    if (failure != null) throw failure!;
    return result;
  }
}

void main() {
  test('forwards canonical subject and trimmed question', () async {
    final repo = _FakeEvidenceRepository(
      const EvidenceResult(available: true, answer: 'Answer'),
    );
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: repo,
    );

    await controller.ask('  What is the evidence?  ');

    expect(repo.subjectId, 'subject-1');
    expect(repo.question, 'What is the evidence?');
    expect(controller.result?.state, EvidenceState.answered);
    expect(controller.askedQuestion, 'What is the evidence?');
    expect(controller.error, isNull);
  });

  test('blank question is rejected without repository call', () async {
    final repo = _FakeEvidenceRepository(
      const EvidenceResult(available: true, answer: 'Answer'),
    );
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: repo,
    );

    await controller.ask('   ');

    expect(repo.subjectId, isNull);
    expect(controller.result, isNull);
    expect(controller.error, contains('question'));
  });

  test('repository failure becomes explicit unavailable error state', () async {
    final repo = _FakeEvidenceRepository(
      const EvidenceResult(available: true, answer: 'Answer'),
    )..failure = Exception('offline');
    final controller = SupportingEvidenceController(
      subjectId: 'subject-1',
      repository: repo,
    );

    await controller.ask('Evidence?');

    expect(controller.result, isNull);
    expect(controller.error, contains('offline'));
    expect(controller.loading, isFalse);
  });

  test('abstention and crisis bypass remain untouched', () async {
    final abstained = _FakeEvidenceRepository(
      const EvidenceResult(
        available: true,
        abstained: true,
        abstentionReason: 'insufficient evidence',
      ),
    );
    final a = SupportingEvidenceController(subjectId: 'S1', repository: abstained);
    await a.ask('Question');
    expect(a.result?.state, EvidenceState.abstained);

    final crisis = _FakeEvidenceRepository(
      const EvidenceResult(
        available: true,
        localCrisisBypass: true,
        safetyMessage: 'Escalate to urgent support.',
      ),
    );
    final c = SupportingEvidenceController(subjectId: 'S1', repository: crisis);
    await c.ask('Question');
    expect(c.result?.state, EvidenceState.crisisBypass);
  });
}
