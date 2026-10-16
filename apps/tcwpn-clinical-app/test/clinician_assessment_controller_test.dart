import 'dart:async';

import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/clinician_assessment.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/clinician_assessment_repository.dart';
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
  ClinicianAssessmentDraft? seen;
  int calls = 0;
  Object? failure;
  Completer<ClinicianAssessmentReceipt>? completer;

  @override
  Future<ClinicianAssessmentReceipt> submit(ClinicianAssessmentDraft draft) async {
    calls++;
    seen = draft;
    if (failure != null) throw failure!;
    if (completer != null) return completer!.future;
    return ClinicianAssessmentReceipt.fromJson(
      const {
        'verdict_id': 11,
        'subject_id': 'subject-1',
        'agrees_with_model': true,
      },
      request: draft,
    );
  }
}

void main() {
  test('starts without a clinician tier selected', () {
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: 'dr-1',
      repository: _FakeAssessmentRepository(),
    );

    expect(controller.selectedTier, isNull);
    expect(controller.canSubmit, isFalse);
  });

  test('blocks submission when fusion result id is unavailable', () async {
    final repo = _FakeAssessmentRepository();
    final controller = ClinicianAssessmentController(
      assessment: _assessment(fusionResultId: null),
      clinicianId: 'dr-1',
      repository: repo,
    )..selectTier('High');

    await controller.submit();

    expect(repo.calls, 0);
    expect(controller.error, contains('fusion'));
  });

  test('blocks submission when clinician identity is unavailable', () async {
    final repo = _FakeAssessmentRepository();
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: '   ',
      repository: repo,
    )..selectTier('Low');

    await controller.submit();

    expect(repo.calls, 0);
    expect(controller.error, contains('Clinician'));
  });

  test('submits exact fusion id, selected tier, clinician and note', () async {
    final repo = _FakeAssessmentRepository();
    final assessment = _assessment();
    final controller = ClinicianAssessmentController(
      assessment: assessment,
      clinicianId: 'dr-1',
      repository: repo,
    )
      ..selectTier('High')
      ..setNote('  Observation  ');

    await controller.submit();

    expect(repo.seen?.fusionResultId, 37);
    expect(repo.seen?.tierLabel, 'High');
    expect(repo.seen?.author, 'dr-1');
    expect(repo.seen?.note, 'Observation');
    expect(controller.receipt?.verdictId, 11);
    expect(assessment.currentAssessment.tier, RiskTier.medium,
        reason: 'clinician judgement must not mutate model state');
  });

  test('prevents duplicate taps while submitting and after success', () async {
    final repo = _FakeAssessmentRepository()
      ..completer = Completer<ClinicianAssessmentReceipt>();
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: 'dr-1',
      repository: repo,
    )..selectTier('Medium');

    final first = controller.submit();
    final second = controller.submit();
    expect(repo.calls, 1);

    repo.completer!.complete(
      ClinicianAssessmentReceipt.fromJson(
        const {'verdict_id': 12, 'subject_id': 'subject-1'},
        request: repo.seen!,
      ),
    );
    await Future.wait([first, second]);

    await controller.submit();
    expect(repo.calls, 1);
    expect(controller.submitted, isTrue);
  });

  test('failure keeps draft values available for retry', () async {
    final repo = _FakeAssessmentRepository()..failure = Exception('offline');
    final controller = ClinicianAssessmentController(
      assessment: _assessment(),
      clinicianId: 'dr-1',
      repository: repo,
    )
      ..selectTier('Low')
      ..setNote('Keep this text');

    await controller.submit();

    expect(controller.selectedTier, 'Low');
    expect(controller.note, 'Keep this text');
    expect(controller.receipt, isNull);
    expect(controller.error, contains('offline'));
  });
}
