import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/core/design/tokens.dart';
import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/features/dashboard/kpi_dashboard_screen.dart';

ClinicalNote _note({
  required String id,
  required DateTime at,
  required double risk,
  bool stale = false,
}) {
  final result = TcwpnResult(
    prediction: risk >= 0.5 ? 'ANXIETY' : 'NO ANXIETY',
    riskScore: risk,
    calibratedProbability: risk,
    confidence: 0.8,
    entropy: 0.2,
    threshold: 0.5,
    supportK: 3,
  );
  return ClinicalNote(
    id: id,
    patientMrn: 'P_0000000000000001',
    recordedAt: at,
    updatedAt: stale ? at.add(const Duration(minutes: 2)) : null,
    analysedAt: at.add(const Duration(minutes: 1)),
    text: 'note',
    noteType: 'Psychiatry note',
    clinicianId: 'DR001',
    result: result,
    status: ClinicalNoteStatus.analysed,
  );
}

void main() {
  test('latest usable clinical note excludes stale results', () {
    final now = DateTime(2026, 8, 31, 20);
    final notes = [
      _note(id: 'new-stale', at: now, risk: 0.9, stale: true),
      _note(id: 'older-valid', at: now.subtract(const Duration(days: 1)), risk: 0.3),
    ];

    final summary = ClinicalDashboardPatient.fromNotes('P_0000000000000001', notes);

    expect(summary.latestScore, 0.3);
    expect(summary.latestBand, AlertBand.amber);
  });

  test('clinical trajectory compares the last two usable analyses', () {
    final now = DateTime(2026, 8, 31, 20);
    final summary = ClinicalDashboardPatient.fromNotes(
      'P_0000000000000001',
      [
        _note(id: 'latest', at: now, risk: 0.82),
        _note(id: 'previous', at: now.subtract(const Duration(days: 2)), risk: 0.22),
      ],
    );

    expect(summary.trajectory, Trajectory.worsening);
  });

  test('caseload stats use clinical note scores when fusion is unavailable', () {
    final now = DateTime(2026, 8, 31, 20);
    final summaries = <String, ClinicalDashboardPatient>{
      'A': ClinicalDashboardPatient.fromNotes('A', [_note(id: 'a', at: now, risk: 0.12)]),
      'B': ClinicalDashboardPatient.fromNotes('B', [_note(id: 'b', at: now, risk: 0.63)]),
      'C': const ClinicalDashboardPatient(patientMrn: 'C'),
    };

    final stats = ClinicalDashboardStats.fromSummaries(
      patientMrns: const ['A', 'B', 'C'],
      summaries: summaries,
    );

    expect(stats.total, 3);
    expect(stats.scoredCount, 2);
    expect(stats.awaitingAssessment, 1);
    expect(stats.meanRisk, closeTo(0.375, 0.000001));
    expect(stats.countFor(AlertBand.green), 1);
    expect(stats.countFor(AlertBand.red), 1);
    expect(stats.needsReview, 1);
  });
}
