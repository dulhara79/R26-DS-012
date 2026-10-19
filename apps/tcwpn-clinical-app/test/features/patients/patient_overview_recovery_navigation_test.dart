import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/repositories/assessment_repository.dart';
import 'package:r26_ds012_app/features/patients/patient_overview_screen.dart';
import 'package:r26_ds012_app/state/patient_overview_controller.dart';

class _AssessmentRepository implements AssessmentRepository {
  _AssessmentRepository({this.failure});

  final ApiException? failure;
  int calls = 0;

  @override
  Future<AssessmentSummary?> latestAssessment(String subjectId) async {
    calls++;
    if (failure != null) throw failure!;
    return null;
  }
}

Future<_AssessmentRepository> _showOverview(
  WidgetTester tester, {
  ApiException? failure,
  void Function(String subjectId, String localId)? onNotes,
  void Function(String subjectId)? onTimeline,
}) async {
  final repo = _AssessmentRepository(failure: failure);
  final controller = PatientOverviewController(
    subjectId: 'subject-test',
    repository: repo,
  );
  await controller.load();
  await tester.pumpWidget(
    MaterialApp(
      home: PatientOverviewScreen(
        controller: controller,
        localRecordId: 'local-record-test',
        onOpenClinicalNotes: onNotes,
        onOpenTimeline: onTimeline,
      ),
    ),
  );
  await tester.pump();
  return repo;
}

Future<void> _tapAction(WidgetTester tester, String label) async {
  final button = find.text(label);
  await tester.ensureVisible(button);
  await tester.pump();
  await tester.tap(button);
  await tester.pump();
}

void main() {
  testWidgets('malformed assessment does not block independent clinical notes', (tester) async {
    String? notesSubject;
    String? localRecord;
    String? timelineSubject;
    final repo = await _showOverview(
      tester,
      failure: const ApiException(kind: ApiFailure.malformed),
      onNotes: (subject, local) {
        notesSubject = subject;
        localRecord = local;
      },
      onTimeline: (subject) => timelineSubject = subject,
    );

    expect(find.text('Assessment unavailable'), findsOneWidget);
    expect(find.text('Low'), findsNothing);
    expect(find.text('View clinical notes'), findsOneWidget);
    expect(find.text('View timeline'), findsOneWidget);
    await _tapAction(tester, 'View clinical notes');
    expect(notesSubject, 'subject-test');
    expect(localRecord, 'local-record-test');

    await _tapAction(tester, 'View timeline');
    expect(timelineSubject, 'subject-test');
    expect(repo.calls, 1); // No extra assessment request to open notes.
  });

  testWidgets('new patient with no fusion can access device-local notes', (tester) async {
    String? notesSubject;
    await _showOverview(
      tester,
      onNotes: (subject, _) => notesSubject = subject,
    );
    expect(find.text('View clinical notes'), findsOneWidget);
    expect(find.text('0.00'), findsNothing);
    await _tapAction(tester, 'View clinical notes');
    expect(notesSubject, 'subject-test');
  });

  testWidgets('offline current assessment still allows drafting notes', (tester) async {
    await _showOverview(
      tester,
      failure: const ApiException(kind: ApiFailure.offline),
      onNotes: (_, __) {},
    );
    expect(find.text('Assessment unavailable offline'), findsOneWidget);
    expect(find.text('View clinical notes'), findsOneWidget);
  });

  for (final kind in [ApiFailure.unauthorized, ApiFailure.forbidden, ApiFailure.conflict]) {
    testWidgets('does not expose clinical-note navigation after ${kind.name}', (tester) async {
      await _showOverview(
        tester,
        failure: ApiException(kind: kind),
        onNotes: (_, __) {},
        onTimeline: (_) {},
      );
      expect(find.text('View clinical notes'), findsNothing);
      expect(find.text('View timeline'), findsNothing);
    });
  }
}
