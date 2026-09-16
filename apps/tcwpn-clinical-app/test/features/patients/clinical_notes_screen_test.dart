import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/domain/repositories/clinical_notes_repository.dart';
import 'package:r26_ds012_app/features/patients/clinical_notes_screen.dart';
import 'package:r26_ds012_app/state/clinical_notes_controller.dart';

class _Repository implements ClinicalNotesRepository {
  List<ClinicalNote> notes = [];
  Object? failure;
  ClinicalNoteIngestResult? result;

  @override
  Future<List<ClinicalNote>> loadLocalNotes(String localRecordId) async =>
      List<ClinicalNote>.from(notes);

  @override
  Future<void> saveLocalNotes(String localRecordId, List<ClinicalNote> value) async {
    notes = List<ClinicalNote>.from(value);
  }

  @override
  Future<List<SupportNote>> effectiveSupportSet(String localRecordId) async => const [];

  @override
  Future<ClinicalNoteIngestResult> submitNote({
    required String subjectId,
    required ClinicalNote note,
    required List<SupportNote> supportSet,
    required int visitCount,
  }) async {
    if (failure != null) throw failure!;
    return result ??
        ClinicalNoteIngestResult(
          subjectId: subjectId,
          status: 'ok',
          score: 0.67,
          result: const TcwpnResult(
            prediction: 'ANXIETY',
            riskScore: 0.67,
            calibratedProbability: 0.67,
            confidence: 0.72,
            entropy: 0.4,
            threshold: 0.5,
            supportK: 3,
            calibrationStatus: 'uncalibrated',
          ),
        );
  }
}

Future<ClinicalNotesController> _pump(
  WidgetTester tester,
  _Repository repository,
) async {
  final controller = ClinicalNotesController(
    subjectId: 'subject-001',
    localRecordId: 'MRN-001',
    clinicianId: 'clinician-01',
    repository: repository,
    refreshCanonicalAssessment: () async {},
    now: () => DateTime.utc(2026, 9, 16, 12),
    newId: () => 'note-001',
  );
  await controller.load();
  await tester.pumpWidget(
    MaterialApp(
      home: ClinicalNotesScreen(controller: controller, displayId: 'Patient A'),
    ),
  );
  await tester.pump();
  return controller;
}

void main() {
  testWidgets('labels local note history honestly', (tester) async {
    await _pump(tester, _Repository());

    expect(find.textContaining('device-local'), findsWidgets);
    expect(find.textContaining('server-authoritative note history'), findsOneWidget);
  });

  testWidgets('clinician can create and save a draft locally', (tester) async {
    final controller = await _pump(tester, _Repository());

    await tester.enterText(find.byKey(const Key('clinical-note-text')), 'Patient reports worry.');
    await tester.tap(find.text('Save draft'));
    await tester.pump();

    expect(controller.notes, hasLength(1));
    expect(find.textContaining('Patient reports worry.'), findsOneWidget);
    expect(find.text('Draft'), findsOneWidget);
  });

  testWidgets('clinician can edit an existing local note without duplicating it',
      (tester) async {
    final controller = await _pump(tester, _Repository());
    await controller.saveDraft(text: 'Original note', noteType: 'Psychiatry note');
    await tester.pump();

    await tester.tap(find.text('Edit'));
    await tester.pump();
    await tester.enterText(find.byKey(const Key('clinical-note-text')), 'Updated note');
    await tester.tap(find.text('Update draft'));
    await tester.pump();

    expect(controller.notes, hasLength(1));
    expect(controller.notes.single.text, 'Updated note');
    expect(find.textContaining('Updated note'), findsOneWidget);
    expect(find.textContaining('Original note'), findsNothing);
  });

  testWidgets('failed analysis preserves note and exposes retry', (tester) async {
    final repository = _Repository()
      ..failure = const ApiException(kind: ApiFailure.offline);
    final controller = await _pump(tester, repository);
    await controller.saveDraft(text: 'Keep this note', noteType: 'Psychiatry note');
    await controller.submit('note-001');
    await tester.pump();

    expect(find.textContaining('Keep this note'), findsOneWidget);
    expect(find.text('Retry analysis'), findsOneWidget);
  });

  testWidgets('C3 result is a Clinical NLP signal, never overall risk', (tester) async {
    final controller = await _pump(tester, _Repository());
    await controller.saveDraft(text: 'Analysed note', noteType: 'Psychiatry note');
    await controller.submit('note-001');
    await tester.pump();

    expect(find.text('Clinical NLP / TC-WPN signal'), findsOneWidget);
    expect(find.textContaining('0.67'), findsOneWidget);
    expect(find.textContaining('TC-WPN Risk'), findsNothing);
    expect(find.textContaining('Overall risk'), findsNothing);
    expect(find.textContaining('largest weight'), findsNothing);
  });

  testWidgets('missing component detail is Not reported rather than zero',
      (tester) async {
    final repository = _Repository()
      ..result = const ClinicalNoteIngestResult(
        subjectId: 'subject-001',
        status: 'ok',
        score: 0.67,
        result: null,
      );
    final controller = await _pump(tester, repository);
    await controller.saveDraft(text: 'No detail note', noteType: 'Psychiatry note');
    await controller.submit('note-001');
    await tester.pump();

    expect(find.textContaining('Not reported'), findsOneWidget);
    expect(find.text('0.00'), findsNothing);
  });
}
