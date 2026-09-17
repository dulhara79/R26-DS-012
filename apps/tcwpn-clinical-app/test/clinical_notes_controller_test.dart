import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/domain/repositories/clinical_notes_repository.dart';
import 'package:r26_ds012_app/state/clinical_notes_controller.dart';

class _NotesRepository implements ClinicalNotesRepository {
  List<ClinicalNote> stored = [];
  List<SupportNote> support = [];
  ClinicalNoteIngestResult? result;
  Object? failure;
  String? loadedKey;
  String? savedKey;
  String? submittedSubject;

  @override
  Future<List<ClinicalNote>> loadLocalNotes(String localRecordId) async {
    loadedKey = localRecordId;
    return List<ClinicalNote>.from(stored);
  }

  @override
  Future<void> saveLocalNotes(String localRecordId, List<ClinicalNote> notes) async {
    savedKey = localRecordId;
    stored = List<ClinicalNote>.from(notes);
  }

  @override
  Future<List<SupportNote>> effectiveSupportSet(String localRecordId) async => support;

  @override
  Future<ClinicalNoteIngestResult> submitNote({
    required String subjectId,
    required ClinicalNote note,
    required List<SupportNote> supportSet,
    required int visitCount,
  }) async {
    submittedSubject = subjectId;
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

void main() {
  final fixedNow = DateTime.utc(2026, 9, 16, 12);

  ClinicalNotesController controller(
    _NotesRepository repo, {
    Future<void> Function()? refresh,
  }) =>
      ClinicalNotesController(
        subjectId: 'subject-001',
        localRecordId: 'MRN-001',
        clinicianId: 'clinician-01',
        repository: repo,
        refreshCanonicalAssessment: refresh ?? () async {},
        now: () => fixedNow,
        newId: () => 'note-001',
      );

  test('loads and saves using local record id, not backend subject id', () async {
    final repo = _NotesRepository();
    final c = controller(repo);
    await c.load();
    await c.saveDraft(text: 'Patient reports worry.', noteType: 'Psychiatry note');

    expect(repo.loadedKey, 'MRN-001');
    expect(repo.savedKey, 'MRN-001');
    expect(c.notes.single.patientMrn, 'MRN-001');
  });

  test('editing a draft updates it instead of duplicating it', () async {
    final repo = _NotesRepository();
    final c = controller(repo);
    await c.load();
    await c.saveDraft(text: 'First', noteType: 'Psychiatry note');

    await c.updateNote('note-001', text: 'Updated');

    expect(c.notes, hasLength(1));
    expect(c.notes.single.text, 'Updated');
  });

  test('submission uses canonical subject id and successful result refreshes once',
      () async {
    final repo = _NotesRepository();
    var refreshCalls = 0;
    final c = controller(repo, refresh: () async => refreshCalls++);
    await c.load();
    await c.saveDraft(text: 'Patient reports worry.', noteType: 'Psychiatry note');

    await c.submit('note-001');

    expect(repo.submittedSubject, 'subject-001');
    expect(c.notes.single.status, ClinicalNoteStatus.analysed);
    expect(c.notes.single.result?.riskScore, 0.67);
    expect(refreshCalls, 1);
  });

  test('backend response without C3 detail does not fabricate a result', () async {
    final repo = _NotesRepository()
      ..result = const ClinicalNoteIngestResult(
        subjectId: 'subject-001',
        status: 'ok',
        score: 0.67,
        result: null,
      );
    final c = controller(repo);
    await c.load();
    await c.saveDraft(text: 'Patient reports worry.', noteType: 'Psychiatry note');

    await c.submit('note-001');

    expect(c.notes.single.result, isNull);
    expect(c.notes.single.status, ClinicalNoteStatus.analysisFailed);
  });

  test('failed submission preserves note text and marks analysis failed', () async {
    final repo = _NotesRepository()
      ..failure = const ApiException(kind: ApiFailure.offline);
    final c = controller(repo);
    await c.load();
    await c.saveDraft(text: 'Keep this note', noteType: 'Psychiatry note');

    await c.submit('note-001');

    expect(c.notes.single.text, 'Keep this note');
    expect(c.notes.single.status, ClinicalNoteStatus.analysisFailed);
    expect(repo.stored.single.text, 'Keep this note');
  });
}
