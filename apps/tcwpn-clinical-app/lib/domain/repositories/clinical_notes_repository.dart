import '../models.dart';

abstract interface class ClinicalNotesRepository {
  Future<List<ClinicalNote>> loadLocalNotes(String localRecordId);

  Future<void> saveLocalNotes(
    String localRecordId,
    List<ClinicalNote> notes,
  );

  Future<List<SupportNote>> effectiveSupportSet(String localRecordId);

  Future<ClinicalNoteIngestResult> submitNote({
    required String subjectId,
    required ClinicalNote note,
    required List<SupportNote> supportSet,
    required int visitCount,
  });
}
