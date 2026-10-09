import 'package:flutter/foundation.dart';

import '../data/api/api_client.dart';
import '../domain/models.dart';
import '../domain/repositories/clinical_notes_repository.dart';

class ClinicalNotesController extends ChangeNotifier {
  final String subjectId;
  final String localRecordId;
  final String clinicianId;
  final ClinicalNotesRepository repository;
  final Future<void> Function() refreshCanonicalAssessment;
  final DateTime Function() now;
  final String Function() newId;

  List<ClinicalNote> _notes = const [];
  bool _loading = false;
  String? _submittingNoteId;
  String? _message;

  ClinicalNotesController({
    required this.subjectId,
    required this.localRecordId,
    required this.clinicianId,
    required this.repository,
    required this.refreshCanonicalAssessment,
    DateTime Function()? now,
    String Function()? newId,
  })  : now = now ?? DateTime.now,
        newId = newId ??
            (() => DateTime.now().microsecondsSinceEpoch.toString());

  List<ClinicalNote> get notes => List.unmodifiable(_notes);
  bool get loading => _loading;
  String? get submittingNoteId => _submittingNoteId;
  String? get message => _message;

  Future<void> load() async {
    _loading = true;
    _message = null;
    notifyListeners();
    try {
      _notes = await repository.loadLocalNotes(localRecordId);
    } catch (_) {
      _message = 'Device-local clinical notes could not be loaded.';
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  ClinicalNote? noteById(String id) {
    for (final note in _notes) {
      if (note.id == id) return note;
    }
    return null;
  }

  Future<ClinicalNote> saveDraft({
    required String text,
    required String noteType,
  }) async {
    final note = ClinicalNote(
      id: newId(),
      patientMrn: localRecordId,
      recordedAt: now(),
      text: text.trim(),
      noteType: noteType.trim().isEmpty ? 'Psychiatry note' : noteType.trim(),
      clinicianId: clinicianId,
      status: ClinicalNoteStatus.draft,
    );
    _notes = [note, ..._notes];
    await _persist();
    return note;
  }

  Future<ClinicalNote?> updateNote(
    String noteId, {
    String? text,
    String? noteType,
  }) async {
    final existing = noteById(noteId);
    if (existing == null) return null;
    final updated = existing.copyWith(
      text: text?.trim(),
      noteType: noteType?.trim(),
      updatedAt: now(),
      clearResult: true,
      clearAnalysisError: true,
      status: ClinicalNoteStatus.draft,
    );
    _notes = _notes.map((n) => n.id == noteId ? updated : n).toList();
    await _persist();
    return updated;
  }

  Future<void> submit(String noteId) async {
    final note = noteById(noteId);
    if (note == null || _submittingNoteId != null) return;

    _submittingNoteId = noteId;
    _message = null;
    notifyListeners();

    try {
      final support = await repository.effectiveSupportSet(localRecordId);
      final ingest = await repository.submitNote(
        subjectId: subjectId,
        note: note,
        supportSet: support,
        visitCount: _notes.length,
      );
      final analysed = note.copyWith(
        result: ingest.result,
        clearResult: ingest.result == null,
        analysedAt: ingest.result == null ? null : now(),
        status: ingest.result == null
            ? ClinicalNoteStatus.analysisFailed
            : ClinicalNoteStatus.analysed,
        clearAnalysisError: ingest.result != null,
        lastAnalysisError: ingest.result == null
            ? 'The backend accepted the note but did not report Clinical NLP component detail.'
            : null,
      );
      _notes = _notes.map((n) => n.id == noteId ? analysed : n).toList();
      await _persist(notify: false);
      await refreshCanonicalAssessment();
    } on ApiException catch (error) {
      await _markFailed(noteId, error.message);
    } catch (_) {
      await _markFailed(noteId, 'Analysis could not be completed. The note remains saved on this device.');
    } finally {
      _submittingNoteId = null;
      notifyListeners();
    }
  }

  Future<void> _markFailed(String noteId, String reason) async {
    final current = noteById(noteId);
    if (current == null) return;
    final failed = current.copyWith(
      status: ClinicalNoteStatus.analysisFailed,
      lastAnalysisError: reason,
    );
    _notes = _notes.map((n) => n.id == noteId ? failed : n).toList();
    _message = reason;
    await _persist(notify: false);
  }

  Future<void> _persist({bool notify = true}) async {
    await repository.saveLocalNotes(localRecordId, _notes);
    if (notify) notifyListeners();
  }
}
