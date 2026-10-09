import '../../core/config/env.dart';
import '../../domain/contracts/timeline_entry.dart';
import '../../domain/models.dart';
import '../../domain/repositories/clinical_notes_repository.dart';
import '../../domain/repositories/timeline_repository.dart';
import '../api/api_client.dart';
import '../api/gateways.dart';
import '../local/stores.dart';

class CentralBackendTimelineRepository implements TimelineRepository {
  final ApiClient _api;

  CentralBackendTimelineRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  @override
  Future<List<TimelineEntry>> history(
    String subjectId, {
    int limit = 200,
  }) async {
    final json = await _api.get(
      '/v1/patients/${Uri.encodeComponent(subjectId)}/assessments',
      timeout: Env.quickTimeout,
    );
    final raw = json['assessments'];
    if (raw is! List) {
      throw const ApiException(
        kind: ApiFailure.malformed,
        detail: 'Missing assessments history.',
      );
    }
    final assessments = raw
        .whereType<Map>()
        .take(limit)
        .map(
          (value) => TimelineEntry.fromJson(Map<String, dynamic>.from(value)),
        )
        .toList();
    final rawEvents = json['events'];
    if (rawEvents is List) {
      assessments.addAll(
        rawEvents.whereType<Map>().map(
          (value) => TimelineEntry.fromEvent(Map<String, dynamic>.from(value)),
        ),
      );
    }
    assessments.sort(
      (a, b) => (b.computedAt ?? DateTime.fromMillisecondsSinceEpoch(0))
          .compareTo(a.computedAt ?? DateTime.fromMillisecondsSinceEpoch(0)),
    );
    return assessments;
  }
}

class LocalCentralBackendClinicalNotesRepository
    implements ClinicalNotesRepository {
  @override
  Future<List<ClinicalNote>> loadLocalNotes(String localRecordId) =>
      RecordStore.loadNotes(localRecordId);

  @override
  Future<void> saveLocalNotes(String localRecordId, List<ClinicalNote> notes) =>
      RecordStore.saveNotes(localRecordId, notes);

  @override
  Future<List<SupportNote>> effectiveSupportSet(String localRecordId) =>
      RecordStore.effectiveSupportSet(localRecordId);

  @override
  Future<ClinicalNoteIngestResult> submitNote({
    required String subjectId,
    required ClinicalNote note,
    required List<SupportNote> supportSet,
    required int visitCount,
  }) async {
    final gateway = CentralBackendGateway();
    try {
      return await gateway.submitNote(
        subjectId: subjectId,
        noteText: note.text,
        noteType: note.noteType,
        noteDate: note.recordedAt,
        supportSet: supportSet,
        visitCount: visitCount,
        author: note.clinicianId,
      );
    } finally {
      gateway.dispose();
    }
  }
}
