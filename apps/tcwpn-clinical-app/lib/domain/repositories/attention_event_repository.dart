import '../contracts/attention_event.dart';

abstract interface class AttentionEventRepository {
  Future<List<AttentionEvent>> openEvents();
  Future<List<AttentionEvent>> activity({String? subjectId});
  Future<AttentionEvent?> eventById(String eventId);
  Future<AttentionEvent> acknowledge(String eventId);
  Future<AttentionEvent> resolve(String eventId);
}
