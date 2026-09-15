import '../contracts/attention_event.dart';

abstract interface class AttentionEventRepository {
  Future<List<AttentionEvent>> openEvents();
}
