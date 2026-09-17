class PushAttentionMessage {
  const PushAttentionMessage({required this.type, required this.eventId});

  static const String attentionEventType = 'attention_event';

  final String type;
  final String eventId;

  static PushAttentionMessage? tryParse(Map<String, dynamic> data) {
    final type = data['type']?.toString().trim() ?? '';
    final eventId = data['event_id']?.toString().trim() ?? '';
    if (type != attentionEventType || eventId.isEmpty) return null;
    return PushAttentionMessage(type: type, eventId: eventId);
  }
}
