class PushAttentionMessage {
  const PushAttentionMessage({required this.type, required this.eventId});

  static const String attentionEventType = 'attention_event';
  static const Set<String> _allowedKeys = <String>{'type', 'event_id'};

  final String type;
  final String eventId;

  static PushAttentionMessage? tryParse(Map<String, dynamic> data) {
    final normalizedKeys = data.keys.map((key) => key.toLowerCase()).toSet();
    if (normalizedKeys.any((key) => !_allowedKeys.contains(key))) return null;

    final type = data['type']?.toString().trim() ?? '';
    final eventId = data['event_id']?.toString().trim() ?? '';
    if (type != attentionEventType || eventId.isEmpty) return null;
    return PushAttentionMessage(type: type, eventId: eventId);
  }
}
