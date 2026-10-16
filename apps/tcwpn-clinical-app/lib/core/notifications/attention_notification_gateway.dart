import 'dart:async';

abstract interface class AttentionNotificationGateway {
  Future<void> initialize();
  Future<bool> requestPermission();
  Future<void> showAttentionEvent(String eventId);
  Stream<String> get openedEventIds;
}
