abstract interface class AttentionPushService {
  Future<void> initialize();
  Future<void> activateAuthenticatedSession();
  Future<void> revoke();
  Future<String?> takeInitialOpenedEventId();
  Stream<String> get foregroundEventIds;
  Stream<String> get openedEventIds;
}

class DisabledAttentionPushService implements AttentionPushService {
  const DisabledAttentionPushService();

  @override
  Future<void> initialize() async {}

  @override
  Future<void> activateAuthenticatedSession() async {}

  @override
  Future<void> revoke() async {}

  @override
  Future<String?> takeInitialOpenedEventId() async => null;

  @override
  Stream<String> get foregroundEventIds => const Stream<String>.empty();

  @override
  Stream<String> get openedEventIds => const Stream<String>.empty();
}
