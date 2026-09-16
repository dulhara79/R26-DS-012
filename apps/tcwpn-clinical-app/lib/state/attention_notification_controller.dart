import '../core/notifications/attention_notification_gateway.dart';
import '../data/api/api_client.dart';
import '../data/local/attention_notification_store.dart';
import '../domain/contracts/contract_enums.dart';
import '../domain/repositories/attention_event_repository.dart';

class AttentionNotificationController {
  AttentionNotificationController({
    required this.repository,
    required this.store,
    required this.gateway,
  });

  final AttentionEventRepository repository;
  final AttentionNotificationStore store;
  final AttentionNotificationGateway gateway;

  ApiFailure? lastApiFailure;
  Object? lastDeliveryError;
  bool _polling = false;

  Future<void> pollOnce() async {
    if (_polling) return;
    _polling = true;
    try {
      lastApiFailure = null;
      final events = await repository.openEvents();
      for (final event in events) {
        if (event.id.isEmpty || event.status != AttentionEventStatus.open) continue;
        if (await store.wasDelivered(event.id)) continue;

        try {
          await gateway.showAttentionEvent(event.id);
          await store.markDelivered(event.id);
          lastDeliveryError = null;
        } catch (error) {
          lastDeliveryError = error;
        }
      }
    } on ApiException catch (error) {
      lastApiFailure = error.kind;
    } finally {
      _polling = false;
    }
  }
}
