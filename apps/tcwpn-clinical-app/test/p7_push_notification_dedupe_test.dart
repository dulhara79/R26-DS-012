import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/core/notifications/attention_notification_gateway.dart';
import 'package:r26_ds012_app/data/local/attention_notification_store.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/attention_event_repository.dart';
import 'package:r26_ds012_app/state/attention_notification_controller.dart';

void main() {
  test('foreground push then polling produces one local notification', () async {
    final store = _Store();
    final gateway = _Gateway();
    final controller = AttentionNotificationController(
      repository: _Repository([_event('evt-1')]),
      store: store,
      gateway: gateway,
    );

    await controller.deliverEventId('evt-1');
    await controller.pollOnce();

    expect(gateway.shown, ['evt-1']);
    expect(store.delivered, {'evt-1'});
  });

  test('opened remote event is marked delivered without local alert replay', () async {
    final store = _Store();
    final gateway = _Gateway();
    final controller = AttentionNotificationController(
      repository: _Repository([_event('evt-2')]),
      store: store,
      gateway: gateway,
    );

    await controller.markOpenedEventDelivered('evt-2');
    await controller.pollOnce();

    expect(gateway.shown, isEmpty);
    expect(store.delivered, {'evt-2'});
  });
}

AttentionEvent _event(String id) => AttentionEvent(
      id: id,
      subjectId: 'subject-1',
      fusionResultId: 1,
      forecastResultId: 'forecast-1',
      eventType: 'acute_escalation_forecast',
      severity: AttentionSeverity.high,
      reason: null,
      forecastHorizon: 10,
      status: AttentionEventStatus.open,
      createdAt: DateTime.utc(2026, 9, 17),
      acknowledgedAt: null,
      acknowledgedBy: null,
      resolvedAt: null,
      resolvedBy: null,
      policyVersion: 'policy-v1',
    );

class _Repository implements AttentionEventRepository {
  _Repository(this.events);
  final List<AttentionEvent> events;

  @override
  Future<List<AttentionEvent>> openEvents() async => events;

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async => events;

  @override
  Future<AttentionEvent?> eventById(String eventId) async => null;

  @override
  Future<AttentionEvent> acknowledge(String eventId) => throw UnimplementedError();

  @override
  Future<AttentionEvent> resolve(String eventId) => throw UnimplementedError();
}

class _Store implements AttentionNotificationStore {
  final Set<String> delivered = {};
  String? pending;

  @override
  Future<bool> wasDelivered(String eventId) async => delivered.contains(eventId);

  @override
  Future<void> markDelivered(String eventId) async => delivered.add(eventId);

  @override
  Future<void> savePendingOpen(String eventId) async => pending = eventId;

  @override
  Future<String?> takePendingOpen() async {
    final value = pending;
    pending = null;
    return value;
  }
}

class _Gateway implements AttentionNotificationGateway {
  final List<String> shown = [];

  @override
  Future<void> initialize() async {}

  @override
  Future<bool> requestPermission() async => true;

  @override
  Future<void> showAttentionEvent(String eventId) async => shown.add(eventId);

  @override
  Stream<String> get openedEventIds => const Stream.empty();
}
