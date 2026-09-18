import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/core/notifications/attention_notification_gateway.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/local/attention_notification_store.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/attention_event_repository.dart';
import 'package:r26_ds012_app/state/attention_notification_controller.dart';

void main() {
  test('new OPEN event produces one notification and is marked delivered', () async {
    final repository = _Repository([_event('evt-1')]);
    final store = _Store();
    final gateway = _Gateway();
    final controller = AttentionNotificationController(
      repository: repository,
      store: store,
      gateway: gateway,
    );

    await controller.pollOnce();
    await controller.pollOnce();

    expect(gateway.shown, ['evt-1']);
    expect(store.delivered, {'evt-1'});
  });

  test('different event ids each produce one notification', () async {
    final gateway = _Gateway();
    final controller = AttentionNotificationController(
      repository: _Repository([_event('evt-1'), _event('evt-2')]),
      store: _Store(),
      gateway: gateway,
    );

    await controller.pollOnce();

    expect(gateway.shown, ['evt-1', 'evt-2']);
  });

  test('non-open event is never converted into a notification', () async {
    final gateway = _Gateway();
    final controller = AttentionNotificationController(
      repository: _Repository([
        _event('evt-ack', status: AttentionEventStatus.acknowledged),
        _event('evt-resolved', status: AttentionEventStatus.resolved),
      ]),
      store: _Store(),
      gateway: gateway,
    );

    await controller.pollOnce();

    expect(gateway.shown, isEmpty);
  });

  test('API failure creates no local event or notification', () async {
    final gateway = _Gateway();
    final controller = AttentionNotificationController(
      repository: _Repository.failure(
        const ApiException(kind: ApiFailure.offline),
      ),
      store: _Store(),
      gateway: gateway,
    );

    await controller.pollOnce();

    expect(gateway.shown, isEmpty);
    expect(controller.lastApiFailure, ApiFailure.offline);
  });

  test('failed OS delivery is not marked delivered so a later poll can retry', () async {
    final store = _Store();
    final gateway = _Gateway(failuresRemaining: 1);
    final controller = AttentionNotificationController(
      repository: _Repository([_event('evt-1')]),
      store: store,
      gateway: gateway,
    );

    await controller.pollOnce();
    expect(store.delivered, isEmpty);

    await controller.pollOnce();
    expect(gateway.shown, ['evt-1']);
    expect(store.delivered, {'evt-1'});
  });
}

AttentionEvent _event(
  String id, {
  AttentionEventStatus status = AttentionEventStatus.open,
}) =>
    AttentionEvent(
      id: id,
      subjectId: 'subject-1',
      fusionResultId: 7,
      forecastResultId: 'forecast-1',
      eventType: 'acute_escalation_forecast',
      severity: AttentionSeverity.high,
      reason: null,
      forecastHorizon: 10,
      status: status,
      createdAt: DateTime.utc(2026, 9, 16),
      acknowledgedAt: null,
      acknowledgedBy: null,
      resolvedAt: null,
      resolvedBy: null,
      policyVersion: 'policy-v1',
    );

class _Repository implements AttentionEventRepository {
  _Repository(this.events) : failure = null;
  _Repository.failure(this.failure) : events = const [];

  final List<AttentionEvent> events;
  final ApiException? failure;

  @override
  Future<List<AttentionEvent>> openEvents() async {
    if (failure != null) throw failure!;
    return events;
  }

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async => events;

  @override
  Future<AttentionEvent?> eventById(String eventId) async {
    for (final event in events) {
      if (event.id == eventId) return event;
    }
    return null;
  }

  @override
  Future<AttentionEvent> acknowledge(String eventId) => throw UnimplementedError();

  @override
  Future<AttentionEvent> resolve(String eventId) => throw UnimplementedError();
}

class _Store implements AttentionNotificationStore {
  final Set<String> delivered = <String>{};
  String? pending;

  @override
  Future<void> markDelivered(String eventId) async => delivered.add(eventId);

  @override
  Future<bool> wasDelivered(String eventId) async => delivered.contains(eventId);

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
  _Gateway({this.failuresRemaining = 0});

  int failuresRemaining;
  final List<String> shown = <String>[];

  @override
  Stream<String> get openedEventIds => const Stream<String>.empty();

  @override
  Future<void> initialize() async {}

  @override
  Future<bool> requestPermission() async => true;

  @override
  Future<void> showAttentionEvent(String eventId) async {
    if (failuresRemaining > 0) {
      failuresRemaining--;
      throw StateError('notification unavailable');
    }
    shown.add(eventId);
  }
}
