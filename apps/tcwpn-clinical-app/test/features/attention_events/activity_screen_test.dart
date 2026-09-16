import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/attention_event_repository.dart';
import 'package:r26_ds012_app/features/attention_events/activity_screen.dart';
import 'package:r26_ds012_app/state/attention_events_controller.dart';

AttentionEvent _event(String id, AttentionEventStatus status) => AttentionEvent(
      id: id,
      subjectId: 'subject-$id',
      fusionResultId: 123,
      forecastResultId: 'fcst-$id',
      eventType: 'acute_escalation_forecast',
      severity: AttentionSeverity.high,
      reason: 'Forecast crossed policy for $id',
      forecastHorizon: 10,
      status: status,
      createdAt: DateTime.utc(2026, 9, 16, 8),
      acknowledgedAt: status == AttentionEventStatus.open
          ? null
          : DateTime.utc(2026, 9, 16, 8, 2),
      acknowledgedBy: status == AttentionEventStatus.open ? null : 'DR001',
      resolvedAt: status == AttentionEventStatus.resolved
          ? DateTime.utc(2026, 9, 16, 8, 8)
          : null,
      resolvedBy: status == AttentionEventStatus.resolved ? 'DR002' : null,
      policyVersion: 'escalation-v1',
    );

class _Events implements AttentionEventRepository {
  final List<AttentionEvent> events;
  final ApiFailure? failure;

  _Events(this.events, {this.failure});

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async {
    if (failure != null) throw ApiException(kind: failure!);
    return events;
  }

  @override
  Future<List<AttentionEvent>> openEvents() async => events;

  @override
  Future<AttentionEvent?> eventById(String eventId) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent> acknowledge(String eventId) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent> resolve(String eventId) => throw UnimplementedError();
}

Future<AttentionEventsController> _controller(
  List<AttentionEvent> events, {
  ApiFailure? failure,
}) async {
  final controller = AttentionEventsController(
    repository: _Events(events, failure: failure),
  );
  await controller.load();
  return controller;
}

void main() {
  testWidgets('groups OPEN ACKNOWLEDGED RESOLVED and UNKNOWN server events',
      (tester) async {
    final controller = await _controller([
      _event('open-1', AttentionEventStatus.open),
      _event('ack-1', AttentionEventStatus.acknowledged),
      _event('resolved-1', AttentionEventStatus.resolved),
      _event('unknown-1', AttentionEventStatus.unknown),
    ]);

    await tester.pumpWidget(
      MaterialApp(home: ActivityScreen(controller: controller)),
    );

    expect(find.text('Activity'), findsOneWidget);
    expect(find.text('OPEN'), findsWidgets);
    expect(find.text('ACKNOWLEDGED'), findsWidgets);
    expect(find.text('open-1'), findsOneWidget);
    expect(find.text('ack-1'), findsOneWidget);

    await tester.drag(find.byType(ListView), const Offset(0, -800));
    await tester.pumpAndSettle();

    expect(find.text('RESOLVED'), findsWidgets);
    expect(find.text('UNKNOWN'), findsWidgets);
    expect(find.text('resolved-1'), findsOneWidget);
    expect(find.text('unknown-1'), findsOneWidget);
  });

  testWidgets('preserves server order within an activity group', (tester) async {
    final controller = await _controller([
      _event('open-3', AttentionEventStatus.open),
      _event('open-2', AttentionEventStatus.open),
      _event('open-1', AttentionEventStatus.open),
    ]);

    await tester.pumpWidget(
      MaterialApp(home: ActivityScreen(controller: controller)),
    );

    final y3 = tester.getTopLeft(find.text('open-3')).dy;
    final y2 = tester.getTopLeft(find.text('open-2')).dy;
    final y1 = tester.getTopLeft(find.text('open-1')).dy;
    expect(y3, lessThan(y2));
    expect(y2, lessThan(y1));
  });

  testWidgets('opens detail with the same server event identity', (tester) async {
    final event = _event('evt-001', AttentionEventStatus.open);
    final controller = await _controller([event]);
    AttentionEvent? opened;

    await tester.pumpWidget(
      MaterialApp(
        home: ActivityScreen(
          controller: controller,
          onOpenEvent: (value) => opened = value,
        ),
      ),
    );

    await tester.tap(find.text('evt-001'));
    await tester.pump();

    expect(opened, same(event));
    expect(opened!.id, 'evt-001');
  });

  testWidgets('notConfigured is unavailable and never falls back to legacy alerts',
      (tester) async {
    final controller = await _controller(
      const [],
      failure: ApiFailure.notConfigured,
    );

    await tester.pumpWidget(
      MaterialApp(home: ActivityScreen(controller: controller)),
    );

    expect(find.textContaining('Activity unavailable'), findsOneWidget);
    expect(find.textContaining('RED or DARK RED band'), findsNothing);
  });

  testWidgets('empty activity is explicit server event history state',
      (tester) async {
    final controller = await _controller(const []);

    await tester.pumpWidget(
      MaterialApp(home: ActivityScreen(controller: controller)),
    );

    expect(find.textContaining('No server attention-event activity'), findsOneWidget);
  });
}
