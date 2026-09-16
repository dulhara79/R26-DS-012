import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/attention_event_repository.dart';
import 'package:r26_ds012_app/domain/repositories/auth_repository.dart';
import 'package:r26_ds012_app/state/async_data_state.dart';
import 'package:r26_ds012_app/state/attention_events_controller.dart';

AttentionEvent _event(String id, AttentionEventStatus status) => AttentionEvent(
      id: id,
      subjectId: 'subject-001',
      fusionResultId: 123,
      forecastResultId: 'fcst-001',
      eventType: 'acute_escalation_forecast',
      severity: AttentionSeverity.high,
      reason: 'Forecast crossed policy',
      forecastHorizon: 10,
      status: status,
      createdAt: DateTime.utc(2026, 9, 16, 8),
      acknowledgedAt: null,
      acknowledgedBy: null,
      resolvedAt: null,
      resolvedBy: null,
      policyVersion: 'escalation-v1',
    );

class _Events implements AttentionEventRepository {
  List<AttentionEvent> value;
  ApiFailure? failure;
  String? lastSubjectId;

  _Events({this.value = const [], this.failure});

  Never _throw(ApiFailure kind) => throw ApiException(kind: kind);

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async {
    lastSubjectId = subjectId;
    if (failure != null) _throw(failure!);
    return value;
  }

  @override
  Future<List<AttentionEvent>> openEvents() async => value;

  @override
  Future<AttentionEvent?> eventById(String eventId) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent> acknowledge(String eventId) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent> resolve(String eventId) => throw UnimplementedError();
}

class _Auth implements AuthRepository {
  int expireCalls = 0;

  @override
  Future<void> expireCurrentSession() async => expireCalls++;

  @override
  Future<void> validateCurrentSession() async {}
}

void main() {
  test('loads server event activity without reordering', () async {
    final repo = _Events(value: [
      _event('evt-003', AttentionEventStatus.resolved),
      _event('evt-002', AttentionEventStatus.acknowledged),
      _event('evt-001', AttentionEventStatus.open),
    ]);
    final controller = AttentionEventsController(repository: repo);

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.data);
    expect(
      controller.state.data!.map((event) => event.id).toList(),
      ['evt-003', 'evt-002', 'evt-001'],
    );
  });

  test('empty server activity becomes empty state', () async {
    final controller = AttentionEventsController(repository: _Events());

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.empty);
  });

  test('401 expires clinician session', () async {
    final auth = _Auth();
    final controller = AttentionEventsController(
      repository: _Events(failure: ApiFailure.unauthorized),
      authRepository: auth,
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.sessionExpired);
    expect(auth.expireCalls, 1);
  });

  test('403 keeps clinician session active', () async {
    final auth = _Auth();
    final controller = AttentionEventsController(
      repository: _Events(failure: ApiFailure.forbidden),
      authRepository: auth,
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.forbidden);
    expect(auth.expireCalls, 0);
  });

  test('notConfigured becomes unavailable', () async {
    final controller = AttentionEventsController(
      repository: _Events(failure: ApiFailure.notConfigured),
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.unavailable);
  });

  test('offline and timeout remain distinguishable from server failure', () async {
    final offline = AttentionEventsController(
      repository: _Events(failure: ApiFailure.offline),
    );
    final timeout = AttentionEventsController(
      repository: _Events(failure: ApiFailure.timeout),
    );
    final server = AttentionEventsController(
      repository: _Events(failure: ApiFailure.server),
    );

    await offline.load();
    await timeout.load();
    await server.load();

    expect(offline.state.status, AsyncDataStatus.offline);
    expect(timeout.state.status, AsyncDataStatus.unavailable);
    expect(server.state.status, AsyncDataStatus.error);
  });

  test('subject-scoped load forwards only the canonical subject id', () async {
    final repo = _Events();
    final controller = AttentionEventsController(
      repository: repo,
      subjectId: 'subject-canonical-001',
    );

    await controller.load();

    expect(repo.lastSubjectId, 'subject-canonical-001');
  });
}
