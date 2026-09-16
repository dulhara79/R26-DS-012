import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/repositories/attention_event_repository.dart';
import 'package:r26_ds012_app/domain/repositories/auth_repository.dart';
import 'package:r26_ds012_app/state/async_data_state.dart';
import 'package:r26_ds012_app/state/attention_event_detail_controller.dart';

AttentionEvent _event(
  AttentionEventStatus status, {
  String id = 'evt-001',
  String? acknowledgedBy,
  DateTime? acknowledgedAt,
  String? resolvedBy,
  DateTime? resolvedAt,
}) =>
    AttentionEvent(
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
      acknowledgedAt: acknowledgedAt,
      acknowledgedBy: acknowledgedBy,
      resolvedAt: resolvedAt,
      resolvedBy: resolvedBy,
      policyVersion: 'escalation-v1',
    );

class _Events implements AttentionEventRepository {
  AttentionEvent? detail;
  AttentionEvent? acknowledgeResult;
  AttentionEvent? resolveResult;
  ApiFailure? acknowledgeFailure;
  ApiFailure? resolveFailure;
  ApiFailure? detailFailure;
  int acknowledgeCalls = 0;
  int resolveCalls = 0;
  int detailCalls = 0;
  Completer<AttentionEvent>? acknowledgeCompleter;

  _Events({this.detail});

  Never _throw(ApiFailure kind) => throw ApiException(kind: kind);

  @override
  Future<AttentionEvent?> eventById(String eventId) async {
    detailCalls++;
    if (detailFailure != null) _throw(detailFailure!);
    return detail;
  }

  @override
  Future<AttentionEvent> acknowledge(String eventId) async {
    acknowledgeCalls++;
    if (acknowledgeFailure != null) _throw(acknowledgeFailure!);
    final completer = acknowledgeCompleter;
    if (completer != null) return completer.future;
    return acknowledgeResult!;
  }

  @override
  Future<AttentionEvent> resolve(String eventId) async {
    resolveCalls++;
    if (resolveFailure != null) _throw(resolveFailure!);
    return resolveResult!;
  }

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) async => const [];

  @override
  Future<List<AttentionEvent>> openEvents() async => const [];
}

class _Auth implements AuthRepository {
  int expireCalls = 0;

  @override
  Future<void> expireCurrentSession() async => expireCalls++;

  @override
  Future<void> validateCurrentSession() async {}
}

void main() {
  test('OPEN acknowledge replaces state with canonical server response', () async {
    final open = _event(AttentionEventStatus.open);
    final canonical = _event(
      AttentionEventStatus.acknowledged,
      acknowledgedBy: 'DR002',
      acknowledgedAt: DateTime.utc(2026, 9, 16, 8, 2),
    );
    final repo = _Events(detail: open)..acknowledgeResult = canonical;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
    );

    await controller.load();
    await controller.acknowledge();

    expect(controller.state.data, same(canonical));
    expect(controller.state.data!.acknowledgedBy, 'DR002');
    expect(repo.acknowledgeCalls, 1);
  });

  test('ACKNOWLEDGED resolve replaces state with canonical server response', () async {
    final acknowledged = _event(
      AttentionEventStatus.acknowledged,
      acknowledgedBy: 'DR001',
      acknowledgedAt: DateTime.utc(2026, 9, 16, 8, 2),
    );
    final canonical = _event(
      AttentionEventStatus.resolved,
      acknowledgedBy: 'DR001',
      acknowledgedAt: DateTime.utc(2026, 9, 16, 8, 2),
      resolvedBy: 'DR003',
      resolvedAt: DateTime.utc(2026, 9, 16, 8, 8),
    );
    final repo = _Events(detail: acknowledged)..resolveResult = canonical;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
    );

    await controller.load();
    await controller.resolve();

    expect(controller.state.data, same(canonical));
    expect(controller.state.data!.resolvedBy, 'DR003');
    expect(repo.resolveCalls, 1);
  });

  test('resolved and unknown events expose no unsafe mutation', () async {
    final resolvedRepo = _Events(detail: _event(AttentionEventStatus.resolved));
    final resolvedController = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: resolvedRepo,
    );
    await resolvedController.load();
    await resolvedController.acknowledge();
    await resolvedController.resolve();

    final unknownRepo = _Events(detail: _event(AttentionEventStatus.unknown));
    final unknownController = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: unknownRepo,
    );
    await unknownController.load();
    await unknownController.acknowledge();
    await unknownController.resolve();

    expect(resolvedRepo.acknowledgeCalls, 0);
    expect(resolvedRepo.resolveCalls, 0);
    expect(unknownRepo.acknowledgeCalls, 0);
    expect(unknownRepo.resolveCalls, 0);
  });

  test('duplicate acknowledge tap sends one mutation while in flight', () async {
    final repo = _Events(detail: _event(AttentionEventStatus.open))
      ..acknowledgeCompleter = Completer<AttentionEvent>();
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
    );
    await controller.load();

    final first = controller.acknowledge();
    final second = controller.acknowledge();
    await Future<void>.delayed(Duration.zero);

    expect(repo.acknowledgeCalls, 1);
    expect(controller.isMutating, isTrue);

    repo.acknowledgeCompleter!.complete(
      _event(
        AttentionEventStatus.acknowledged,
        acknowledgedBy: 'DR001',
        acknowledgedAt: DateTime.utc(2026, 9, 16, 8, 2),
      ),
    );
    await first;
    await second;
  });

  test('401 expires session without fabricating acknowledgement', () async {
    final auth = _Auth();
    final repo = _Events(detail: _event(AttentionEventStatus.open))
      ..acknowledgeFailure = ApiFailure.unauthorized;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
      authRepository: auth,
    );
    await controller.load();

    await controller.acknowledge();

    expect(controller.state.status, AsyncDataStatus.sessionExpired);
    expect(auth.expireCalls, 1);
  });

  test('403 preserves canonical event and valid session', () async {
    final auth = _Auth();
    final open = _event(AttentionEventStatus.open);
    final repo = _Events(detail: open)
      ..acknowledgeFailure = ApiFailure.forbidden;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
      authRepository: auth,
    );
    await controller.load();

    await controller.acknowledge();

    expect(controller.state.status, AsyncDataStatus.forbidden);
    expect(controller.state.data, same(open));
    expect(auth.expireCalls, 0);
  });

  test('404 load is explicit unavailable', () async {
    final repo = _Events()..detailFailure = ApiFailure.notFound;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.unavailable);
  });

  test('409 reloads canonical server state and actor/time', () async {
    final open = _event(AttentionEventStatus.open);
    final canonical = _event(
      AttentionEventStatus.acknowledged,
      acknowledgedBy: 'DR009',
      acknowledgedAt: DateTime.utc(2026, 9, 16, 8, 3),
    );
    final repo = _Events(detail: open)
      ..acknowledgeFailure = ApiFailure.conflict;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
    );
    await controller.load();
    repo.detail = canonical;

    await controller.acknowledge();

    expect(controller.state.data, same(canonical));
    expect(controller.state.data!.acknowledgedBy, 'DR009');
    expect(controller.mutationMessage, contains('changed on the server'));
    expect(repo.detailCalls, 2);
  });

  test('offline mutation leaves canonical lifecycle unchanged', () async {
    final open = _event(AttentionEventStatus.open);
    final repo = _Events(detail: open)
      ..acknowledgeFailure = ApiFailure.offline;
    final controller = AttentionEventDetailController(
      eventId: 'evt-001',
      repository: repo,
    );
    await controller.load();

    await controller.acknowledge();

    expect(controller.state.status, AsyncDataStatus.offline);
    expect(controller.state.data, same(open));
    expect(controller.state.data!.status, AttentionEventStatus.open);
  });
}
