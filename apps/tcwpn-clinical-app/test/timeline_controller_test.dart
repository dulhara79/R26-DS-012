import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/timeline_entry.dart';
import 'package:r26_ds012_app/domain/repositories/timeline_repository.dart';
import 'package:r26_ds012_app/state/async_data_state.dart';
import 'package:r26_ds012_app/state/timeline_controller.dart';

class _TimelineRepository implements TimelineRepository {
  List<TimelineEntry> rows = const [];
  Object? failure;
  String? subject;

  @override
  Future<List<TimelineEntry>> history(String subjectId, {int limit = 200}) async {
    subject = subjectId;
    if (failure != null) throw failure!;
    return rows;
  }
}

TimelineEntry _row(DateTime? time, double? composite) => TimelineEntry(
      composite: composite,
      tier: composite == null ? null : 'Medium',
      band: composite == null ? null : 'AMBER',
      assessmentStatus: composite == null ? 'insufficient' : 'complete',
      missingModalities: const [],
      computedAt: time,
      trigger: 'manual',
    );

void main() {
  final now = DateTime.utc(2026, 9, 16, 12);

  test('loads history for canonical subject id', () async {
    final repo = _TimelineRepository()..rows = [_row(now, 0.58)];
    final controller = TimelineController(
      subjectId: 'subject-001',
      repository: repo,
      now: () => now,
    );

    await controller.load();

    expect(repo.subject, 'subject-001');
    expect(controller.state.status, AsyncDataStatus.data);
  });

  test('24 hour view filters returned rows without synthesizing points', () async {
    final unknown = _row(null, null);
    final inside = _row(now.subtract(const Duration(hours: 6)), 0.58);
    final outside = _row(now.subtract(const Duration(hours: 30)), 0.41);
    final repo = _TimelineRepository()..rows = [inside, outside, unknown];
    final controller = TimelineController(
      subjectId: 'subject-001',
      repository: repo,
      now: () => now,
    );
    await controller.load();

    final visible = controller.visibleEntries;

    expect(visible, contains(inside));
    expect(visible, isNot(contains(outside)));
    expect(visible, contains(unknown));
    expect(visible.length, 2);
  });

  test('7 day view includes only returned rows from seven-day window plus unknown time',
      () async {
    final recent = _row(now.subtract(const Duration(days: 5)), 0.58);
    final old = _row(now.subtract(const Duration(days: 8)), 0.44);
    final unknown = _row(null, null);
    final repo = _TimelineRepository()..rows = [recent, old, unknown];
    final controller = TimelineController(
      subjectId: 'subject-001',
      repository: repo,
      now: () => now,
    );
    await controller.load();

    controller.setWindow(TimelineWindow.days7);

    expect(controller.visibleEntries, [recent, unknown]);
  });

  test('empty history is explicit empty state', () async {
    final controller = TimelineController(
      subjectId: 'subject-001',
      repository: _TimelineRepository(),
      now: () => now,
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.empty);
    expect(controller.visibleEntries, isEmpty);
  });

  test('offline history never fabricates timeline data', () async {
    final repo = _TimelineRepository()
      ..failure = const ApiException(kind: ApiFailure.offline);
    final controller = TimelineController(
      subjectId: 'subject-001',
      repository: repo,
      now: () => now,
    );

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.offline);
    expect(controller.state.data, isNull);
    expect(controller.visibleEntries, isEmpty);
  });
}
