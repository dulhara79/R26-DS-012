import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/timeline_entry.dart';
import 'package:r26_ds012_app/domain/repositories/timeline_repository.dart';
import 'package:r26_ds012_app/features/patients/timeline_screen.dart';
import 'package:r26_ds012_app/state/timeline_controller.dart';

class _Repository implements TimelineRepository {
  final List<TimelineEntry> rows;
  _Repository(this.rows);

  @override
  Future<List<TimelineEntry>> history(String subjectId, {int limit = 200}) async => rows;
}

Future<TimelineController> _pump(
  WidgetTester tester,
  List<TimelineEntry> rows,
) async {
  final now = DateTime.utc(2026, 9, 16, 12);
  final controller = TimelineController(
    subjectId: 'subject-001',
    repository: _Repository(rows),
    now: () => now,
  );
  await controller.load();
  await tester.pumpWidget(
    MaterialApp(home: TimelineScreen(controller: controller, displayId: 'Patient A')),
  );
  await tester.pump();
  return controller;
}

void main() {
  testWidgets('renders server timeline values and gap warning without interpolation',
      (tester) async {
    await _pump(tester, [
      TimelineEntry(
        composite: 0.58,
        tier: 'Medium',
        band: 'AMBER',
        assessmentStatus: 'complete',
        missingModalities: const ['c2_behavioral'],
        computedAt: DateTime.utc(2026, 9, 16, 8),
        trigger: 'note-ingest',
      ),
    ]);

    expect(find.text('24 hours'), findsOneWidget);
    expect(find.text('7 days'), findsOneWidget);
    expect(find.textContaining('0.58'), findsOneWidget);
    expect(find.textContaining('Medium'), findsOneWidget);
    expect(find.textContaining('AMBER'), findsOneWidget);
    expect(find.textContaining('Complete'), findsOneWidget);
    expect(find.textContaining('note-ingest'), findsOneWidget);
    expect(find.textContaining('Gaps mean no assessment record was returned'), findsOneWidget);
  });

  testWidgets('missing composite is not rendered as zero or low', (tester) async {
    await _pump(tester, [
      const TimelineEntry(
        composite: null,
        tier: null,
        band: null,
        assessmentStatus: 'insufficient',
        missingModalities: ['c1_physiological'],
        computedAt: null,
        trigger: 'manual',
      ),
    ]);

    expect(find.textContaining('—'), findsWidgets);
    expect(find.textContaining('Time not reported'), findsOneWidget);
    expect(find.text('0.00'), findsNothing);
    expect(find.text('Low'), findsNothing);
  });

  testWidgets('empty selected window stays explicit and does not invent forecast history',
      (tester) async {
    await _pump(tester, [
      TimelineEntry(
        composite: 0.42,
        tier: 'Medium',
        band: 'AMBER',
        assessmentStatus: 'complete',
        missingModalities: const [],
        computedAt: DateTime.utc(2026, 9, 10),
        trigger: 'manual',
      ),
    ]);

    expect(find.textContaining('No assessment records were returned for this window'), findsOneWidget);
    expect(find.textContaining('Forecast history'), findsNothing);
    expect(find.textContaining('Attention event'), findsNothing);
  });
}
