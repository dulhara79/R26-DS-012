import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/timeline_entry.dart';

void main() {
  test('parses only server-reported timeline fields', () {
    final entry = TimelineEntry.fromJson({
      'composite': 0.58,
      'tier': 'Medium',
      'band': 'AMBER',
      'assessment_status': 'complete',
      'missing_modalities': ['c2_behavioral'],
      'computed_at': '2026-09-16T08:00:00Z',
      'trigger': 'note-ingest',
    });

    expect(entry.composite, 0.58);
    expect(entry.tier, 'Medium');
    expect(entry.band, 'AMBER');
    expect(entry.assessmentStatus, 'complete');
    expect(entry.missingModalities, ['c2_behavioral']);
    expect(
      entry.computedAt?.toUtc(),
      DateTime.utc(2026, 9, 16, 8),
    );
    expect(entry.trigger, 'note-ingest');
  });

  test('missing composite and timestamp remain missing', () {
    final entry = TimelineEntry.fromJson({
      'tier': null,
      'band': null,
      'assessment_status': null,
    });

    expect(entry.composite, isNull);
    expect(entry.computedAt, isNull);
    expect(entry.tier, isNull);
    expect(entry.band, isNull);
    expect(entry.assessmentStatus, isNull);
    expect(entry.missingModalities, isEmpty);
  });

  test('does not derive tier or band from composite', () {
    final entry = TimelineEntry.fromJson({'composite': 0.99});

    expect(entry.composite, 0.99);
    expect(entry.tier, isNull);
    expect(entry.band, isNull);
  });
}
