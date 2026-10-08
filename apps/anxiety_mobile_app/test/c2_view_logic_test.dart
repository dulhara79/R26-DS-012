import 'package:anxiety_mobile_app/services/c2_view_logic.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('observation set matches the four thesis observations', () {
    expect(kC2Observations.map((o) => o.key), [
      'screen_activity',
      'mobility',
      'movement_proxy',
      'social_media_use',
    ]);
    expect(
      kC2Observations.map((o) => o.label),
      isNot(contains('Physical activity')),
    );
  });

  test('timeline stages follow Days 1-28, Day 29+ and Day 57+', () {
    expect(c2StageFor(0), C2Stage.baseline);
    expect(c2StageFor(28), C2Stage.baseline);
    expect(c2StageFor(29), C2Stage.observations);
    expect(c2StageFor(56), C2Stage.observations);
    expect(c2StageFor(57), C2Stage.changeDetection);
  });

  test('range marker centres on the usual pattern and pins extremes', () {
    expect(c2RangePosition(0), closeTo(0.5, 1e-9));
    expect(c2RangePosition(1), closeTo(kC2UsualBandEnd, 1e-9));
    expect(c2RangePosition(-1), closeTo(kC2UsualBandStart, 1e-9));
    expect(c2RangePosition(26.67), 1.0);
    expect(c2RangePosition(-9), 0.0);
  });

  test('sync freshness flags a missed daily run', () {
    final now = DateTime(2026, 10, 8, 12);
    expect(c2SyncFreshness(null, now), C2SyncFreshness.never);
    expect(
      c2SyncFreshness(now.subtract(const Duration(hours: 20)), now),
      C2SyncFreshness.fresh,
    );
    expect(
      c2SyncFreshness(now.subtract(const Duration(hours: 40)), now),
      C2SyncFreshness.stale,
    );
  });

  test('relative time and value formatting are plain language', () {
    final now = DateTime(2026, 10, 8, 12);
    expect(c2RelativeTime(now, now), 'just now');
    expect(
      c2RelativeTime(now.subtract(const Duration(minutes: 5)), now),
      '5 min ago',
    );
    expect(
      c2RelativeTime(now.subtract(const Duration(hours: 1)), now),
      '1 hour ago',
    );
    expect(
      c2RelativeTime(now.subtract(const Duration(days: 3)), now),
      '3 days ago',
    );
    expect(c2FormatValue(4.25, 'hours/day'), '4.3 hours/day');
    expect(c2FormatValue(118, 'min/day'), '118 min/day');
    expect(c2FormatValue(null, 'km/day'), isNull);
    expect(
      c2FormatValue(10.8, '% high-motion samples'),
      '10.8% high-motion samples',
    );
  });
}
