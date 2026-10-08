/// Pure display rules for the Component 2 behavioural-context pages.
///
/// Kept free of Flutter widgets so the rules that the thesis describes
/// (Section 4.4.5: Days 1–28 baseline, Day 29+ observations, Day 57+ EWMA
/// change detection) can be unit-tested directly.
library;

/// The four participant-facing observations defined in the thesis, in the
/// order they are shown. Keys match `build_observation()` in processor.py.
const List<C2ObservationSpec> kC2Observations = [
  C2ObservationSpec('screen_activity', 'Screen activity'),
  C2ObservationSpec('mobility', 'Mobility'),
  C2ObservationSpec('movement_proxy', 'Movement proxy'),
  C2ObservationSpec('social_media_use', 'Social media use'),
];

class C2ObservationSpec {
  final String key;
  final String label;
  const C2ObservationSpec(this.key, this.label);
}

const int kC2BaselineDays = 28;
const int kC2ChangeDetectionDay = 57;

enum C2Stage { baseline, observations, changeDetection }

/// The participant-timeline stage for a given number of days enrolled.
C2Stage c2StageFor(int daysEnrolled) {
  if (daysEnrolled >= kC2ChangeDetectionDay) return C2Stage.changeDetection;
  if (daysEnrolled > kC2BaselineDays) return C2Stage.observations;
  return C2Stage.baseline;
}

/// Where a within-person z-score sits on the "usual range" marker, as a
/// fraction 0..1 of a bar that spans -3σ to +3σ. Values beyond ±3σ are pinned
/// to the ends so a single extreme day cannot push the dot off the bar.
double c2RangePosition(double z) => ((z.clamp(-3.0, 3.0) + 3.0) / 6.0);

/// The shaded "usual range" on the marker is ±1σ, matching the backend rule
/// that labels |z| < 1 as "similar to your usual pattern".
const double kC2UsualBandStart = 2.0 / 6.0;
const double kC2UsualBandEnd = 4.0 / 6.0;

enum C2SyncFreshness { never, fresh, stale }

/// Observations are processed once per completed day, so data older than
/// 36 hours means at least one daily run has been missed.
C2SyncFreshness c2SyncFreshness(DateTime? lastSync, DateTime now) {
  if (lastSync == null) return C2SyncFreshness.never;
  return now.difference(lastSync) > const Duration(hours: 36)
      ? C2SyncFreshness.stale
      : C2SyncFreshness.fresh;
}

/// "Updated 5 min ago" style label for the last successful sync.
String c2RelativeTime(DateTime then, DateTime now) {
  final diff = now.difference(then);
  if (diff.inMinutes < 1) return 'just now';
  if (diff.inMinutes < 60) return '${diff.inMinutes} min ago';
  if (diff.inHours < 24) {
    return '${diff.inHours} hour${diff.inHours == 1 ? '' : 's'} ago';
  }
  final days = diff.inDays;
  return '$days day${days == 1 ? '' : 's'} ago';
}

/// Formats an observation value with its unit, e.g. "4.2 hours/day".
String? c2FormatValue(double? value, String unit) {
  if (value == null) return null;
  final text = value.abs() >= 100
      ? value.toStringAsFixed(0)
      : value.toStringAsFixed(1);
  final u = unit.trim();
  if (u.isEmpty) return text;
  return u.startsWith('%') ? '$text$u' : '$text $u';
}
