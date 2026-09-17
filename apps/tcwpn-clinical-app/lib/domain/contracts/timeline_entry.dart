import 'contract_parsing.dart';

class TimelineEntry {
  final double? composite;
  final String? tier;
  final String? band;
  final String? assessmentStatus;
  final List<String> missingModalities;
  final DateTime? computedAt;
  final String? trigger;

  const TimelineEntry({
    required this.composite,
    required this.tier,
    required this.band,
    required this.assessmentStatus,
    required this.missingModalities,
    required this.computedAt,
    required this.trigger,
  });

  factory TimelineEntry.fromJson(Map<String, dynamic> json) {
    final rawMissing = json['missing_modalities'];
    return TimelineEntry(
      composite: contractDouble(json['composite']),
      tier: contractString(json['tier']),
      band: contractString(json['band']),
      assessmentStatus: contractString(json['assessment_status']),
      missingModalities: rawMissing is List
          ? rawMissing
              .where((value) => value != null)
              .map((value) => value.toString())
              .toList(growable: false)
          : const <String>[],
      computedAt: contractDateTime(json['computed_at']),
      trigger: contractString(json['trigger']),
    );
  }
}
