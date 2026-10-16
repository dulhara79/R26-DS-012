import 'contract_parsing.dart';

class TimelineEntry {
  final double? composite;
  final String? tier;
  final String? band;
  final String? assessmentStatus;
  final List<String> missingModalities;
  final DateTime? computedAt;
  final String? trigger;
  final int? fusionResultId;
  final String? eventId;
  final String? eventType;
  final String? eventStatus;
  final double? forecastScore;
  final String? forecastTier;
  final String? forecastScope;
  final int? forecastHorizonMinutes;

  const TimelineEntry({
    required this.composite,
    required this.tier,
    required this.band,
    required this.assessmentStatus,
    required this.missingModalities,
    required this.computedAt,
    required this.trigger,
    this.fusionResultId,
    this.eventId,
    this.eventType,
    this.eventStatus,
    this.forecastScore,
    this.forecastTier,
    this.forecastScope,
    this.forecastHorizonMinutes,
  });

  factory TimelineEntry.fromJson(Map<String, dynamic> json) {
    final current = json['current_assessment'] is Map
        ? Map<String, dynamic>.from(json['current_assessment'] as Map)
        : const <String, dynamic>{};
    final forecast = json['forecast'] is Map
        ? Map<String, dynamic>.from(json['forecast'] as Map)
        : const <String, dynamic>{};
    final modalities = json['modalities'];
    return TimelineEntry(
      composite: contractDouble(current['score']),
      tier: contractString(current['tier']),
      band: contractString(current['band']),
      assessmentStatus: contractString(json['assessment_status']),
      missingModalities: modalities is List
          ? modalities
                .whereType<Map>()
                .where((value) => value['available'] != true)
                .map((value) => value['component_id']?.toString())
                .whereType<String>()
                .toList(growable: false)
          : const <String>[],
      computedAt: contractDateTime(json['computed_at']),
      trigger: null,
      fusionResultId: json['fusion_result_id'] is int
          ? json['fusion_result_id'] as int
          : null,
      forecastScore: contractDouble(forecast['score']),
      forecastTier: contractString(forecast['tier']),
      forecastScope: contractString(forecast['scope']),
      forecastHorizonMinutes: contractInt(forecast['horizon_minutes']),
    );
  }

  factory TimelineEntry.fromEvent(Map<String, dynamic> json) => TimelineEntry(
    composite: null,
    tier: null,
    band: null,
    assessmentStatus: null,
    missingModalities: const [],
    computedAt: contractDateTime(json['created_at']),
    trigger: null,
    fusionResultId: json['fusion_result_id'] is int
        ? json['fusion_result_id'] as int
        : null,
    eventId: contractString(json['id']),
    eventType: contractString(json['event_type']),
    eventStatus: contractString(json['status']),
  );
}
