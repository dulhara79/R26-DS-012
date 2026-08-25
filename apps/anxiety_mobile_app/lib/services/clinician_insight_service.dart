import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

/// Builds a privacy-safe, descriptive handoff from participant check-ins and
/// Component 2 behavioural observations.
///
/// This service DOES NOT send data over the network. It prepares and caches a
/// structured payload that can be handed to the central backend once the team
/// agrees a dedicated clinician-context endpoint.
///
/// Important safety rules:
/// - no exact GPS coordinates or app package names;
/// - no message/call content;
/// - no Component 2 experimental probability;
/// - Component 2 is always labelled `not_validated` / non-fusable here;
/// - check-in statistics are descriptive self-report context, not diagnosis.
class ClinicianInsightService {
  static const String _eventsKey = 'anxiety_alert_events_v1';
  static const String _c2ObservationKey = 'c2_observation_payload';
  static const String _handoffKey = 'clinician_insight_handoff_v1';

  static Future<List<CheckInRecord>> loadCheckInRecords(
    String userId, {
    int? days,
    int limit = 100,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    final raw = prefs.getString(_eventsKey);
    if (raw == null || raw.isEmpty) return const [];

    try {
      final decoded = jsonDecode(raw);
      if (decoded is! List) return const [];
      final cutoff = days == null
          ? null
          : DateTime.now().subtract(Duration(days: days));

      final records = decoded
          .whereType<Map>()
          .map((item) => CheckInRecord.fromJson(Map<String, dynamic>.from(item)))
          .where(
            (record) =>
                record.userId == userId &&
                record.riskSource != 'notification_test' &&
                (cutoff == null || record.detectedAt.isAfter(cutoff)),
          )
          .toList()
        ..sort((a, b) => b.detectedAt.compareTo(a.detectedAt));

      return records.take(limit).toList(growable: false);
    } catch (_) {
      return const [];
    }
  }

  static Future<Map<String, dynamic>> buildAndCache(String userId) async {
    final sevenDayRecords = await loadCheckInRecords(userId, days: 7);
    final thirtyDayRecords = await loadCheckInRecords(userId, days: 30);
    final prefs = await SharedPreferences.getInstance();

    Map<String, dynamic> c2 = <String, dynamic>{};
    final c2Raw = prefs.getString(_c2ObservationKey);
    if (c2Raw != null && c2Raw.isNotEmpty) {
      try {
        final decoded = jsonDecode(c2Raw);
        if (decoded is Map) c2 = Map<String, dynamic>.from(decoded);
      } catch (_) {}
    }

    final payload = <String, dynamic>{
      'schema_version': 'clinician_context_v1',
      'app_user_id': userId,
      'generated_at': DateTime.now().toUtc().toIso8601String(),
      'check_ins': {
        'seven_day': _summary(sevenDayRecords),
        'thirty_day': _summary(thirtyDayRecords),
        'recent_events': thirtyDayRecords
            .take(10)
            .map((record) => record.toClinicianSafeJson())
            .toList(growable: false),
      },
      'behavioral_context': _safeBehavioralContext(c2),
      'interpretation_limits': {
        'check_ins':
            'Participant-reported context and follow-up outcomes; descriptive only.',
        'behavioral':
            'Within-person behavioural observations only. Component 2 is not a validated clinical anxiety predictor.',
        'fusion':
            'Component 2 remains excluded from the composite and contributes no numerical risk score.',
      },
    };

    await prefs.setString(_handoffKey, jsonEncode(payload));
    return payload;
  }

  static Future<Map<String, dynamic>?> loadCachedHandoff() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_handoffKey);
    if (raw == null || raw.isEmpty) return null;
    try {
      final decoded = jsonDecode(raw);
      return decoded is Map ? Map<String, dynamic>.from(decoded) : null;
    } catch (_) {
      return null;
    }
  }

  static Map<String, dynamic> _summary(List<CheckInRecord> records) {
    final answered = records.where((r) => r.confirmedAnxious != null).toList();
    final confirmed = answered.where((r) => r.confirmedAnxious == true).toList();
    final followups = records.where((r) => r.feltBetter != null).toList();
    final helpful = followups.where((r) => r.feltBetter == true).toList();
    final interventionAttempts = records
        .where((r) => r.intervention != null || r.alternativeAction != null)
        .length;

    return {
      'events': records.length,
      'answered': answered.length,
      'confirmed_anxiety': confirmed.length,
      'not_confirmed': answered.length - confirmed.length,
      'response_rate': records.isEmpty ? null : answered.length / records.length,
      'confirmation_rate':
          answered.isEmpty ? null : confirmed.length / answered.length,
      'common_context': _mostCommon(records.map((r) => r.activity)),
      'intervention_attempts': interventionAttempts,
      'followups_answered': followups.length,
      'felt_better_count': helpful.length,
      'felt_better_rate':
          followups.isEmpty ? null : helpful.length / followups.length,
      'most_helpful_action': _mostCommon(
        helpful.map((r) => r.alternativeAction ?? r.intervention),
      ),
    };
  }

  static String? _mostCommon(Iterable<String?> values) {
    final counts = <String, int>{};
    for (final value in values) {
      final cleaned = value?.trim();
      if (cleaned == null || cleaned.isEmpty) continue;
      counts[cleaned] = (counts[cleaned] ?? 0) + 1;
    }
    if (counts.isEmpty) return null;
    return counts.entries.reduce((a, b) => a.value >= b.value ? a : b).key;
  }

  static Map<String, dynamic> _safeBehavioralContext(
    Map<String, dynamic> source,
  ) {
    final observations = source['observations'];
    final patterns = <Map<String, dynamic>>[];
    if (observations is Map) {
      for (final entry in observations.entries) {
        if (entry.value is! Map) continue;
        final value = Map<String, dynamic>.from(entry.value as Map);
        patterns.add({
          'key': entry.key.toString(),
          'label': value['label']?.toString() ?? entry.key.toString(),
          'direction': value['direction']?.toString() ?? 'unknown',
          if (value['z'] is num)
            'within_person_z': (value['z'] as num).toDouble(),
        });
      }
    }

    Map<String, dynamic>? change;
    if (source['change_detection'] is Map) {
      final raw = Map<String, dynamic>.from(source['change_detection'] as Map);
      change = {
        'detected': raw['detected'] == true,
        if (raw['feature'] != null) 'feature': raw['feature'].toString(),
        if (raw['direction'] != null)
          'direction': raw['direction'].toString(),
        if (raw['ewma_z'] is num)
          'ewma_z': (raw['ewma_z'] as num).toDouble(),
      };
    }

    final quality = source['data_quality'] is Map
        ? Map<String, dynamic>.from(source['data_quality'] as Map)
        : <String, dynamic>{};

    return {
      'status': 'not_validated',
      'fusion_eligible': false,
      'score': null,
      'baseline_ready': source['baseline_ready'] == true,
      'reportable': source['reportable'] == true,
      'patterns': patterns,
      if (change != null) 'change_detection': change,
      'data_quality': {
        if (quality['days_enrolled'] is num)
          'days_enrolled': (quality['days_enrolled'] as num).toInt(),
        if (quality['baseline_days_required'] is num)
          'baseline_days_required':
              (quality['baseline_days_required'] as num).toInt(),
        if (quality['baseline_usable_days'] is num)
          'baseline_usable_days':
              (quality['baseline_usable_days'] as num).toInt(),
        if (quality['recent_usable_days'] is num)
          'recent_usable_days':
              (quality['recent_usable_days'] as num).toInt(),
      },
    };
  }
}

class CheckInRecord {
  final String eventId;
  final String userId;
  final DateTime detectedAt;
  final String riskSource;
  final bool? confirmedAnxious;
  final String? activity;
  final String? intervention;
  final String? alternativeAction;
  final bool? interventionCompleted;
  final DateTime? followupAt;
  final bool? feltBetter;

  const CheckInRecord({
    required this.eventId,
    required this.userId,
    required this.detectedAt,
    required this.riskSource,
    this.confirmedAnxious,
    this.activity,
    this.intervention,
    this.alternativeAction,
    this.interventionCompleted,
    this.followupAt,
    this.feltBetter,
  });

  factory CheckInRecord.fromJson(Map<String, dynamic> json) {
    return CheckInRecord(
      eventId: json['event_id']?.toString() ?? '',
      userId: json['user_id']?.toString() ?? '',
      detectedAt: DateTime.tryParse(json['detected_at']?.toString() ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
      riskSource: json['risk_source']?.toString() ?? 'physiological',
      confirmedAnxious: json['confirmed_anxious'] as bool?,
      activity: json['activity']?.toString(),
      intervention: json['intervention']?.toString(),
      alternativeAction: json['alternative_action']?.toString(),
      interventionCompleted: json['intervention_completed'] as bool?,
      followupAt: json['followup_at'] == null
          ? null
          : DateTime.tryParse(json['followup_at'].toString()),
      feltBetter: json['felt_better'] as bool?,
    );
  }

  String? get actionTaken => alternativeAction ?? intervention;

  Map<String, dynamic> toClinicianSafeJson() {
    return {
      'detected_at': detectedAt.toUtc().toIso8601String(),
      'source': riskSource,
      'participant_confirmed_anxiety': confirmedAnxious,
      if (activity != null) 'context': activity,
      if (actionTaken != null) 'action_taken': actionTaken,
      if (interventionCompleted != null)
        'guided_intervention_completed': interventionCompleted,
      if (followupAt != null)
        'followup_at': followupAt!.toUtc().toIso8601String(),
      'felt_better_at_followup': feltBetter,
    };
  }
}
