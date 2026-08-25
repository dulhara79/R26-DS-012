import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

/// Stores privacy-safe summaries of participant self-reports for longitudinal
/// display and clinician-context handoff.
///
/// Only summary fields needed for trends are retained locally. Full questionnaire
/// item responses remain in the normal research event pipeline and are not copied
/// into the clinician-context cache.
class SelfReportHistoryService {
  static const String _historyKey = 'self_report_history_v1';
  static const int _maxRecords = 500;

  static bool isSupportedType(String type) =>
      type.startsWith('EMA_Rating_') ||
      type == 'GAD7_Weekly' ||
      type == 'PSS10_Weekly';

  static Future<void> captureResearchEvent({
    required String userId,
    required String type,
    required dynamic value,
    DateTime? recordedAt,
  }) async {
    if (!isSupportedType(type)) return;
    if (userId.isEmpty || userId == 'Unknown' || userId == 'No_User_ID') return;

    final payload = _decodeMap(value);
    if (payload == null) return;

    final record = _recordFromPayload(
      userId: userId,
      type: type,
      payload: payload,
      recordedAt: recordedAt ?? DateTime.now(),
    );
    if (record == null) return;

    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    final existing = await loadRecords(userId, includeAllUsers: true);

    final merged = <SelfReportRecord>[
      record,
      ...existing.where((item) => item.id != record.id),
    ]..sort((a, b) => b.recordedAt.compareTo(a.recordedAt));

    final retained = merged.take(_maxRecords).map((e) => e.toJson()).toList();
    await prefs.setString(_historyKey, jsonEncode(retained));
  }

  static Future<List<SelfReportRecord>> loadRecords(
    String userId, {
    int? days,
    int limit = _maxRecords,
    bool includeAllUsers = false,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    final raw = prefs.getString(_historyKey);
    if (raw == null || raw.isEmpty) return const [];

    try {
      final decoded = jsonDecode(raw);
      if (decoded is! List) return const [];
      final cutoff = days == null
          ? null
          : DateTime.now().subtract(Duration(days: days));
      final records = decoded
          .whereType<Map>()
          .map((e) => SelfReportRecord.fromJson(Map<String, dynamic>.from(e)))
          .where(
            (record) =>
                (includeAllUsers || record.userId == userId) &&
                (cutoff == null || record.recordedAt.isAfter(cutoff)),
          )
          .toList()
        ..sort((a, b) => b.recordedAt.compareTo(a.recordedAt));
      return records.take(limit).toList(growable: false);
    } catch (_) {
      return const [];
    }
  }

  static Future<Map<String, dynamic>> buildTrendSummary(String userId) async {
    final thirtyDays = await loadRecords(userId, days: 30);
    final sevenDays = thirtyDays
        .where(
          (record) => record.recordedAt.isAfter(
            DateTime.now().subtract(const Duration(days: 7)),
          ),
        )
        .toList();

    final all = await loadRecords(userId, limit: _maxRecords);
    final gad7 = all.where((r) => r.instrument == 'GAD-7').toList();
    final pss10 = all.where((r) => r.instrument == 'PSS-10').toList();

    return {
      'seven_day': {
        'ema': _emaSummary(sevenDays.where((r) => r.instrument == 'EMA')),
        'self_report_count': sevenDays.length,
      },
      'thirty_day': {
        'ema': _emaSummary(thirtyDays.where((r) => r.instrument == 'EMA')),
        'self_report_count': thirtyDays.length,
      },
      'gad7': _scoreTrend(gad7),
      'pss10': _scoreTrend(pss10),
      'recent_records': thirtyDays
          .take(12)
          .map((record) => record.toClinicianSafeJson())
          .toList(growable: false),
      'limits': {
        'meaning':
            'Self-report summaries are participant-reported measures and context, not a diagnosis.',
        'history':
            'This local trend history is available for submissions recorded after this feature is installed.',
      },
    };
  }

  static Map<String, dynamic> _emaSummary(Iterable<SelfReportRecord> records) {
    final ema = records.toList();
    if (ema.isEmpty) {
      return {
        'count': 0,
        'mean_stress': null,
        'mean_anxiety': null,
        'mean_fatigue': null,
        'mean_social_connection': null,
        'common_context': null,
      };
    }

    double? meanOf(String key) {
      final values = ema
          .map((r) => r.metrics[key])
          .whereType<num>()
          .map((v) => v.toDouble())
          .toList();
      if (values.isEmpty) return null;
      return values.reduce((a, b) => a + b) / values.length;
    }

    return {
      'count': ema.length,
      'mean_stress': _round2(meanOf('stress')),
      'mean_anxiety': _round2(meanOf('anxiety')),
      'mean_fatigue': _round2(meanOf('fatigue')),
      'mean_social_connection': _round2(meanOf('social')),
      'common_context': _mostCommon(ema.map((r) => r.context)),
      'first_recorded_at': ema.last.recordedAt.toUtc().toIso8601String(),
      'last_recorded_at': ema.first.recordedAt.toUtc().toIso8601String(),
    };
  }

  static Map<String, dynamic> _scoreTrend(List<SelfReportRecord> records) {
    if (records.isEmpty) {
      return {
        'available': false,
        'latest_score': null,
        'previous_score': null,
        'delta': null,
      };
    }

    final latest = records.first;
    final previous = records.length > 1 ? records[1] : null;
    final latestScore = latest.totalScore;
    final previousScore = previous?.totalScore;
    final delta = latestScore == null || previousScore == null
        ? null
        : latestScore - previousScore;

    return {
      'available': true,
      'latest_score': latestScore,
      'latest_recorded_at': latest.recordedAt.toUtc().toIso8601String(),
      if (latest.severity != null) 'latest_label': latest.severity,
      'previous_score': previousScore,
      if (previous != null)
        'previous_recorded_at': previous.recordedAt.toUtc().toIso8601String(),
      'delta': delta,
      'direction': delta == null
          ? 'insufficient_history'
          : delta > 0
              ? 'higher_than_previous'
              : delta < 0
                  ? 'lower_than_previous'
                  : 'unchanged',
      'interpretation':
          'Score change is shown descriptively and should be interpreted with the questionnaire definition and clinical context.',
    };
  }

  static SelfReportRecord? _recordFromPayload({
    required String userId,
    required String type,
    required Map<String, dynamic> payload,
    required DateTime recordedAt,
  }) {
    if (type.startsWith('EMA_Rating_')) {
      final period = payload['period']?.toString() ??
          type.substring('EMA_Rating_'.length);
      return SelfReportRecord(
        id: 'ema:${recordedAt.toUtc().microsecondsSinceEpoch}:$period',
        userId: userId,
        instrument: 'EMA',
        recordedAt: recordedAt,
        period: period,
        context: payload['context']?.toString(),
        metrics: {
          if (payload['stress'] is num) 'stress': payload['stress'],
          if (payload['anxiety'] is num) 'anxiety': payload['anxiety'],
          if (payload['fatigue'] is num) 'fatigue': payload['fatigue'],
          if (payload['social'] is num) 'social': payload['social'],
        },
      );
    }

    if (type == 'GAD7_Weekly') {
      return SelfReportRecord(
        id: 'gad7:${recordedAt.toUtc().microsecondsSinceEpoch}',
        userId: userId,
        instrument: 'GAD-7',
        recordedAt: recordedAt,
        totalScore: (payload['total_score'] as num?)?.toDouble(),
        severity: payload['severity']?.toString(),
      );
    }

    if (type == 'PSS10_Weekly') {
      return SelfReportRecord(
        id: 'pss10:${recordedAt.toUtc().microsecondsSinceEpoch}',
        userId: userId,
        instrument: 'PSS-10',
        recordedAt: recordedAt,
        totalScore: (payload['total_score'] as num?)?.toDouble(),
      );
    }

    return null;
  }

  static Map<String, dynamic>? _decodeMap(dynamic value) {
    if (value is Map) return Map<String, dynamic>.from(value);
    if (value is! String) return null;
    try {
      final decoded = jsonDecode(value);
      return decoded is Map ? Map<String, dynamic>.from(decoded) : null;
    } catch (_) {
      return null;
    }
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

  static double? _round2(double? value) =>
      value == null ? null : (value * 100).round() / 100;
}

class SelfReportRecord {
  final String id;
  final String userId;
  final String instrument;
  final DateTime recordedAt;
  final String? period;
  final String? context;
  final Map<String, dynamic> metrics;
  final double? totalScore;
  final String? severity;

  const SelfReportRecord({
    required this.id,
    required this.userId,
    required this.instrument,
    required this.recordedAt,
    this.period,
    this.context,
    this.metrics = const {},
    this.totalScore,
    this.severity,
  });

  factory SelfReportRecord.fromJson(Map<String, dynamic> json) {
    final metrics = json['metrics'] is Map
        ? Map<String, dynamic>.from(json['metrics'] as Map)
        : <String, dynamic>{};
    return SelfReportRecord(
      id: json['id']?.toString() ?? '',
      userId: json['user_id']?.toString() ?? '',
      instrument: json['instrument']?.toString() ?? 'Unknown',
      recordedAt: DateTime.tryParse(json['recorded_at']?.toString() ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
      period: json['period']?.toString(),
      context: json['context']?.toString(),
      metrics: metrics,
      totalScore: (json['total_score'] as num?)?.toDouble(),
      severity: json['severity']?.toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'user_id': userId,
        'instrument': instrument,
        'recorded_at': recordedAt.toUtc().toIso8601String(),
        if (period != null) 'period': period,
        if (context != null) 'context': context,
        if (metrics.isNotEmpty) 'metrics': metrics,
        if (totalScore != null) 'total_score': totalScore,
        if (severity != null) 'severity': severity,
      };

  Map<String, dynamic> toClinicianSafeJson() => {
        'instrument': instrument,
        'recorded_at': recordedAt.toUtc().toIso8601String(),
        if (period != null) 'period': period,
        if (context != null) 'context': context,
        if (metrics.isNotEmpty) 'metrics': metrics,
        if (totalScore != null) 'total_score': totalScore,
        if (severity != null) 'label': severity,
      };
}
