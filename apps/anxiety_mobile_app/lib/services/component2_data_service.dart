import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import 'supabase_research_service.dart';

/// Fetches validated Component 2 behavioural outputs and stores only
/// display-safe, descriptive data for the Flutter UI.
///
/// Configure the real backend with:
///   --dart-define=COMPONENT2_API_URL=https://your-component2-api.example
///
/// Expected endpoint:
///   GET /behavioral/{participantId}
///
/// Until that backend is available, debug builds running in Chrome receive a
/// clearly labeled synthetic fixture so the participant-facing UI can be
/// reviewed. The fixture is impossible to activate in release mode.
///
/// IMPORTANT: GATv2/risk probabilities are deliberately not persisted into the
/// participant-facing cache. The current research evidence does not support a
/// behavioural anxiety-risk score for display or active fusion.
class Component2DataService {
  static const String _baseUrl = String.fromEnvironment(
    'COMPONENT2_API_URL',
    defaultValue: '',
  );

  static const Duration _timeout = Duration(seconds: 15);

  static bool get isConfigured => _baseUrl.trim().isNotEmpty;

  static Future<Component2SyncResult> sync(String participantId) async {
    final id = participantId.trim();
    if (id.isEmpty) {
      return const Component2SyncResult(
        success: false,
        status: 'missing_participant_id',
      );
    }

    if (!isConfigured) {
      // Chrome/debug preview only. Never available in a release build.
      if (kDebugMode && kIsWeb) {
        await _seedSyntheticWebDemo(id);
        debugPrint(
          '[Component2] Real backend not configured; loaded SYNTHETIC WEB DEMO data.',
        );
        return const Component2SyncResult(
          success: true,
          status: 'demo_data',
        );
      }

      debugPrint(
        '[Component2] COMPONENT2_API_URL is not configured; using cached/local data.',
      );
      return const Component2SyncResult(
        success: false,
        status: 'not_configured',
      );
    }

    try {
      final authUserId = await SupabaseResearchService.ensureParticipant(id);
      final accessToken =
          SupabaseResearchService.client?.auth.currentSession?.accessToken;

      if (authUserId == null || accessToken == null || accessToken.isEmpty) {
        debugPrint('[Component2] No authenticated Supabase session available.');
        return const Component2SyncResult(
          success: false,
          status: 'missing_auth_session',
        );
      }

      final base = _baseUrl.endsWith('/')
          ? _baseUrl.substring(0, _baseUrl.length - 1)
          : _baseUrl;
      final uri = Uri.parse('$base/behavioral/${Uri.encodeComponent(id)}');
      final response = await http
          .get(
            uri,
            headers: {
              'Accept': 'application/json',
              'Authorization': 'Bearer $accessToken',
            },
          )
          .timeout(_timeout);

      if (response.statusCode != 200) {
        debugPrint(
          '[Component2] Sync failed: HTTP ${response.statusCode} ${response.body}',
        );
        return Component2SyncResult(
          success: false,
          status: 'http_${response.statusCode}',
        );
      }

      final decoded = jsonDecode(response.body);
      if (decoded is! Map<String, dynamic>) {
        return const Component2SyncResult(
          success: false,
          status: 'invalid_payload',
        );
      }

      final prefs = await SharedPreferences.getInstance();

      final rawObservation = decoded['observation_payload'];
      final observation = rawObservation is Map<String, dynamic>
          ? rawObservation
          : decoded;

      final safeObservation = _displaySafeObservationPayload(observation, id);
      if (_looksLikeObservationPayload(safeObservation)) {
        await prefs.setString(
          'c2_observation_payload',
          jsonEncode(safeObservation),
        );
      }

      final passive = decoded['passive_metrics'];
      if (passive is Map<String, dynamic>) {
        await prefs.setString('c2_passive_metrics', jsonEncode(passive));
      }

      final coverage = decoded['day_coverage'];
      if (coverage is List) {
        await prefs.setString('c2_day_coverage', jsonEncode(coverage));
      }

      final checkIns = decoded['checkin_history'];
      if (checkIns is List) {
        await prefs.setString('c2_checkin_history', jsonEncode(checkIns));
      }

      await prefs.setString(
        'c2_fusion_handoff',
        jsonEncode(buildFusionHandoff(participantId: id)),
      );

      await prefs.setString(
        'c2_last_sync_utc',
        DateTime.now().toUtc().toIso8601String(),
      );

      return const Component2SyncResult(success: true, status: 'ok');
    } catch (e, st) {
      debugPrint('[Component2] Sync error: $e');
      debugPrint('$st');
      return const Component2SyncResult(
        success: false,
        status: 'network_or_parse_error',
      );
    }
  }

  /// Contract for the multimodal fusion team while Component 2 remains
  /// insufficiently validated for inferential use.
  ///
  /// `behavioral_score` is intentionally null. A numeric 0 would incorrectly
  /// communicate "very low anxiety" rather than "no validated estimate".
  static Map<String, dynamic> buildFusionHandoff({
    required String participantId,
  }) {
    return {
      'component': 'behavioral',
      'participant_id': participantId,
      'model_status': 'withheld_pending_validation',
      'fusion_eligible': false,
      'behavioral_score': null,
      'recommended_weight': 0.0,
      'display_permitted': false,
      'timestamp': DateTime.now().toUtc().toIso8601String(),
    };
  }

  static bool _looksLikeObservationPayload(Map<String, dynamic> payload) {
    return payload.containsKey('observations') ||
        payload.containsKey('baseline_ready') ||
        payload.containsKey('reportable');
  }

  /// Allow-list only fields the participant-facing page may consume.
  /// Research-only model probabilities, risk bands, phenotype labels and
  /// attention explanations are intentionally excluded.
  static Map<String, dynamic> _displaySafeObservationPayload(
    Map<String, dynamic> source,
    String participantId,
  ) {
    return {
      'participant_id': source['participant_id'] ?? participantId,
      if (source['synthetic'] == true) 'synthetic': true,
      if (source['window'] is Map) 'window': source['window'],
      'baseline_ready': source['baseline_ready'] ?? false,
      'reportable': source['reportable'] ?? false,
      if (source['observations'] is Map) 'observations': source['observations'],
      if (source['change_detection'] is Map)
        'change_detection': source['change_detection'],
      if (source['data_quality'] is Map) 'data_quality': source['data_quality'],
      if (source['blocking_issues'] is List)
        'blocking_issues': source['blocking_issues'],
    };
  }

  /// Synthetic fixture used only when running a debug build in Chrome and no
  /// real Component 2 endpoint has been configured. These values are fabricated
  /// solely to exercise the UI and MUST NOT be used in research results.
  static Future<void> _seedSyntheticWebDemo(String participantId) async {
    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now();
    final start = now.subtract(const Duration(days: 27));

    final observationPayload = <String, dynamic>{
      'participant_id': participantId,
      'synthetic': true,
      'baseline_ready': true,
      'reportable': true,
      'window': {
        'start': _dateOnly(start),
        'end': _dateOnly(now),
      },
      'observations': {
        'screen_activity': {
          'label': 'Screen activity',
          'value': 6.1,
          'unit': 'hours/day',
          'z': 1.2,
          'direction': 'above',
          'confidence': 'demo',
        },
        'mobility': {
          'label': 'Mobility',
          'value': 3.4,
          'unit': 'km/day',
          'z': -1.4,
          'direction': 'below',
          'confidence': 'demo',
        },
        'physical_activity': {
          'label': 'Physical activity',
          'value': 71.0,
          'unit': 'min/day',
          'z': -0.3,
          'direction': 'stable',
          'confidence': 'demo',
        },
        'routine_regularity': {
          'label': 'Routine regularity',
          'value': 0.76,
          'unit': '',
          'z': 0.2,
          'direction': 'stable',
          'confidence': 'demo',
        },
      },
      'data_quality': {
        'days_with_data': 13,
        'baseline_days_available': 28,
        'baseline_days_required': 28,
        'ema_received': 18,
        'ema_expected': 21,
      },
      'blocking_issues': <String>[],
    };

    final coverage = List.generate(14, (index) {
      final date = now.subtract(Duration(days: 13 - index));
      // Two intentionally incomplete days make the demo look realistic.
      final usable = index != 3 && index != 10;
      return {
        'date': _dateOnly(date),
        'usable': usable,
      };
    });

    final checkIns = [
      {
        'timestamp': now.subtract(const Duration(days: 1)).toIso8601String(),
        'answers': {
          'How are you feeling today?': 'A little tired but okay.',
        },
      },
      {
        'timestamp': now.subtract(const Duration(days: 4)).toIso8601String(),
        'answers': {
          'How are you feeling today?': 'Busy day. Felt mostly normal.',
        },
      },
      {
        'timestamp': now.subtract(const Duration(days: 8)).toIso8601String(),
        'answers': {
          'How are you feeling today?': 'Calm today.',
        },
      },
    ];

    final passiveMetrics = {
      'home_hours': 14.8,
      'away_hours': 9.2,
      'significant_places': 4,
      'sleep_proxy_window': '11:35 PM – 7:10 AM',
      'overnight_screen_off_hours': 7.6,
      'activity_proxy_score': 0.63,
      'activity_data_available': true,
      'synthetic': true,
    };

    await prefs.setString(
      'c2_observation_payload',
      jsonEncode(observationPayload),
    );
    await prefs.setString('c2_day_coverage', jsonEncode(coverage));
    await prefs.setString('c2_checkin_history', jsonEncode(checkIns));
    await prefs.setString('c2_passive_metrics', jsonEncode(passiveMetrics));
    await prefs.setString(
      'c2_fusion_handoff',
      jsonEncode(buildFusionHandoff(participantId: participantId)),
    );
    await prefs.setString(
      'c2_last_sync_utc',
      DateTime.now().toUtc().toIso8601String(),
    );
  }

  static String _dateOnly(DateTime value) =>
      '${value.year.toString().padLeft(4, '0')}-'
      '${value.month.toString().padLeft(2, '0')}-'
      '${value.day.toString().padLeft(2, '0')}';
}

class Component2SyncResult {
  final bool success;
  final String status;

  const Component2SyncResult({required this.success, required this.status});
}
