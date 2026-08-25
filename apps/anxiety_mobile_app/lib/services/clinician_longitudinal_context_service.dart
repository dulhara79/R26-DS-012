import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import 'clinician_insight_service.dart';
import 'self_report_history_service.dart';

/// Combines four descriptive streams for clinician review:
/// 1) self-report trend,
/// 2) physiological event confirmations,
/// 3) intervention response,
/// 4) Component 2 behavioural changes.
///
/// This is context for clinical review, not an additional fusion score.
class ClinicianLongitudinalContextService {
  static const String cacheKey = 'clinician_longitudinal_context_v2';

  static Future<Map<String, dynamic>> buildAndCache(String userId) async {
    final base = await ClinicianInsightService.buildAndCache(userId);
    final selfReport = await SelfReportHistoryService.buildTrendSummary(userId);

    final checkIns = base['check_ins'] is Map
        ? Map<String, dynamic>.from(base['check_ins'] as Map)
        : <String, dynamic>{};
    final thirtyDay = checkIns['thirty_day'] is Map
        ? Map<String, dynamic>.from(checkIns['thirty_day'] as Map)
        : <String, dynamic>{};
    final sevenDay = checkIns['seven_day'] is Map
        ? Map<String, dynamic>.from(checkIns['seven_day'] as Map)
        : <String, dynamic>{};
    final behavioral = base['behavioral_context'] is Map
        ? Map<String, dynamic>.from(base['behavioral_context'] as Map)
        : <String, dynamic>{};

    Map<String, dynamic> physiologicalSummary(Map<String, dynamic> source) => {
          'events': source['events'],
          'answered': source['answered'],
          'confirmed_anxiety': source['confirmed_anxiety'],
          'not_confirmed': source['not_confirmed'],
          'response_rate': source['response_rate'],
          'confirmation_rate': source['confirmation_rate'],
          'common_context': source['common_context'],
        };

    Map<String, dynamic> interventionSummary(Map<String, dynamic> source) => {
          'intervention_attempts': source['intervention_attempts'],
          'followups_answered': source['followups_answered'],
          'felt_better_count': source['felt_better_count'],
          'felt_better_rate': source['felt_better_rate'],
          'most_helpful_action': source['most_helpful_action'],
        };

    final payload = <String, dynamic>{
      'schema_version': 'clinician_longitudinal_context_v2',
      'app_user_id': userId,
      'generated_at': DateTime.now().toUtc().toIso8601String(),
      'self_report_trend': selfReport,
      'physiological_event_confirmations': {
        'seven_day': physiologicalSummary(sevenDay),
        'thirty_day': physiologicalSummary(thirtyDay),
        'recent_events': checkIns['recent_events'] ?? const [],
      },
      'intervention_response': {
        'seven_day': interventionSummary(sevenDay),
        'thirty_day': interventionSummary(thirtyDay),
      },
      'c2_behavioral_changes': behavioral,
      'clinical_reading_notes': {
        'self_report':
            'Participant-reported EMA/GAD-7/PSS-10 trends are descriptive and should be interpreted with the instrument and clinical context.',
        'physiological':
            'Confirmation statistics describe how often the participant agreed with an Aura physiological-event check-in. They do not validate every model alert as an anxiety episode.',
        'intervention':
            'Follow-up response is participant reported and observational; it does not establish treatment efficacy.',
        'c2':
            'Component 2 provides within-person behavioural context only and remains excluded from the composite because its final model was not validated for active clinical risk scoring.',
      },
      'fusion_policy': {
        'c2_status': 'not_validated',
        'c2_fusion_eligible': false,
        'c2_score': null,
        'context_payload_affects_composite': false,
      },
      'privacy': {
        'exact_gps_shared': false,
        'app_package_names_shared': false,
        'call_or_sms_content_shared': false,
        'c2_experimental_probability_shared': false,
      },
    };

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(cacheKey, jsonEncode(payload));
    return payload;
  }

  static Future<Map<String, dynamic>?> loadCached() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(cacheKey);
    if (raw == null || raw.isEmpty) return null;
    try {
      final decoded = jsonDecode(raw);
      return decoded is Map ? Map<String, dynamic>.from(decoded) : null;
    } catch (_) {
      return null;
    }
  }
}
