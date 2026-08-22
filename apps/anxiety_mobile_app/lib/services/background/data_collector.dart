import 'package:battery_plus/battery_plus.dart';
import 'package:call_log/call_log.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';
import 'package:geolocator/geolocator.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:usage_stats/usage_stats.dart';
import '../background_service_helper.dart';

class DataCollector {
  static const Duration _hourly = Duration(hours: 1);

  static Future<void> collectAndSync(String userId) async {
    debugPrint('🚀 DataCollector: Starting sync for $userId');
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    await _collectHourlyHeartbeat(userId, prefs);
    await _collectLocation(userId);
    await _collectPreviousDayCommunication(userId, prefs);
    await _collectAppUsage(userId);
    await _collectHourlyBattery(userId, prefs);
    await BackgroundServiceHelper.retryOfflineQueue();
    debugPrint('✅ DataCollector: Sync Complete');
  }

  static Future<void> _collectHourlyHeartbeat(String userId, SharedPreferences prefs) async {
    final now = DateTime.now();
    final last = DateTime.tryParse(prefs.getString('c2_last_heartbeat_at') ?? '');
    if (last != null && now.difference(last) < _hourly) return;
    await _send(userId, 'Service_Heartbeat', {'status': 'active'}, eventTime: now);
    await prefs.setString('c2_last_heartbeat_at', now.toIso8601String());
  }

  static Future<void> _collectLocation(String userId) async {
    try {
      final p = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.high),
      ).timeout(const Duration(seconds: 15));
      await _send(userId, 'Location_Grid_100m', {
        'lat': _round(p.latitude, 3),
        'lng': _round(p.longitude, 3),
        'speed_mps': _round(p.speed, 2),
        'accuracy_m': _round(p.accuracy, 1),
        'privacy_grid_decimals': 3,
      });
    } catch (e) {
      debugPrint('Location Error or Timeout: $e');
      await _send(userId, 'System_Log', {'event': 'location_error', 'message': e.toString()});
    }
  }

  /// Stores one complete communication aggregate for the previous local day.
  /// This avoids 96 near-duplicate rolling records/day and avoids incomplete
  /// "today so far" counts. Data before the recorded enrollment date is skipped.
  static Future<void> _collectPreviousDayCommunication(
    String userId,
    SharedPreferences prefs,
  ) async {
    final now = DateTime.now();
    final todayStart = DateTime(now.year, now.month, now.day);
    final targetStart = todayStart.subtract(const Duration(days: 1));
    final targetEnd = todayStart;
    final targetKey = _dateKey(targetStart);

    if (prefs.getString('c2_last_communication_day') == targetKey) return;

    final enrolledRaw = prefs.getString('enrolled_date');
    final enrolled = enrolledRaw == null ? null : DateTime.tryParse(enrolledRaw);
    if (enrolled != null) {
      final enrolledStart = DateTime(enrolled.year, enrolled.month, enrolled.day);
      if (targetStart.isBefore(enrolledStart)) {
        await prefs.setString('c2_last_communication_day', targetKey);
        return;
      }
    }

    try {
      final entries = await CallLog.query(
        dateFrom: targetStart.millisecondsSinceEpoch,
        dateTo: targetEnd.millisecondsSinceEpoch - 1,
      ).timeout(const Duration(seconds: 10));
      await _send(userId, 'Call_Stats_Daily', {
        'date': targetKey,
        'incoming': entries.where((c) => c.callType == CallType.incoming).length,
        'outgoing': entries.where((c) => c.callType == CallType.outgoing).length,
        'missed': entries.where((c) => c.callType == CallType.missed).length,
        'rejected': entries.where((c) => c.callType == CallType.rejected).length,
      });
    } catch (e) {
      debugPrint('Call Log Error or Timeout: $e');
    }

    try {
      final query = SmsQuery();
      final inbox = await query.querySms(kinds: [SmsQueryKind.inbox]).timeout(const Duration(seconds: 10));
      final sent = await query.querySms(kinds: [SmsQueryKind.sent]).timeout(const Duration(seconds: 10));
      final received = inbox.where((m) => _isSameLocalDay(m.date, targetStart)).length;
      final sentCount = sent.where((m) => _isSameLocalDay(m.date, targetStart)).length;
      await _send(userId, 'SMS_Activity_Daily', {
        'date': targetKey,
        'received': received,
        'sent': sentCount,
        'total': received + sentCount,
      });
    } catch (e) {
      debugPrint('SMS Error or Timeout: $e');
    }

    await prefs.setString('c2_last_communication_day', targetKey);
  }

  static Future<void> _collectAppUsage(String userId) async {
    try {
      final end = DateTime.now();
      final start = end.subtract(const Duration(minutes: 15));
      final usage = await UsageStats.queryUsageStats(start, end).timeout(const Duration(seconds: 10));
      final categories = <String, double>{};
      for (final u in usage) {
        final ms = int.tryParse(u.totalTimeInForeground ?? '0') ?? 0;
        if (ms <= 1000) continue;
        final category = _categorizeApp(u.packageName ?? 'unknown');
        categories[category] = (categories[category] ?? 0) + ms / 1000.0;
      }
      if (categories.isNotEmpty) {
        await _send(userId, 'App_Usage_Category_15m', {
          'window_minutes': 15,
          'categories_sec': {for (final e in categories.entries) e.key: double.parse(e.value.toStringAsFixed(1))},
        });
      }
    } catch (e) {
      debugPrint('Usage Stats Error or Timeout: $e');
    }
  }

  static Future<void> _collectHourlyBattery(String userId, SharedPreferences prefs) async {
    final now = DateTime.now();
    final last = DateTime.tryParse(prefs.getString('c2_last_battery_upload_at') ?? '');
    if (last != null && now.difference(last) < _hourly) return;
    try {
      final battery = Battery();
      final level = await battery.batteryLevel.timeout(const Duration(seconds: 5));
      final state = await battery.batteryState.timeout(const Duration(seconds: 5));
      await prefs.setInt('last_battery_level', level);
      await _send(userId, 'Battery_Status', {'level_percent': level, 'state': state.name});
      await prefs.setString('c2_last_battery_upload_at', now.toIso8601String());
    } catch (e) {
      debugPrint('Battery Status Error: $e');
    }
  }

  static Future<void> _send(String userId, String type, dynamic value, {DateTime? eventTime}) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload();
      final currentId = prefs.getString('user_id') ?? userId;
      await BackgroundServiceHelper.enqueueResearchEvent(currentId, type, value, eventTime: eventTime);
      debugPrint('📤 Queued: $type');
    } catch (e) {
      debugPrint('Queue Error for $type: $e');
    }
  }

  static bool _isSameLocalDay(DateTime? value, DateTime day) {
    if (value == null) return false;
    return value.year == day.year && value.month == day.month && value.day == day.day;
  }

  static String _dateKey(DateTime d) => '${d.year.toString().padLeft(4, '0')}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';

  static double _round(double value, int decimals) {
    final factor = decimals == 1 ? 10.0 : decimals == 2 ? 100.0 : 1000.0;
    return (value * factor).roundToDouble() / factor;
  }

  static String _categorizeApp(String pkg) {
    final p = pkg.toLowerCase();
    if (RegExp(r'whatsapp|telegram|signal|viber|messenger|instagram|snapchat|tiktok|facebook|twitter|linkedin').hasMatch(p)) return 'Social_Media';
    if (RegExp(r'chrome|firefox|brave|opera|samsung.*internet|browser').hasMatch(p)) return 'Browser';
    if (RegExp(r'youtube|netflix|spotify|prime.*video|disney|media').hasMatch(p)) return 'Entertainment';
    if (RegExp(r'gmail|outlook|mail|email').hasMatch(p)) return 'Email';
    if (RegExp(r'maps|waze|uber|grab|ola|navigation|gps').hasMatch(p)) return 'Maps_Navigation';
    if (RegExp(r'camera|gallery|photo|video').hasMatch(p)) return 'Camera_Gallery';
    if (RegExp(r'game|clash|pubg|free.*fire').hasMatch(p)) return 'Games';
    if (RegExp(r'settings|launcher|home|systemui|android\.').hasMatch(p)) return 'System';
    if (RegExp(r'bank|pay|wallet|finance|money').hasMatch(p)) return 'Finance';
    if (RegExp(r'learn|study|course|education|university').hasMatch(p)) return 'Education';
    if (RegExp(r'health|fitness|medic|hospital|therapy|mental|anxiety|doctor').hasMatch(p)) return 'Health_Wellness';
    return 'Other';
  }
}
