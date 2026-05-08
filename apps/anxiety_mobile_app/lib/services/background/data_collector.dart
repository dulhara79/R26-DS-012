import 'dart:convert';
import 'package:battery_plus/battery_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';
import 'package:call_log/call_log.dart';
import 'package:usage_stats/usage_stats.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../background_service_helper.dart';

class DataCollector {
  static Future<void> collectAndSync(String userId) async {
    debugPrint("🚀 DataCollector: Starting sync for $userId");

    // 0. SERVICE HEARTBEAT — always succeeds, proves the service is alive.
    await _sendData(
      userId,
      "Service_Heartbeat",
      "Active_${DateTime.now().toIso8601String()}",
    );

    // A. LOCATION
    try {
      debugPrint("📍 DataCollector: Requesting Location...");
      final Position position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.medium,
        ),
      ).timeout(const Duration(seconds: 15));

      final locData = {
        'lat': position.latitude,
        'lng': position.longitude,
        'speed': position.speed,
        'accuracy': position.accuracy,
      };
      await _sendData(userId, "Location", jsonEncode(locData));
    } catch (e) {
      debugPrint("Location Error or Timeout: $e");
      await _sendData(userId, "System_Log", "Location_Error: $e");
    }

    // B. CALL LOGS (Last 24 h)
    try {
      debugPrint("📞 DataCollector: Querying Call Logs...");
      final int now = DateTime.now().millisecondsSinceEpoch;
      final Iterable<CallLogEntry> entries = await CallLog.query(
        dateFrom: now - (24 * 60 * 60 * 1000),
      ).timeout(const Duration(seconds: 10));

      final callStats = {
        'incoming':
            entries.where((c) => c.callType == CallType.incoming).length,
        'outgoing':
            entries.where((c) => c.callType == CallType.outgoing).length,
        'missed': entries.where((c) => c.callType == CallType.missed).length,
        'rejected':
            entries.where((c) => c.callType == CallType.rejected).length,
      };
      await _sendData(userId, "Call_Stats_24h", jsonEncode(callStats));
    } catch (e) {
      debugPrint("Call Log Error or Timeout: $e");
    }

    // C. SMS COUNTS (today only — no content)
    try {
      debugPrint("💬 DataCollector: Querying SMS...");
      final SmsQuery query = SmsQuery();

      final List<SmsMessage> inbox = await query
          .querySms(kinds: [SmsQueryKind.inbox])
          .timeout(const Duration(seconds: 10));

      final List<SmsMessage> sent = await query
          .querySms(kinds: [SmsQueryKind.sent])
          .timeout(const Duration(seconds: 10));

      final int receivedToday = inbox.where((m) => _isToday(m.date)).length;
      final int sentToday = sent.where((m) => _isToday(m.date)).length;

      await _sendData(
        userId,
        "SMS_Activity",
        jsonEncode({
          "received_today": receivedToday,
          "sent_today": sentToday,
          "total_today": receivedToday + sentToday,
        }),
      );
    } catch (e) {
      debugPrint("SMS Error or Timeout: $e");
    }

    // D. APP USAGE (Last 15 min)
    try {
      debugPrint("📱 DataCollector: Querying Usage Stats...");
      final DateTime end = DateTime.now();
      final DateTime start = end.subtract(const Duration(minutes: 15));

      final List<UsageInfo> usage =
          await UsageStats.queryUsageStats(start, end)
              .timeout(const Duration(seconds: 10));

      final Map<String, String> appUsage = {};
      for (final u in usage) {
        final int totalTime =
            int.tryParse(u.totalTimeInForeground ?? "0") ?? 0;
        if (totalTime > 1000) {
          appUsage[u.packageName ?? "unknown"] =
              "${(totalTime / 1000).toStringAsFixed(1)}s";
        }
      }
      if (appUsage.isNotEmpty) {
        await _sendData(userId, "App_Usage_15m", jsonEncode(appUsage));
      }
    } catch (e) {
      debugPrint("Usage Stats Error or Timeout: $e");
    }

    // E. BATTERY STATUS — read live from battery_plus, not from stale prefs.
    // BUG FIX: the original code read a prefs int that was only written when
    // the battery STATE changed.  On a phone sitting idle at 80 % all day the
    // key is never written, so every record shows "0%".
    // Fix: query the battery level directly here.
    try {
      final battery = Battery();
      final int level = await battery.batteryLevel
          .timeout(const Duration(seconds: 5));
      final BatteryState state = await battery.batteryState
          .timeout(const Duration(seconds: 5));

      // Also update the prefs key so the background monitor stays in sync.
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('last_battery_level', level);

      await _sendData(
        userId,
        "Battery_Status",
        jsonEncode({
          "level_percent": level,
          "state": state.name,            // 'charging', 'discharging', etc.
        }),
      );
    } catch (e) {
      debugPrint("Battery Status Error: $e");
    }

    // Push everything queued in this cycle to the server immediately.
    await BackgroundServiceHelper.retryOfflineQueue();

    debugPrint("✅ DataCollector: Sync Complete");
  }

  // ─────────────────────────────────────────────────────────────
  // HELPERS
  // ─────────────────────────────────────────────────────────────

  static Future<void> _sendData(
    String userId,
    String dataType,
    String value,
  ) async {
    try {
      // Always re-read userId in case it was set after the service started.
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload();
      final String currentId = prefs.getString('user_id') ?? userId;
      await BackgroundServiceHelper.sendToSheet(currentId, dataType, value);
      debugPrint("📤 Queued: $dataType");
    } catch (e) {
      debugPrint("Queue Error for $dataType: $e");
    }
  }

  static bool _isToday(DateTime? date) {
    if (date == null) return false;
    final now = DateTime.now();
    return date.year == now.year &&
        date.month == now.month &&
        date.day == now.day;
  }
}