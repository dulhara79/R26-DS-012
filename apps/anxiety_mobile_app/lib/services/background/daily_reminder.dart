import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
import 'service_config.dart';

class DailyReminder {
  static Future<void> checkAndShow(
    FlutterLocalNotificationsPlugin plugin,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final now = DateTime.now();
      final today = DateFormat('yyyy-MM-dd').format(now);

      // We check for 3 distinct periods: morning, afternoon, evening
      await _checkPeriod(prefs, plugin, now, today, 'morning');
      await _checkPeriod(prefs, plugin, now, today, 'afternoon');
      await _checkPeriod(prefs, plugin, now, today, 'evening');
    } catch (e) {
      debugPrint('Daily check error: $e');
    }
  }

  static Future<void> _checkPeriod(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
    String today,
    String period,
  ) async {
    // 1. Get the target time for this period (default to 9am, 2pm, 8pm if not set)
    int targetHour = prefs.getInt('ema_${period}_hour') ?? (period == 'morning' ? 9 : period == 'afternoon' ? 14 : 20);
    int targetMinute = prefs.getInt('ema_${period}_minute') ?? 0;

    // 2. Check if already submitted today
    String lastSubmitted = prefs.getString('ema_submitted_$period') ?? "";
    if (lastSubmitted == today) return;

    // 3. Check if notification already shown for this period today
    String lastShown = prefs.getString('ema_notified_$period') ?? "";
    if (lastShown == today) return;

    // 4. Trigger if current time matches target time (check within 1-minute window)
    if (now.hour == targetHour && now.minute == targetMinute) {
      final title = {
        'morning': '☀️ Morning Check-in',
        'afternoon': '🌤️ Afternoon Check-in',
        'evening': '🌙 Evening Check-in',
      }[period];

      await plugin.show(
        _getNotificationId(period),
        title,
        'How are you feeling right now? Tap to rate.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'ema_channel',
            'Daily Check-ins',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'ema_rating_$period',
      );

      await prefs.setString('ema_notified_$period', today);
    }
  }

  static int _getNotificationId(String period) {
    if (period == 'morning') return 901;
    if (period == 'afternoon') return 902;
    return 903;
  }
}
