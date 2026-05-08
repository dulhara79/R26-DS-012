import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
import 'service_config.dart';
import '../../ema_and_gad7.dart';

class DailyReminder {
  static Future<void> checkAndShow(
    FlutterLocalNotificationsPlugin plugin,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // 0. Respect the enabled toggle from settings
      final bool enabled = prefs.getBool('rating_enabled') ?? true;
      if (!enabled) return;

      final now = DateTime.now();
      final today = DateFormat('yyyy-MM-dd').format(now);

      // 1. Check for 3 distinct periods: morning, afternoon, evening
      await _checkPeriod(prefs, plugin, now, today, 'morning');
      await _checkPeriod(prefs, plugin, now, today, 'afternoon');
      await _checkPeriod(prefs, plugin, now, today, 'evening');

      // 2. Check for Weekly GAD-7 Assessment (Mondays)
      await _checkWeeklyGad7(prefs, plugin, now, today);

      // 3. Check for Weekly PSS-10 Assessment
      await _checkWeeklyPss10(prefs, plugin, now, today);
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
    // 1. Get the target time for this period
    int targetHour = prefs.getInt('ema_${period}_hour') ?? (period == 'morning' ? 9 : period == 'afternoon' ? 14 : 20);
    int targetMinute = prefs.getInt('ema_${period}_minute') ?? 0;

    // 2. Check if already submitted today
    String lastSubmitted = prefs.getString('ema_submitted_$period') ?? "";
    if (lastSubmitted == today) return;

    // 3. Logic: Trigger if current time is after target time
    final int nowMinutes = now.hour * 60 + now.minute;
    final int targetMinutes = targetHour * 60 + targetMinute;
    
    // Define the active window for this period (e.g., 4 hours after target)
    const int activeWindowMinutes = 240; 

    if (nowMinutes >= targetMinutes && nowMinutes < (targetMinutes + activeWindowMinutes)) {
      
      // 4. Check if we should remind again
      // We show a reminder at most once every 60 minutes if pending
      int lastReminderTimestamp = prefs.getInt('ema_reminder_ts_$period') ?? 0;
      int currentTimestamp = DateTime.now().millisecondsSinceEpoch;
      
      // If it's been more than 55 mins since last reminder
      if (currentTimestamp - lastReminderTimestamp > (55 * 60 * 1000)) {
        final title = {
          'morning': '☀️ Morning Check-in',
          'afternoon': '🌤️ Afternoon Check-in',
          'evening': '🌙 Evening Check-in',
        }[period];

        debugPrint("🔔 DailyReminder: Showing notification for $period");

        await plugin.show(
          _getNotificationId(period),
          title,
          'How are you feeling right now? Tap to rate.',
          const NotificationDetails(
            android: AndroidNotificationDetails(
              'ema_channel',
              'Daily Check-ins',
              channelDescription: 'Persistent research reminders',
              importance: Importance.high,
              priority: Priority.high,
              showWhen: true,
              enableVibration: true,
              fullScreenIntent: false,
            ),
          ),
          payload: 'ema_rating_$period',
        );

        await prefs.setInt('ema_reminder_ts_$period', currentTimestamp);
        await prefs.setString('ema_notified_$today', period); // Legacy support
      }
    }
  }

  static Future<void> _checkWeeklyGad7(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
    String today,
  ) async {
    // Only trigger on Mondays (or any day if we want to catch missed ones)
    // Research requirement usually says remind until done.
    
    // Check if assessment is due for this week
    bool isDue = await isGad7DueThisWeek();
    if (!isDue) return;

    // Only between 9 AM and 9 PM to avoid disturbing sleep
    if (now.hour < 9 || now.hour > 21) return;

    // Remind once every 4 hours for the weekly survey
    int lastReminder = prefs.getInt('gad7_reminder_ts') ?? 0;
    int nowMs = DateTime.now().millisecondsSinceEpoch;

    if (nowMs - lastReminder > (4 * 60 * 60 * 1000)) {
      debugPrint("🔔 DailyReminder: Showing Weekly GAD-7 notification");
      
      await plugin.show(
        777,
        '📊 Weekly Health Check',
        'It\'s time for your weekly GAD-7 anxiety assessment. Tap to begin.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'gad7_channel',
            'Weekly Assessments',
            channelDescription: 'Weekly research assessments',
            importance: Importance.high,
            priority: Priority.high,
            showWhen: true,
            enableVibration: true,
          ),
        ),
        payload: 'gad7_weekly',
      );

      await prefs.setInt('gad7_reminder_ts', nowMs);
    }
  }

  static Future<void> _checkWeeklyPss10(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
    String today,
  ) async {
    // Persistent logic: Notify daily between 9 AM and 9 PM until done for the week
    if (now.hour < 9 || now.hour > 21) return;

    String lastNotified = prefs.getString('pss10_notified_today') ?? "";
    if (lastNotified == today) return;

    if (await isPss10DueThisWeek()) {
      await plugin.show(
        888,
        '🧘 Weekly Reflection',
        'It\'s time for your weekly stress assessment. Tap to begin.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'pss_channel',
            'Weekly Assessments',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'pss10_monthly', // Route to PSS screen
      );
      await prefs.setString('pss10_notified_today', today);
    }
  }

  static int _getNotificationId(String period) {
    if (period == 'morning') return 901;
    if (period == 'afternoon') return 902;
    return 903;
  }
}
