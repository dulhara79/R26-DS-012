import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart' show Color;
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
import '../../ema_and_gad7.dart';

/// DailyReminder
///
/// Notification schedule:
///   • 3 EMA check-ins per day (morning / afternoon / evening)
///     – Fires at the user-configured time.
///     – Active window = configured time + 4 hours.
///     – Re-fires every 55 min inside the window until submitted.
///     – Throttle timestamp is cleared whenever the user saves new times
///       (call [clearThrottleTimestamps] from RatingSettingsPage._save).
///
///   • GAD-7 and PSS-10 each fire once per week at the configured day and time.
///
/// Called from Timer.periodic(Duration(minutes: 1)) in background_service.dart.
class DailyReminder {
  static bool _didLogNotificationCapability = false;
  static const String _lastScheduledReminderKey =
      'scheduled_reminder_last_shown_ts';
  static const Duration _scheduledReminderSpacing = Duration(minutes: 30);

  // ─────────────────────────────────────────────────────────────────────────
  // PUBLIC ENTRY POINT
  // ─────────────────────────────────────────────────────────────────────────

  static Future<void> checkAndShow(
    FlutterLocalNotificationsPlugin plugin,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    // Always reload so the background isolate sees the latest prefs written
    // by the UI isolate (settings changes, submission flags).
    await prefs.reload();

    final bool dailyEnabled = prefs.getBool('rating_enabled') ?? true;
    final bool weeklyEnabled = prefs.getBool('weekly_checkins_enabled') ?? true;
    if (!dailyEnabled && !weeklyEnabled) {
      debugPrint("DailyReminder: all scheduled check-ins disabled by user.");
      return;
    }

    await _logNotificationCapability(plugin);

    final DateTime now = DateTime.now();
    final String today = DateFormat('yyyy-MM-dd').format(now);
    final int nowMs = now.millisecondsSinceEpoch;
    final int lastScheduledReminder =
        prefs.getInt(_lastScheduledReminderKey) ?? 0;

    if (nowMs - lastScheduledReminder <
        _scheduledReminderSpacing.inMilliseconds) {
      debugPrint('DailyReminder: scheduled reminders are being spaced apart.');
      return;
    }

    debugPrint(
      "DailyReminder: tick "
      "${now.hour.toString().padLeft(2, '0')}:"
      "${now.minute.toString().padLeft(2, '0')} — $today",
    );

    if (dailyEnabled) {
      for (final period in ['morning', 'afternoon', 'evening']) {
        if (await _checkPeriod(prefs, plugin, now, today, period)) {
          await prefs.setInt(_lastScheduledReminderKey, nowMs);
          return;
        }
      }
    }
    if (!weeklyEnabled) return;

    if (await _checkWeeklyGad7(prefs, plugin, now)) {
      await prefs.setInt(_lastScheduledReminderKey, nowMs);
      return;
    }
    if (await _checkWeeklyPss10(prefs, plugin, now)) {
      await prefs.setInt(_lastScheduledReminderKey, nowMs);
    }
  }

  static Future<void> _logNotificationCapability(
    FlutterLocalNotificationsPlugin plugin,
  ) async {
    if (_didLogNotificationCapability) return;

    try {
      final android = plugin
          .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin
          >();

      if (android == null) {
        debugPrint("EMA_DEBUG: notifications_capability platform=non_android");
      } else {
        final bool? enabled = await android.areNotificationsEnabled();
        debugPrint("EMA_DEBUG: notifications_enabled=$enabled");
      }
    } catch (e) {
      debugPrint("EMA_DEBUG: notifications_enabled_check_failed error=$e");
    } finally {
      _didLogNotificationCapability = true;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CALLED BY RatingSettingsPage AFTER USER SAVES NEW TIMES
  // ─────────────────────────────────────────────────────────────────────────

  /// Clears all EMA throttle timestamps so the first tick after a time-change
  /// is not silently blocked by a recent timestamp from the old schedule.
  static Future<void> clearThrottleTimestamps() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('ema_reminder_ts_morning');
    await prefs.remove('ema_reminder_ts_afternoon');
    await prefs.remove('ema_reminder_ts_evening');
    await prefs.remove('ema_random_times_date');
    await prefs.remove(_lastScheduledReminderKey);
    debugPrint(
      "DailyReminder: throttle timestamps and random times cleared after settings change.",
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // EMA PERIOD
  // ─────────────────────────────────────────────────────────────────────────

  static Future<bool> _checkPeriod(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
    String today,
    String period,
  ) async {
    // Already submitted today — skip completely.
    final String submitted = prefs.getString('ema_submitted_$period') ?? '';
    if (submitted == today) {
      debugPrint("DailyReminder: EMA [$period] already submitted today.");
      debugPrint(
        "EMA_DEBUG: period=$period action=skip reason=already_submitted date=$today",
      );
      return false;
    }

    final int defaultHour = period == 'morning'
        ? 9
        : period == 'afternoon'
        ? 14
        : 20;
    final int targetHour = prefs.getInt('ema_${period}_hour') ?? defaultHour;
    final int targetMinute = prefs.getInt('ema_${period}_minute') ?? 0;
    final int targetMinutes = targetHour * 60 + targetMinute;

    final int nowMinutes = now.hour * 60 + now.minute;

    // Active window: target time → target time + 4 hours (240 minutes).
    const int windowMinutes = 240;
    final int endMinutes = targetMinutes + windowMinutes;

    bool inWindow = false;
    if (endMinutes < 1440) {
      inWindow = nowMinutes >= targetMinutes && nowMinutes < endMinutes;
    } else {
      // Handles windows that cross midnight (e.g. 11 PM to 3 AM)
      inWindow =
          nowMinutes >= targetMinutes || nowMinutes < (endMinutes - 1440);
    }

    debugPrint(
      "DailyReminder: EMA [$period] "
      "target=${targetHour.toString().padLeft(2, '0')}:"
      "${targetMinute.toString().padLeft(2, '0')} "
      "now=${now.hour.toString().padLeft(2, '0')}:"
      "${now.minute.toString().padLeft(2, '0')} "
      "inWindow=$inWindow",
    );

    if (!inWindow) {
      debugPrint(
        "EMA_DEBUG: period=$period action=skip reason=outside_window now=${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')} target=${targetHour.toString().padLeft(2, '0')}:${targetMinute.toString().padLeft(2, '0')}",
      );
      return false;
    }

    // Throttle: fire at most once per 55 minutes inside the window.
    final int lastTs = prefs.getInt('ema_reminder_ts_$period') ?? 0;
    final int nowMs = DateTime.now().millisecondsSinceEpoch;
    final int elapsedMin = ((nowMs - lastTs) / 60000).round();

    debugPrint(
      "DailyReminder: EMA [$period] lastTs=$lastTs "
      "elapsed=${elapsedMin}min throttle=55min",
    );

    if ((nowMs - lastTs) < 55 * 60 * 1000) {
      debugPrint(
        "DailyReminder: EMA [$period] throttled — ${55 - elapsedMin} min remaining.",
      );
      debugPrint(
        "EMA_DEBUG: period=$period action=skip reason=throttled elapsed_min=$elapsedMin",
      );
      return false;
    }

    final titles = {
      'morning': '☀️ Morning Check-in',
      'afternoon': '🌤️ Afternoon Check-in',
      'evening': '🌙 Evening Check-in',
    };
    final bodies = {
      'morning': 'Good morning! Take a moment to rate how you feel today.',
      'afternoon': 'Midday check-in — how are you feeling right now?',
      'evening': 'Evening check-in — wrap up your day with a quick rating.',
    };

    debugPrint("DailyReminder: ▶ FIRING EMA [$period] notification");
    debugPrint("EMA_DEBUG: period=$period action=attempt_send");

    try {
      await plugin.show(
        _idForPeriod(period),
        titles[period],
        bodies[period],
        NotificationDetails(
          android: AndroidNotificationDetails(
            'ema_channel',
            'Daily Check-ins',
            channelDescription: 'Reminders to check how you feel',
            importance: Importance.high,
            priority: Priority.high,
            color: const Color(0xFF5E60CE),
            styleInformation: const BigTextStyleInformation(''),
          ),
        ),
        payload: 'ema_rating_$period',
      );
      // Persist throttle timestamp ONLY after a successful show().
      await prefs.setInt('ema_reminder_ts_$period', nowMs);
      debugPrint("DailyReminder: ✅ EMA [$period] notification sent.");
      debugPrint(
        "EMA_DEBUG: period=$period action=sent notification_id=${_idForPeriod(period)}",
      );
      return true;
    } catch (e, st) {
      debugPrint(
        "DailyReminder: ❌ EMA [$period] plugin.show() failed: $e\n$st",
      );
      debugPrint("EMA_DEBUG: period=$period action=send_failed error=$e");
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // WEEKLY GAD-7
  // ─────────────────────────────────────────────────────────────────────────

  static Future<bool> _checkWeeklyGad7(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
  ) async {
    if (!_weeklyScheduleReached(
      prefs,
      now,
      hourKey: 'gad7_hour',
      minuteKey: 'gad7_minute',
      defaultHour: 20,
    )) {
      return false;
    }

    final weekKey = weeklyCheckInWeekKey(now);
    if (prefs.getString('gad7_notified_week') == weekKey) return false;

    final bool due = await isGad7DueThisWeek();
    debugPrint("DailyReminder: GAD-7 due=$due");
    if (!due) return false;

    debugPrint("DailyReminder: ▶ FIRING GAD-7 weekly notification");

    try {
      await plugin.show(
        700,
        '📋 Weekly Anxiety Check',
        'Your 7-question anxiety check is ready. It takes about 2 minutes.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'gad7_channel',
            'Weekly Check-ins',
            channelDescription: 'Reminder for your weekly anxiety check-in',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'gad7_weekly',
      );
      await prefs.setString('gad7_notified_week', weekKey);
      debugPrint("DailyReminder: ✅ GAD-7 notification sent.");
      return true;
    } catch (e) {
      debugPrint("DailyReminder: ❌ GAD-7 plugin.show() failed: $e");
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // WEEKLY PSS-10
  // ─────────────────────────────────────────────────────────────────────────

  static Future<bool> _checkWeeklyPss10(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
  ) async {
    if (!_weeklyScheduleReached(
      prefs,
      now,
      hourKey: 'pss10_hour',
      minuteKey: 'pss10_minute',
      defaultHour: 21,
    )) {
      return false;
    }

    final weekKey = weeklyCheckInWeekKey(now);
    if (prefs.getString('pss10_notified_week') == weekKey) return false;

    final bool due = await isPss10DueThisWeek();
    debugPrint("DailyReminder: PSS-10 due=$due");
    if (!due) return false;

    debugPrint("DailyReminder: ▶ FIRING PSS-10 weekly notification");

    try {
      await plugin.show(
        800,
        '📊 Weekly Stress Check',
        'Your 10-question stress check is ready. It takes about 3 minutes.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'pss_channel',
            'Weekly Check-ins',
            channelDescription: 'Reminder for your weekly stress check-in',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'pss10_weekly',
      );
      await prefs.setString('pss10_notified_week', weekKey);
      debugPrint("DailyReminder: ✅ PSS-10 notification sent.");
      return true;
    } catch (e) {
      debugPrint("DailyReminder: ❌ PSS-10 plugin.show() failed: $e");
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  static bool _weeklyScheduleReached(
    SharedPreferences prefs,
    DateTime now, {
    required String hourKey,
    required String minuteKey,
    required int defaultHour,
  }) {
    final weekday = prefs.getInt('weekly_checkin_weekday') ?? DateTime.sunday;
    if (now.weekday != weekday) return false;

    final targetMinutes =
        (prefs.getInt(hourKey) ?? defaultHour) * 60 +
        (prefs.getInt(minuteKey) ?? 0);
    return now.hour * 60 + now.minute >= targetMinutes;
  }

  static int _idForPeriod(String period) {
    if (period == 'morning') return 901;
    if (period == 'afternoon') return 902;
    return 903; // evening
  }
}
