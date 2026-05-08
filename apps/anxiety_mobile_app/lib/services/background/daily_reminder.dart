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
///   • GAD-7 weekly  — between 09:00–21:00, re-fires every 4 h until done.
///   • PSS-10 weekly — between 09:00–21:00, once per day until done.
///
/// Called from Timer.periodic(Duration(minutes: 1)) in background_service.dart.
class DailyReminder {
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

    final bool enabled = prefs.getBool('rating_enabled') ?? true;
    if (!enabled) {
      debugPrint("DailyReminder: disabled by user — skip.");
      return;
    }

    final DateTime now   = DateTime.now();
    final String   today = DateFormat('yyyy-MM-dd').format(now);

    debugPrint(
      "DailyReminder: tick "
      "${now.hour.toString().padLeft(2, '0')}:"
      "${now.minute.toString().padLeft(2, '0')} — $today",
    );

    await _checkPeriod(prefs, plugin, now, today, 'morning');
    await _checkPeriod(prefs, plugin, now, today, 'afternoon');
    await _checkPeriod(prefs, plugin, now, today, 'evening');
    await _checkWeeklyGad7(prefs, plugin, now, today);
    await _checkWeeklyPss10(prefs, plugin, now, today);
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
    debugPrint("DailyReminder: throttle timestamps cleared after settings change.");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // EMA PERIOD
  // ─────────────────────────────────────────────────────────────────────────

  static Future<void> _checkPeriod(
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
      return;
    }

    // Read configured time (written by RatingSettingsPage).
    final int targetHour = prefs.getInt('ema_${period}_hour') ??
        (period == 'morning' ? 9 : period == 'afternoon' ? 14 : 20);
    final int targetMinute = prefs.getInt('ema_${period}_minute') ?? 0;

    final int nowMinutes    = now.hour * 60 + now.minute;
    final int targetMinutes = targetHour * 60 + targetMinute;

    // Active window: target time → target time + 4 hours.
    const int windowMinutes = 240;

    final bool inWindow =
        nowMinutes >= targetMinutes &&
        nowMinutes < targetMinutes + windowMinutes;

    debugPrint(
      "DailyReminder: EMA [$period] "
      "target=${targetHour.toString().padLeft(2, '0')}:"
      "${targetMinute.toString().padLeft(2, '0')} "
      "now=${now.hour.toString().padLeft(2, '0')}:"
      "${now.minute.toString().padLeft(2, '0')} "
      "inWindow=$inWindow",
    );

    if (!inWindow) return;

    // Throttle: fire at most once per 55 minutes inside the window.
    final int lastTs = prefs.getInt('ema_reminder_ts_$period') ?? 0;
    final int nowMs  = DateTime.now().millisecondsSinceEpoch;
    final int elapsedMin = ((nowMs - lastTs) / 60000).round();

    debugPrint(
      "DailyReminder: EMA [$period] lastTs=$lastTs "
      "elapsed=${elapsedMin}min throttle=55min",
    );

    if ((nowMs - lastTs) < 55 * 60 * 1000) {
      debugPrint("DailyReminder: EMA [$period] throttled — ${55 - elapsedMin} min remaining.");
      return;
    }

    final titles = {
      'morning'  : '☀️ Morning Check-in',
      'afternoon': '🌤️ Afternoon Check-in',
      'evening'  : '🌙 Evening Check-in',
    };
    final bodies = {
      'morning'  : 'Good morning! Take a moment to rate how you feel today.',
      'afternoon': 'Midday check-in — how are you feeling right now?',
      'evening'  : 'Evening check-in — wrap up your day with a quick rating.',
    };

    debugPrint("DailyReminder: ▶ FIRING EMA [$period] notification");

    try {
      await plugin.show(
        _idForPeriod(period),
        titles[period],
        bodies[period],
        NotificationDetails(
          android: AndroidNotificationDetails(
            'ema_channel',
            'Daily Check-ins',
            channelDescription: 'Scheduled mood and anxiety ratings',
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
    } catch (e, st) {
      debugPrint("DailyReminder: ❌ EMA [$period] plugin.show() failed: $e\n$st");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // WEEKLY GAD-7
  // ─────────────────────────────────────────────────────────────────────────

  static Future<void> _checkWeeklyGad7(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
    String today,
  ) async {
    if (now.hour < 9 || now.hour > 21) return;

    final bool due = await isGad7DueThisWeek();
    debugPrint("DailyReminder: GAD-7 due=$due");
    if (!due) return;

    final int lastTs = prefs.getInt('gad7_reminder_ts') ?? 0;
    final int nowMs  = DateTime.now().millisecondsSinceEpoch;
    if ((nowMs - lastTs) < 4 * 60 * 60 * 1000) return;

    debugPrint("DailyReminder: ▶ FIRING GAD-7 weekly notification");

    try {
      await plugin.show(
        700,
        '📋 Weekly Anxiety Check (GAD-7)',
        'Your 7-question weekly anxiety questionnaire is ready — about 2 minutes.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'gad7_channel',
            'Weekly Assessments',
            channelDescription: 'Weekly GAD-7 clinical questionnaires',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'gad7_weekly',
      );
      await prefs.setInt('gad7_reminder_ts', nowMs);
      debugPrint("DailyReminder: ✅ GAD-7 notification sent.");
    } catch (e) {
      debugPrint("DailyReminder: ❌ GAD-7 plugin.show() failed: $e");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // WEEKLY PSS-10
  // ─────────────────────────────────────────────────────────────────────────

  static Future<void> _checkWeeklyPss10(
    SharedPreferences prefs,
    FlutterLocalNotificationsPlugin plugin,
    DateTime now,
    String today,
  ) async {
    if (now.hour < 9 || now.hour > 21) return;
    if (prefs.getString('pss10_notified_today') == today) return;

    final bool due = await isPss10DueThisWeek();
    debugPrint("DailyReminder: PSS-10 due=$due");
    if (!due) return;

    debugPrint("DailyReminder: ▶ FIRING PSS-10 weekly notification");

    try {
      await plugin.show(
        800,
        '📊 Weekly Stress Check (PSS-10)',
        'Your 10-question perceived stress scale is ready — about 3 minutes.',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'pss_channel',
            'Monthly Assessments',
            channelDescription: 'Monthly PSS-10 stress scale assessments',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'pss10_monthly',
      );
      await prefs.setString('pss10_notified_today', today);
      debugPrint("DailyReminder: ✅ PSS-10 notification sent.");
    } catch (e) {
      debugPrint("DailyReminder: ❌ PSS-10 plugin.show() failed: $e");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  static int _idForPeriod(String period) {
    if (period == 'morning')   return 901;
    if (period == 'afternoon') return 902;
    return 903; // evening
  }
}