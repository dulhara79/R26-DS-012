import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:geolocator/geolocator.dart';
import 'package:call_log/call_log.dart';
import 'package:usage_stats/usage_stats.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:screen_state/screen_state.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:battery_plus/battery_plus.dart';
import 'background_service_helper.dart';
import 'config.dart';

// GOOGLE SCRIPT URL (Injected at build time via --dart-define=SCRIPT_URL=...)
const String kGoogleScriptUrl = AppConfig.googleScriptUrl;

// ─── Notification IDs ──────────────────────────────────────
const int kForegroundNotifId = 888;
const int kMorningEmaId = 901;
const int kAfternoonEmaId = 902;
const int kEveningEmaId = 903;
const int kGad7ReminderId = 904;

Future<void> initializeService() async {
  final service = FlutterBackgroundService();

  const AndroidNotificationChannel channel = AndroidNotificationChannel(
    'research_channel_01',
    'Data Collection Service',
    description: 'Running background research tasks',
    importance: Importance.low,
  );

  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  await flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin
      >()
      ?.createNotificationChannel(channel);

  const AndroidNotificationChannel emaChannel = AndroidNotificationChannel(
    'ema_channel',
    'Daily Check-ins',
    description: 'Anxiety check-in reminders (3x per day)',
    importance: Importance.high,
  );
  await flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin
      >()
      ?.createNotificationChannel(emaChannel);

  await service.configure(
    androidConfiguration: AndroidConfiguration(
      onStart: onStart,
      autoStart: true,
      autoStartOnBoot: true,
      isForegroundMode: true,
      notificationChannelId: 'research_channel_01',
      initialNotificationTitle: 'Research Active',
      initialNotificationContent: 'Collecting anonymous usage data...',
      foregroundServiceNotificationId: kForegroundNotifId,
    ),
    iosConfiguration: IosConfiguration(
      autoStart: true,
      onForeground: onStart,
      onBackground: onIosBackground,
    ),
  );
}

@pragma('vm:entry-point')
bool onIosBackground(ServiceInstance service) {
  WidgetsFlutterBinding.ensureInitialized();
  return true;
}

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  DartPluginRegistrant.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  String userId = prefs.getString('user_id') ?? "Unknown_User";

  try {
    await BackgroundServiceHelper.retryOfflineQueue();
  } catch (e) {
    debugPrint('Retry offline queue on service start failed: $e');
  }

  // ── FIX: connectivity_plus now delivers List<ConnectivityResult> ──
  try {
    Connectivity().onConnectivityChanged.listen((
      List<ConnectivityResult> results,
    ) async {
      final isConnected = results.any((r) => r != ConnectivityResult.none);
      if (isConnected) {
        try {
          await BackgroundServiceHelper.retryOfflineQueue();
        } catch (e) {
          debugPrint('Background service connectivity retry failed: $e');
        }
      }
    });
  } catch (e) {
    debugPrint('Background connectivity listener setup failed: $e');
  }

  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  if (service is AndroidServiceInstance) {
    service
        .on('setAsForeground')
        .listen((_) => service.setAsForegroundService());
    service
        .on('setAsBackground')
        .listen((_) => service.setAsBackgroundService());
  }
  service.on('stopService').listen((_) => service.stopSelf());

  // ── REAL-TIME: SCREEN STATE ─────────────────────────────
  Screen screen = Screen();
  try {
    screen.screenStateStream.listen(
      (ScreenStateEvent event) {
        String status = "Unknown";
        if (event == ScreenStateEvent.SCREEN_ON) status = "Screen_On";
        if (event == ScreenStateEvent.SCREEN_OFF) status = "Screen_Off";
        if (event == ScreenStateEvent.SCREEN_UNLOCKED) {
          status = "Screen_Unlocked";
        }
        _sendData(userId, "Screen_Event", status);
      },
      onError: (e) => debugPrint("Screen State Stream Error: $e"),
      cancelOnError: false,
    );
  } catch (e) {
    debugPrint("Screen State Setup Error: $e");
  }

  // ── REAL-TIME: ACCELEROMETER ────────────────────────────
  try {
    accelerometerEventStream().listen(
      (AccelerometerEvent event) {
        double magnitude = sqrt(
          event.x * event.x + event.y * event.y + event.z * event.z,
        );
        if (magnitude > 15.0) {
          _sendData(userId, "High_Motion_Event", magnitude.toStringAsFixed(2));
        }
      },
      onError: (e) => debugPrint("Accelerometer Sensor Error: $e"),
      cancelOnError: false,
    );
  } catch (e) {
    debugPrint("Sensor Setup Error: $e");
  }

  // ── REAL-TIME: BATTERY MONITOR ──────────────────────────
  try {
    final battery = Battery();
    battery.onBatteryStateChanged.listen((BatteryState state) async {
      final level = await battery.batteryLevel;
      if (level <= 15 && state == BatteryState.discharging) {
        _sendData(userId, "Critical_Battery_Warning", "Level: $level%");
      }
    });
  } catch (e) {
    debugPrint("Battery Monitor Error: $e");
  }

  // ── PERIODIC: HEAVY TASKS (Every 15 min) ───────────────
  Timer.periodic(const Duration(minutes: 15), (timer) async {
    if (service is AndroidServiceInstance) {
      if (await service.isForegroundService()) {
        flutterLocalNotificationsPlugin.show(
          kForegroundNotifId,
          'Research Active',
          'Last Sync: ${DateFormat('HH:mm').format(DateTime.now())}',
          const NotificationDetails(
            android: AndroidNotificationDetails(
              'research_channel_01',
              'Data Collection Service',
              icon: 'ic_bg_service_small',
              ongoing: true,
            ),
          ),
        );
      }
    }
    final freshPrefs = await SharedPreferences.getInstance();
    userId = freshPrefs.getString('user_id') ?? userId;

    await _collectAndSync(userId);

    // Heartbeat to confirm service is alive
    await _sendData(
      userId,
      "Service_Heartbeat",
      "Isolate_Active_${DateTime.now().toIso8601String()}",
    );
  });

  // ── EVERY MINUTE: EMA check-in + GAD-7 reminders ───────
  Timer.periodic(const Duration(minutes: 1), (timer) async {
    try {
      final freshPrefs = await SharedPreferences.getInstance();
      final freshUid = freshPrefs.getString('user_id') ?? userId;
      await _checkEmaSchedule(
        freshUid,
        flutterLocalNotificationsPlugin,
        freshPrefs,
      );
      await _checkGad7Reminder(flutterLocalNotificationsPlugin, freshPrefs);
    } catch (e) {
      debugPrint('Periodic check error: $e');
    }
  });
}

// ─────────────────────────────────────────────────────────
// EMA Schedule
// ─────────────────────────────────────────────────────────
Future<void> _checkEmaSchedule(
  String userId,
  FlutterLocalNotificationsPlugin plugin,
  SharedPreferences prefs,
) async {
  if (!(prefs.getBool('rating_enabled') ?? true)) return;

  final now = DateTime.now();
  final today = DateFormat('yyyy-MM-dd').format(now);

  final periods = {
    'morning': {
      'id': kMorningEmaId,
      'hour': prefs.getInt('ema_morning_hour') ?? 9,
      'minute': prefs.getInt('ema_morning_minute') ?? 0,
      'label': '☀️ Morning Check-in',
    },
    'afternoon': {
      'id': kAfternoonEmaId,
      'hour': prefs.getInt('ema_afternoon_hour') ?? 14,
      'minute': prefs.getInt('ema_afternoon_minute') ?? 0,
      'label': '🌤️ Afternoon Check-in',
    },
    'evening': {
      'id': kEveningEmaId,
      'hour': prefs.getInt('ema_evening_hour') ?? 20,
      'minute': prefs.getInt('ema_evening_minute') ?? 0,
      'label': '🌙 Evening Check-in',
    },
  };

  for (final entry in periods.entries) {
    final period = entry.key;
    final config = entry.value;
    final lastSubmitted = prefs.getString('ema_submitted_$period') ?? '';
    final lastShown = prefs.getString('ema_notif_shown_$period') ?? '';

    if (lastSubmitted == today) continue;
    if (lastShown == today) continue;

    if (now.hour == config['hour'] && now.minute == config['minute']) {
      await plugin.show(
        config['id'] as int,
        config['label'] as String,
        'How anxious do you feel right now? Tap to rate (1–5)',
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'ema_channel',
            'Daily Check-ins',
            importance: Importance.high,
            priority: Priority.high,
          ),
        ),
        payload: 'ema_$period',
      );
      await prefs.setString('ema_notif_shown_$period', today);
    }
  }
}

// ─────────────────────────────────────────────────────────
// GAD-7 Weekly Reminder
// ─────────────────────────────────────────────────────────
Future<void> _checkGad7Reminder(
  FlutterLocalNotificationsPlugin plugin,
  SharedPreferences prefs,
) async {
  final now = DateTime.now();
  if (now.weekday != DateTime.monday) return;
  if (now.hour != 9 || now.minute != 5) return;

  final weekNum =
      ((now.difference(DateTime(now.year, 1, 1)).inDays +
                  DateTime(now.year, 1, 1).weekday) /
              7)
          .ceil();
  final thisWeek = '${now.year}-W${weekNum.toString().padLeft(2, '0')}';
  final lastWeek = prefs.getString('last_gad7_week') ?? '';

  if (lastWeek == thisWeek) return;

  final lastNotifWeek = prefs.getString('gad7_notif_week') ?? '';
  if (lastNotifWeek == thisWeek) return;

  await plugin.show(
    kGad7ReminderId,
    '📋 Weekly Assessment Due',
    'Please complete your weekly GAD-7 anxiety questionnaire.',
    const NotificationDetails(
      android: AndroidNotificationDetails(
        'ema_channel',
        'Daily Check-ins',
        importance: Importance.high,
        priority: Priority.high,
      ),
    ),
    payload: 'gad7',
  );
  await prefs.setString('gad7_notif_week', thisWeek);
}

// ─────────────────────────────────────────────────────────
// Periodic Data Collection (every 15 min)
// ─────────────────────────────────────────────────────────
Future<void> _collectAndSync(String userId) async {
  // A. LOCATION
  try {
    Position position = await Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.high,
    );
    await _sendData(
      userId,
      "Location",
      jsonEncode({
        'lat': position.latitude,
        'lng': position.longitude,
        'speed': position.speed,
        'accuracy': position.accuracy,
      }),
    );
  } catch (e) {
    debugPrint("Location Error: $e");
  }

  // B. CALL LOGS (last 24 h)
  try {
    int now = DateTime.now().millisecondsSinceEpoch;
    Iterable<CallLogEntry> entries = await CallLog.query(
      dateFrom: now - (24 * 60 * 60 * 1000),
    );
    await _sendData(
      userId,
      "Call_Stats_24h",
      jsonEncode({
        'incoming': entries
            .where((c) => c.callType == CallType.incoming)
            .length,
        'outgoing': entries
            .where((c) => c.callType == CallType.outgoing)
            .length,
        'missed': entries.where((c) => c.callType == CallType.missed).length,
        'rejected': entries
            .where((c) => c.callType == CallType.rejected)
            .length,
        'total_duration_s': entries.fold<int>(
          0,
          (sum, c) => sum + (c.duration ?? 0),
        ),
      }),
    );
  } catch (e) {
    debugPrint("Call Log Error: $e");
  }

  // C. SMS (today)
  try {
    final SmsQuery query = SmsQuery();
    List<SmsMessage> inbox = await query.querySms(kinds: [SmsQueryKind.inbox]);
    List<SmsMessage> sent = await query.querySms(kinds: [SmsQueryKind.sent]);
    int receivedToday = inbox.where((m) => _isToday(m.date)).length;
    int sentToday = sent.where((m) => _isToday(m.date)).length;
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
    debugPrint("SMS Error: $e");
  }

  // D. APP USAGE (last 15 min)
  try {
    DateTime end = DateTime.now();
    DateTime start = end.subtract(const Duration(minutes: 15));
    List<UsageInfo> usage = await UsageStats.queryUsageStats(start, end);
    Map<String, String> appUsage = {};
    for (var u in usage) {
      int totalTime = int.parse(u.totalTimeInForeground ?? "0");
      if (totalTime > 1000) {
        appUsage[u.packageName ?? "unknown"] =
            "${(totalTime / 1000).toStringAsFixed(1)}s";
      }
    }
    if (appUsage.isNotEmpty) {
      await _sendData(userId, "App_Usage_15m", jsonEncode(appUsage));
    }
  } catch (e) {
    debugPrint("Usage Stats Error: $e");
  }

  // E. BATTERY
  try {
    final battery = Battery();
    final level = await battery.batteryLevel;
    final state = await battery.batteryState;
    await _sendData(
      userId,
      "Battery_Status",
      jsonEncode({
        'level_percent': level,
        'state': state.toString().split('.').last,
      }),
    );
  } catch (e) {
    debugPrint("Battery Error: $e");
  }
}

Future<void> _sendData(String userId, String dataType, String value) async {
  try {
    final prefs = await SharedPreferences.getInstance();
    String currentId = prefs.getString('user_id') ?? userId;
    await BackgroundServiceHelper.sendToSheet(currentId, dataType, value);
    debugPrint("Data Sent: $dataType");
  } catch (e) {
    debugPrint("Network Error: $e");
  }
}

bool _isToday(DateTime? date) {
  if (date == null) return false;
  final now = DateTime.now();
  return date.year == now.year &&
      date.month == now.month &&
      date.day == now.day;
}
