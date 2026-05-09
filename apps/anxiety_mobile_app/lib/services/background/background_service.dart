import 'dart:async';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:battery_plus/battery_plus.dart';

import '../background_service_helper.dart';
import 'service_config.dart';
import 'data_collector.dart';
import 'sensor_listener.dart';
import 'daily_reminder.dart';

/// Called once from main.dart / login_page.dart to register Android notification
/// channels and configure the background service.  Must run in the UI isolate.
Future<void> initializeService() async {
  final service = FlutterBackgroundService();

  // ── Create ALL notification channels before configuring the service ──────
  // Channels must exist before any notification is shown on them.
  final FlutterLocalNotificationsPlugin flnp =
      FlutterLocalNotificationsPlugin();
  final AndroidFlutterLocalNotificationsPlugin? androidPlugin = flnp
      .resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin
      >();

  if (androidPlugin != null) {
    // Foreground-service persistent notification (low importance = no sound).
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        ServiceConfig.channelId,
        ServiceConfig.channelName,
        description: 'Running background research tasks',
        importance: Importance.low,
      ),
    );
    // EMA check-in alerts (high importance = heads-up).
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'ema_channel',
        'Daily Check-ins',
        description: 'Scheduled mood and anxiety ratings',
        importance: Importance.high,
      ),
    );
    // GAD-7 weekly alert.
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'gad7_channel',
        'Weekly Assessments',
        description: 'Weekly GAD-7 clinical questionnaires',
        importance: Importance.high,
      ),
    );
    // PSS-10 weekly alert.
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'pss_channel',
        'Monthly Assessments',
        description: 'Monthly PSS-10 stress scale assessments',
        importance: Importance.high,
      ),
    );
  }

  await service.configure(
    androidConfiguration: AndroidConfiguration(
      onStart: onStart,
      autoStart: true,
      isForegroundMode: true,
      notificationChannelId: ServiceConfig.channelId,
      initialNotificationTitle: 'Research Active',
      initialNotificationContent: 'Monitoring in background…',
      foregroundServiceNotificationId: ServiceConfig.notificationId,
      autoStartOnBoot: true,
    ),
    iosConfiguration: IosConfiguration(),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// BACKGROUND ISOLATE ENTRY POINT
// ─────────────────────────────────────────────────────────────────────────────

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  DartPluginRegistrant.ensureInitialized();

  // ── Mark as background isolate FIRST — before any sendToSheet call ────────
  // This routes queue writes to 'offline_queue_bg' instead of 'offline_queue_main'.
  BackgroundServiceHelper.isMainIsolate = false;

  debugPrint("🔋 Background Service: onStart — initialising…");

  final prefs = await SharedPreferences.getInstance();
  // Force reload so we see the user_id written by the UI isolate.
  await prefs.reload();

  final String? userId = prefs.getString('user_id');
  if (userId == null || userId.isEmpty) {
    debugPrint("Background Service: no user_id — stopping.");
    service.stopSelf();
    return;
  }

  // ── 1. Connectivity — retry queued data when network returns ──────────────
  try {
    await BackgroundServiceHelper.retryOfflineQueue();
    Connectivity().onConnectivityChanged.listen((event) async {
      final bool connected = event is List
          ? (event as List).any((r) => r != ConnectivityResult.none)
          : event != ConnectivityResult.none;
      if (connected) {
        debugPrint("🌐 Network restored — retrying queue…");
        await BackgroundServiceHelper.retryOfflineQueue();
      }
    });
  } catch (e) {
    debugPrint("Connectivity setup error: $e");
  }

  // ── 2. Service control messages ────────────────────────────────────────────
  if (service is AndroidServiceInstance) {
    service
        .on('setAsForeground')
        .listen((_) => service.setAsForegroundService());
    service
        .on('setAsBackground')
        .listen((_) => service.setAsBackgroundService());
  }
  service.on('stopService').listen((_) => service.stopSelf());

  // ── 3. Battery monitor ─────────────────────────────────────────────────────
  try {
    final battery = Battery();
    battery.onBatteryStateChanged.listen((BatteryState state) async {
      try {
        final int level = await battery.batteryLevel;
        await prefs.setInt('last_battery_level', level);
        if (level <= 15 && state == BatteryState.discharging) {
          await BackgroundServiceHelper.sendToSheet(
            userId,
            "Critical_Battery_Warning",
            "Level: $level%",
            immediate: true,
          );
        }
      } catch (e) {
        debugPrint("Battery event error: $e");
      }
    });
  } catch (e) {
    debugPrint("Battery monitor setup error: $e");
  }

  // ── 4. Real-time sensors ──────────────────────────────────────────────────
  try {
    SensorListener().startListening(userId);
  } catch (e) {
    debugPrint("SensorListener setup error: $e");
  }

  // ── 5. Periodic 15-minute data collection ─────────────────────────────────
  unawaited(
    DataCollector.collectAndSync(
      userId,
    ).catchError((e) => debugPrint("Initial DataCollector error: $e")),
  );
  Timer.periodic(const Duration(minutes: 15), (_) async {
    try {
      await DataCollector.collectAndSync(userId);
    } catch (e) {
      debugPrint("Periodic data collection error: $e");
    }
  });

  // ── 6. Background notification plugin ─────────────────────────────────────
  //
  // CRITICAL: The background isolate is a separate Dart VM entry point.
  // It cannot share the FlutterLocalNotificationsPlugin instance created in
  // initializeService() (which runs in the UI isolate).  We must create and
  // initialise a fresh instance here.
  //
  // Channels were already registered by initializeService() — Android persists
  // them per-app, so we only need to call initialize() here, NOT create channels.
  final FlutterLocalNotificationsPlugin bgPlugin =
      FlutterLocalNotificationsPlugin();

  final bool? pluginReady = await bgPlugin.initialize(
    const InitializationSettings(
      android: AndroidInitializationSettings('@mipmap/launcher_icon'),
    ),
  );
  debugPrint(
    "Background Service: notification plugin init result = $pluginReady",
  );
  debugPrint("EMA_DEBUG: bg_plugin_initialized=$pluginReady");

  // ── 7. 1-minute reminder timer ────────────────────────────────────────────
  //
  // DailyReminder.checkAndShow() manages all throttling internally.
  // It reads prefs.reload() on every tick so time-changes from Settings
  // are picked up immediately.
  Timer.periodic(const Duration(minutes: 1), (_) async {
    debugPrint("EMA_DEBUG: reminder_timer_tick");
    try {
      await DailyReminder.checkAndShow(bgPlugin);
    } catch (e) {
      debugPrint("Reminder timer error: $e");
      debugPrint("EMA_DEBUG: reminder_timer_error error=$e");
    }
  });

  debugPrint("✅ Background Service: fully started for user '$userId'.");
}

// Silences the unawaited-future lint for intentional fire-and-forget.
void unawaited(Future<void> future) {}
