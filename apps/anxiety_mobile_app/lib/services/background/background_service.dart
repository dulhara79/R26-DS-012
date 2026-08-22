import 'dart:async';
import 'dart:io';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
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
bool _serviceConfigured = false;

Future<void> initializeService() async {
  if (_serviceConfigured) return;
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
        description: 'Aura is working in the background',
        importance: Importance.low,
      ),
    );
    // EMA check-in alerts (high importance = heads-up).
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'ema_channel',
        'Daily Check-ins',
        description: 'Reminders to check how you feel',
        importance: Importance.high,
      ),
    );
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'anxiety_alerts',
        'Anxiety check-ins',
        description: 'Gentle check-ins based on recent readings',
        importance: Importance.max,
      ),
    );
    // GAD-7 weekly alert.
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'gad7_channel',
        'Weekly Check-ins',
        description: 'Reminder for your weekly anxiety check-in',
        importance: Importance.high,
      ),
    );
    // PSS-10 weekly alert.
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'pss_channel',
        'Weekly Check-ins',
        description: 'Reminder for your weekly stress check-in',
        importance: Importance.high,
      ),
    );
  }

  await service.configure(
    androidConfiguration: AndroidConfiguration(
      onStart: onStart,
      // Starting a location foreground service before the first frame, while
      // permission is denied, or from a boot receiver can make Android throw a
      // native SecurityException and terminate the whole process. Configure it
      // here, then start it only through startBackgroundServiceIfPermitted().
      autoStart: false,
      isForegroundMode: true,
      notificationChannelId: ServiceConfig.channelId,
      initialNotificationTitle: 'Aura is running',
      initialNotificationContent: 'Keeping your check-ins ready',
      foregroundServiceNotificationId: ServiceConfig.notificationId,
      autoStartOnBoot: false,
    ),
    iosConfiguration: IosConfiguration(),
  );
  _serviceConfigured = true;
}

/// Starts the long-running collector only while the app is visible and Android
/// has the location permission required by its foreground-service type.
///
/// Android foreground-service failures occur in native code, outside Dart's
/// try/catch boundary, so preventing an invalid start is the reliable fix.
Future<bool> startBackgroundServiceIfPermitted() async {
  await initializeService();

  if (kIsWeb) return false;

  if (Platform.isAndroid) {
    final locationStatus = await Permission.locationWhenInUse.status;
    if (!locationStatus.isGranted) {
      debugPrint(
        'Background Service: location permission is not granted; start skipped.',
      );
      return false;
    }

    final locationServicesEnabled = await Geolocator.isLocationServiceEnabled();
    if (!locationServicesEnabled) {
      debugPrint(
        'Background Service: Android location services are disabled; start skipped.',
      );
      return false;
    }

    if (WidgetsBinding.instance.lifecycleState != AppLifecycleState.resumed) {
      debugPrint(
        'Background Service: app is not resumed; unsafe start skipped.',
      );
      return false;
    }
  }

  final service = FlutterBackgroundService();
  if (await service.isRunning()) return true;
  return service.startService();
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

  // Log every permitted service start explicitly so heartbeat-gap analysis
  // can distinguish a restart from a continuously running collector.
  try {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    final String? uid = prefs.getString('user_id');
    if (uid != null && uid.isNotEmpty) {
      await BackgroundServiceHelper.sendToSheet(
        uid,
        "Service_Restart",
        "Restarted_${DateTime.now().toIso8601String()}",
        immediate: true,
      );
    }
  } catch (e) {
    debugPrint("Service restart logging error: $e");
  }

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
