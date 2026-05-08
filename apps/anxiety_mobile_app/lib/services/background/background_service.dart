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

Future<void> initializeService() async {
  final service = FlutterBackgroundService();

  const AndroidNotificationChannel channel = AndroidNotificationChannel(
    ServiceConfig.channelId,
    ServiceConfig.channelName,
    description: 'Running background research tasks',
    importance: Importance.low,
  );

  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  final androidPlugin = flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin
      >();

  if (androidPlugin != null) {
    await androidPlugin.createNotificationChannel(channel);
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'ema_channel',
        'Daily Check-ins',
        description: 'Scheduled mood and anxiety ratings',
        importance: Importance.high,
      ),
    );
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'gad7_channel',
        'Weekly Assessments',
        description: 'Weekly GAD-7 clinical questionnaires',
        importance: Importance.high,
      ),
    );
    await androidPlugin.createNotificationChannel(
      const AndroidNotificationChannel(
        'pss_channel',
        'Monthly Assessments',
        description: 'Monthly Perceived Stress Scale (PSS-10) assessments',
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
      initialNotificationContent: 'Collecting anonymous usage data...',
      foregroundServiceNotificationId: ServiceConfig.notificationId,
      autoStartOnBoot: true,
    ),
    iosConfiguration: IosConfiguration(),
  );
}

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  DartPluginRegistrant.ensureInitialized();
  
  debugPrint("🔋 Background Service: onStart beginning...");
  
  final prefs = await SharedPreferences.getInstance();
  String? userId = prefs.getString('user_id');
  
  if (userId == null || userId.isEmpty) {
    debugPrint("Background Service: No User ID found. Service will not start.");
    service.stopSelf();
    return;
  }

  // 1. Setup Connectivity & Offline Sync
  try {
    await BackgroundServiceHelper.retryOfflineQueue();
    Connectivity().onConnectivityChanged.listen((ConnectivityResult result) async {
      if (result != ConnectivityResult.none) {
        debugPrint("🌐 Connectivity Restored: Retrying sync...");
        await BackgroundServiceHelper.retryOfflineQueue();
      }
    });
  } catch (e) {
    debugPrint('Connectivity Setup Error: $e');
  }

  // 2. Setup Background Notifications
  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();
  
  await flutterLocalNotificationsPlugin.initialize(
    const InitializationSettings(
      android: AndroidInitializationSettings('ic_launcher'),
    ),
  );

  if (service is AndroidServiceInstance) {
    service.on('setAsForeground').listen((event) => service.setAsForegroundService());
    service.on('setAsBackground').listen((event) => service.setAsBackgroundService());
  }
  service.on('stopService').listen((event) => service.stopSelf());

  // ── REAL-TIME: BATTERY MONITOR ──────────────────────────
  try {
    final battery = Battery();
    battery.onBatteryStateChanged.listen((BatteryState state) async {
      try {
        final level = await battery.batteryLevel;
        await prefs.setInt('last_battery_level', level);
        if (level <= 15 && state == BatteryState.discharging) {
          await BackgroundServiceHelper.sendToSheet(userId, "Critical_Battery_Warning", "Level: $level%");
        }
      } catch(e) {
        debugPrint("Battery Event Error: $e");
      }
    });
  } catch (e) {
    debugPrint("Battery Monitor Setup Error: $e");
  }

  // 3. Start Real-Time Sensors (Screen, Motion)
  try {
    final sensorListener = SensorListener();
    sensorListener.startListening(userId);
  } catch (e) {
    debugPrint("SensorListener Setup Error: $e");
  }

  // 4. Start Periodic Data Collection (Every 15 Minutes)
  // Heartbeat immediately on start
  DataCollector.collectAndSync(userId);
  
  Timer.periodic(const Duration(minutes: 15), (timer) async {
    debugPrint("⏰ Periodic Task: Triggering 15m collection...");
    try {
      if (service is AndroidServiceInstance) {
        if (await service.isForegroundService()) {
          // FIX: Use static notification text — do NOT display sync times to the user
          flutterLocalNotificationsPlugin.show(
            ServiceConfig.notificationId,
            'Research Active',
            'Collecting anonymous usage data...',
            const NotificationDetails(
              android: AndroidNotificationDetails(
                ServiceConfig.channelId,
                ServiceConfig.channelName,
                icon: 'ic_bg_service_small',
                ongoing: true,
                importance: Importance.low,
              ),
            ),
          );
        }
      }
      await DataCollector.collectAndSync(userId);
    } catch (e) {
      debugPrint("Periodic Timer Error: $e");
    }
  });

  // 5. Start Daily Rating Checker (Every 1 Minute)
  Timer.periodic(const Duration(minutes: 1), (timer) async {
    try {
      await DailyReminder.checkAndShow(flutterLocalNotificationsPlugin);
    } catch(e) {
      debugPrint("Reminder Timer Error: $e");
    }
  });
}
