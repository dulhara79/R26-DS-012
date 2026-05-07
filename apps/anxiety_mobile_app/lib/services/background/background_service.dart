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

  await flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin
      >()
      ?.createNotificationChannel(channel);

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
  final prefs = await SharedPreferences.getInstance();
  String? userId = prefs.getString('user_id');
  if (userId == null || userId.isEmpty) {
    debugPrint("Background Service: No User ID found. Service will not start.");
    service.stopSelf();
    return;
  }

  // 1. Initial Retry Logic
  try {
    await BackgroundServiceHelper.retryOfflineQueue();
    Connectivity().onConnectivityChanged.listen((result) async {
      if (result != ConnectivityResult.none) {
        await BackgroundServiceHelper.retryOfflineQueue();
      }
    });
  } catch (e) {
    debugPrint('Connectivity Setup Error: $e');
  }

  // 2. Setup Foreground Notification
  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  if (service is AndroidServiceInstance) {
    service
        .on('setAsForeground')
        .listen((event) => service.setAsForegroundService());
    service
        .on('setAsBackground')
        .listen((event) => service.setAsBackgroundService());
  }
  service.on('stopService').listen((event) => service.stopSelf());

  // ── REAL-TIME: BATTERY MONITOR ──────────────────────────
  try {
    final battery = Battery();
    battery.onBatteryStateChanged.listen((BatteryState state) async {
      final level = await battery.batteryLevel;
      await prefs.setInt('last_battery_level', level);
      if (level <= 15 && state == BatteryState.discharging) {
        await BackgroundServiceHelper.sendToSheet(
          userId,
          "Critical_Battery_Warning",
          "Level: $level%",
        );
      }
    });
  } catch (e) {
    debugPrint("Battery Monitor Error: $e");
  }

  // 3. Start Real-Time Sensors
  final sensorListener = SensorListener();
  sensorListener.startListening(userId);

  // 4. Start Periodic Data Collection (Every 15 Minutes)
  Timer.periodic(const Duration(minutes: 15), (timer) async {
    if (service is AndroidServiceInstance) {
      if (await service.isForegroundService()) {
        flutterLocalNotificationsPlugin.show(
          ServiceConfig.notificationId,
          'Research Active',
          'Last Sync: ${DateTime.now().hour}:${DateTime.now().minute}',
          const NotificationDetails(
            android: AndroidNotificationDetails(
              ServiceConfig.channelId,
              ServiceConfig.channelName,
              icon: 'ic_bg_service_small',
              ongoing: true,
            ),
          ),
        );
      }
    }
    await DataCollector.collectAndSync(userId);
  });

  // 5. Start Daily Rating Checker (Every 1 Minute)
  Timer.periodic(const Duration(minutes: 1), (timer) async {
    await DailyReminder.checkAndShow(flutterLocalNotificationsPlugin);
  });
}
