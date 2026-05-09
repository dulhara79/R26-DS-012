import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter/material.dart';

class NotificationHelper {
  static final FlutterLocalNotificationsPlugin plugin =
      FlutterLocalNotificationsPlugin();

  // Called when the user taps the notification
  static void Function(String?)? onNotificationClick;

  static Future<void> init() async {
    try {
      const AndroidInitializationSettings androidInit =
          AndroidInitializationSettings('@mipmap/launcher_icon');

      const InitializationSettings initSettings = InitializationSettings(
        android: androidInit,
      );

      await plugin.initialize(
        initSettings,
        onDidReceiveNotificationResponse: (response) {
          // Route to the app via callback
          if (onNotificationClick != null) onNotificationClick!(response.payload);
        },
      );
    } catch (e, st) {
      // Don't rethrow — log and allow app to continue.
      debugPrint('Notification plugin init failed: $e');
      debugPrint('$st');
    }
  }

  static Future<void> showRatingNotification() async {
    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
          'rating_channel',
          'Daily Rating',
          importance: Importance.high,
          priority: Priority.high,
          showWhen: false,
        );

    const NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );

    await plugin.show(
      999,
      'How was your stress today?',
      'Tap to rate 0–5',
      details,
      payload: 'stress_rating',
    );
  }
}
