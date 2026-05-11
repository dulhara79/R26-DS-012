import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import '../models/models.dart';

class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  Future<void> init() async {
    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const DarwinInitializationSettings initializationSettingsIOS =
        DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    const InitializationSettings initializationSettings = InitializationSettings(
      android: initializationSettingsAndroid,
      iOS: initializationSettingsIOS,
    );

    await _notificationsPlugin.initialize(
      initializationSettings,
      onDidReceiveNotificationResponse: (details) {
        // Handle notification tap
      },
    );
  }

  Future<void> showRiskNotification({
    required String patientName,
    required RiskLevel riskLevel,
  }) async {
    final bool isHighRisk = riskLevel == RiskLevel.high || riskLevel == RiskLevel.veryHigh;

    final AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
      isHighRisk ? 'high_risk_channel' : 'normal_risk_channel',
      isHighRisk ? 'High Risk Alerts' : 'Patient Assessments',
      channelDescription: isHighRisk 
          ? 'Urgent alerts for high-risk patients' 
          : 'Standard patient assessment notifications',
      importance: isHighRisk ? Importance.max : Importance.defaultImportance,
      priority: isHighRisk ? Priority.high : Priority.defaultPriority,
      enableVibration: true,
      // In a real app, we would use custom sound files here
      // For now, we use default sounds which vary by importance
      styleInformation: BigTextStyleInformation(''),
    );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    final NotificationDetails platformDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notificationsPlugin.show(
      DateTime.now().millisecond,
      isHighRisk ? '⚠️ High Risk Detected' : 'New Assessment',
      'Patient $patientName has been assessed as ${riskLevel.label}.',
      platformDetails,
    );
  }
}
