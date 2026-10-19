import 'dart:async';
import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import '../../data/local/attention_notification_store.dart';
import 'attention_notification_gateway.dart';

class FlutterAttentionNotificationGateway
    implements AttentionNotificationGateway {
  FlutterAttentionNotificationGateway({
    FlutterLocalNotificationsPlugin? plugin,
    AttentionNotificationStore? tapStore,
  })  : _plugin = plugin ?? FlutterLocalNotificationsPlugin(),
        _tapStore = tapStore ??
            const SharedPreferencesAttentionNotificationStore(
              deliveryScope: '__routing__',
            );

  final FlutterLocalNotificationsPlugin _plugin;
  final AttentionNotificationStore _tapStore;
  final StreamController<String> _opened = StreamController<String>.broadcast();
  bool _initialized = false;

  @override
  Stream<String> get openedEventIds => _opened.stream;

  @override
  Future<void> initialize() async {
    if (_initialized) return;

    const settings = InitializationSettings(
      android: AndroidInitializationSettings('@mipmap/launcher_icon'),
      iOS: DarwinInitializationSettings(
        requestAlertPermission: false,
        requestBadgePermission: false,
        requestSoundPermission: false,
      ),
    );

    await _plugin.initialize(
      settings,
      onDidReceiveNotificationResponse: (response) async {
        await _recordOpen(response.payload);
      },
    );

    final launch = await _plugin.getNotificationAppLaunchDetails();
    if (launch?.didNotificationLaunchApp ?? false) {
      await _recordOpen(launch?.notificationResponse?.payload, emit: false);
    }

    _initialized = true;
  }

  Future<void> _recordOpen(String? eventId, {bool emit = true}) async {
    final id = eventId?.trim() ?? '';
    if (id.isEmpty) return;
    await _tapStore.savePendingOpen(id);
    if (emit && !_opened.isClosed) _opened.add(id);
  }

  @override
  Future<bool> requestPermission() async {
    final android = _plugin.resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin>();
    if (android != null) {
      return await android.requestNotificationsPermission() ?? false;
    }

    final ios = _plugin.resolvePlatformSpecificImplementation<
        IOSFlutterLocalNotificationsPlugin>();
    if (ios != null) {
      return await ios.requestPermissions(alert: true, badge: true, sound: true) ??
          false;
    }

    return true;
  }

  @override
  Future<void> showAttentionEvent(String eventId) async {
    final id = eventId.trim();
    if (id.isEmpty) return;

    const details = NotificationDetails(
      android: AndroidNotificationDetails(
        'attention_events',
        'Attention events',
        channelDescription: 'Server-owned ClinAnx attention events',
        importance: Importance.high,
        priority: Priority.high,
      ),
      iOS: DarwinNotificationDetails(),
    );

    await _plugin.show(
      _notificationId(id),
      'ClinAnx',
      'New attention event available.',
      details,
      payload: id,
    );
  }

  int _notificationId(String eventId) {
    final digest = sha256.convert(utf8.encode(eventId)).bytes;
    final value = (digest[0] << 24) |
        (digest[1] << 16) |
        (digest[2] << 8) |
        digest[3];
    return value & 0x7fffffff;
  }
}

final AttentionNotificationGateway attentionNotificationGateway =
    FlutterAttentionNotificationGateway();
