import 'dart:async';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';

import '../../data/repositories/central_backend_device_token_repository.dart';
import 'attention_push_service.dart';
import 'device_token_registration_coordinator.dart';
import 'push_attention_message.dart';
import 'push_firebase_config.dart';

class FirebaseAttentionPushService implements AttentionPushService {
  FirebaseAttentionPushService._();

  static final FirebaseAttentionPushService instance =
      FirebaseAttentionPushService._();

  final StreamController<String> _foreground = StreamController.broadcast();
  final StreamController<String> _opened = StreamController.broadcast();

  FirebaseMessaging? _messaging;
  DeviceTokenRegistrationCoordinator? _coordinator;
  StreamSubscription<String>? _refreshSub;
  String? _initialOpenedEventId;
  bool _initialized = false;

  @override
  Stream<String> get foregroundEventIds => _foreground.stream;

  @override
  Stream<String> get openedEventIds => _opened.stream;

  @override
  Future<void> initialize() async {
    if (_initialized || !PushFirebaseConfig.isConfigured) return;

    await Firebase.initializeApp(options: PushFirebaseConfig.options);
    final messaging = FirebaseMessaging.instance;
    _messaging = messaging;

    _coordinator = DeviceTokenRegistrationCoordinator(
      repository: CentralBackendDeviceTokenRepository(),
      providerProjectId: PushFirebaseConfig.options.projectId,
      platform: PushFirebaseConfig.platformName,
    );

    FirebaseMessaging.onMessage.listen((message) {
      final parsed = PushAttentionMessage.tryParse(message.data);
      if (parsed != null && !_foreground.isClosed) {
        _foreground.add(parsed.eventId);
      }
    });

    FirebaseMessaging.onMessageOpenedApp.listen((message) {
      final parsed = PushAttentionMessage.tryParse(message.data);
      if (parsed != null && !_opened.isClosed) {
        _opened.add(parsed.eventId);
      }
    });

    final initial = await messaging.getInitialMessage();
    final parsedInitial = initial == null
        ? null
        : PushAttentionMessage.tryParse(initial.data);
    _initialOpenedEventId = parsedInitial?.eventId;

    _initialized = true;
  }

  @override
  Future<void> activateAuthenticatedSession() async {
    await initialize();
    final messaging = _messaging;
    final coordinator = _coordinator;
    if (messaging == null || coordinator == null) return;

    await messaging.requestPermission(alert: true, badge: true, sound: true);

    final token = await messaging.getToken();
    if (token != null && token.trim().isNotEmpty) {
      await coordinator.register(token);
    }

    await _refreshSub?.cancel();
    _refreshSub = messaging.onTokenRefresh.listen((token) {
      unawaited(coordinator.rotate(token));
    });
  }

  @override
  Future<void> revoke() async {
    try {
      await _coordinator?.revoke();
    } catch (_) {
      // Sign-out and clinical access must not be blocked by push cleanup.
    }
  }

  @override
  Future<String?> takeInitialOpenedEventId() async {
    final value = _initialOpenedEventId;
    _initialOpenedEventId = null;
    return value;
  }
}

final AttentionPushService attentionPushService =
    PushFirebaseConfig.isConfigured
        ? FirebaseAttentionPushService.instance
        : const DisabledAttentionPushService();
