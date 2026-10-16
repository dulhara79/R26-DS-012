// lib/main.dart

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'core/config/env.dart';
import 'core/design/theme.dart';
import 'core/notifications/firebase_attention_push_service.dart';
import 'core/notifications/flutter_attention_notification_gateway.dart';
import 'core/security/secure_http.dart';
import 'core/design/tokens.dart';
import 'data/api/session.dart';
import 'data/local/consent_store.dart';
import 'data/local/stores.dart';
import 'features/auth/login_screen.dart';
import 'features/consent/consent_gate_screen.dart';
import 'features/shell.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
    statusBarBrightness: Brightness.light,
  ));

  try {
    await attentionNotificationGateway.initialize();
  } catch (_) {
    // Local notification failure must never block clinical access.
  }

  // Push is optional at runtime. Missing/invalid Firebase build configuration
  // degrades to the server-backed Activity view plus polling fallback.
  try {
    await attentionPushService.initialize();
  } catch (_) {
    // Never block consent/sign-in because a push provider is unavailable.
  }

  Session.installBeforeSignOutHook(attentionPushService.revoke);
  unawaited(SecureHttp.verifyAll());

  final consented = await ConsentStore.hasValidConsent();
  final signedIn = consented && await SecureStore.hasSession();

  if (signedIn) {
    Session.set(
      token: await SecureStore.token() ?? '',
      clinicianId: await SecureStore.clinicianId(),
      expiresAt: await SecureStore.expiresAt(),
    );
  }

  runApp(ClinAnxApp(consented: consented, signedIn: signedIn));
}

class ClinAnxApp extends StatefulWidget {
  final bool consented;
  final bool signedIn;

  const ClinAnxApp({
    super.key,
    required this.consented,
    required this.signedIn,
  });

  @override
  State<ClinAnxApp> createState() => _ClinAnxAppState();
}

class _ClinAnxAppState extends State<ClinAnxApp> {
  late bool _consented = widget.consented;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ClinAnx',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      builder: (context, child) => ColoredBox(color: Ds.canvas, child: child!),
      home: !_consented
          ? ConsentGateScreen(
              appVersion: Env.appVersion,
              onAccepted: () => setState(() => _consented = true),
            )
          : (widget.signedIn ? const AppShell() : const LoginScreen()),
    );
  }
}
