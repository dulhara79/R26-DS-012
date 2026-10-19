import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';

import '../config/env.dart';

class PushFirebaseConfig {
  const PushFirebaseConfig._();

  static bool get isConfigured => Env.hasPushFirebase;

  static String get slotName =>
      Env.useSecondaryFirebase ? 'secondary' : 'primary';

  static FirebaseOptions get options {
    if (!isConfigured) {
      throw StateError('Firebase push is not configured for this build.');
    }

    return FirebaseOptions(
      apiKey: Env.activeFirebaseApiKey,
      appId: Env.activeFirebaseAppId,
      messagingSenderId: Env.activeFirebaseSenderId,
      projectId: Env.activeFirebaseProjectId,
      iosBundleId: defaultTargetPlatform == TargetPlatform.iOS &&
              Env.activeFirebaseIosBundleId.isNotEmpty
          ? Env.activeFirebaseIosBundleId
          : null,
    );
  }

  static String get platformName => switch (defaultTargetPlatform) {
        TargetPlatform.android => 'android',
        TargetPlatform.iOS => 'ios',
        TargetPlatform.macOS => 'macos',
        TargetPlatform.windows => 'windows',
        TargetPlatform.linux => 'linux',
        TargetPlatform.fuchsia => 'fuchsia',
      };
}
