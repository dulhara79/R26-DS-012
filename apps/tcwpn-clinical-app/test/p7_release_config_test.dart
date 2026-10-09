import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'AuthService uses the Central Backend and hides unsupported self-service',
    () {
      final source = File('lib/data/api/auth_service.dart').readAsStringSync();

      expect(source, isNot(contains("String.fromEnvironment('AUTH_BASE'")));
      expect(source, isNot(contains("String.fromEnvironment('AUTH_SALT'")));
      expect(source, isNot(contains("String.fromEnvironment('AUTH_LOCAL'")));
      expect(source, contains('ApiClient(Env.backendBase)'));
      expect(source, contains('supportsSelfService => false'));
      expect(source, isNot(contains("'/auth/register'")));
      expect(source, isNot(contains("'/auth/forgot-password'")));
      expect(source, contains('Env.authSalt'));
      expect(source, contains('Env.authLocalAccounts'));
    },
  );

  test('demo clinical fixtures require explicit opt-in', () {
    final source = File('lib/core/config/env.dart').readAsStringSync();

    expect(
      source,
      contains("bool.fromEnvironment('DEMO_DATA', defaultValue: false)"),
      reason: 'A release build must not seed demo patients when a define is omitted.',
    );
    expect(source, contains("!bool.fromEnvironment('dart.vm.product')"));
  });

  test(
    'release build requires a protected signing key and P0 uses polling',
    () {
      final gradle = File('android/app/build.gradle.kts').readAsStringSync();
      final push = File(
        'lib/core/notifications/firebase_attention_push_service.dart',
      ).readAsStringSync();
      expect(gradle, contains('key.properties'));
      expect(gradle, contains('releaseTaskRequested && !hasReleaseKey'));
      expect(gradle, isNot(contains('signingConfigs.getByName("debug")')));
      expect(push, contains('kDebugMode &&'));
      expect(push, contains("ENABLE_EXPERIMENTAL_PUSH"));
    },
  );
}
