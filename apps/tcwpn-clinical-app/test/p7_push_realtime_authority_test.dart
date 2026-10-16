import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('Phase 7 push modules and device-token route exist', () {
    const required = [
      'lib/core/notifications/push_attention_message.dart',
      'lib/core/notifications/attention_push_service.dart',
      'lib/core/notifications/firebase_attention_push_service.dart',
      'lib/core/notifications/device_token_registration_coordinator.dart',
      'lib/data/repositories/central_backend_device_token_repository.dart',
      'lib/domain/contracts/device_token_registration.dart',
    ];

    for (final path in required) {
      expect(File(path).existsSync(), isTrue, reason: path);
    }

    final repository = File(
      'lib/data/repositories/central_backend_device_token_repository.dart',
    ).readAsStringSync();
    expect(repository, contains("'/v1/device-tokens'"));
  });

  test('push transport consumes only event identity, never clinical detail', () {
    final source = [
      'lib/core/notifications/push_attention_message.dart',
      'lib/core/notifications/firebase_attention_push_service.dart',
      'lib/core/notifications/device_token_registration_coordinator.dart',
      'lib/domain/contracts/device_token_registration.dart',
    ].map((p) => File(p).readAsStringSync()).join('\n').toLowerCase();

    expect(source, contains('event_id'));
    expect(source, isNot(contains('patient_name')));
    expect(source, isNot(contains('patientname')));
    expect(source, isNot(contains('patient_mrn')));
    expect(source, isNot(contains('patientmrn')));
    expect(source, isNot(contains('note_text')));
    expect(source, isNot(contains('current_score')));
    expect(source, isNot(contains('forecast_score')));
    expect(source, isNot(contains('fusion_score')));
  });

  test('Firebase is messaging-only and two build slots are declared', () {
    final pubspec = File('pubspec.yaml').readAsStringSync();
    final env = File('lib/core/config/env.dart').readAsStringSync();

    expect(pubspec, contains('firebase_core:'));
    expect(pubspec, contains('firebase_messaging:'));
    expect(pubspec, isNot(contains('firebase_database:')));
    expect(pubspec, isNot(contains('cloud_firestore:')));
    expect(pubspec, isNot(contains('firebase_storage:')));
    expect(pubspec, isNot(contains('cloud_functions:')));

    expect(env, contains('FIREBASE_PRIMARY_PROJECT_ID'));
    expect(env, contains('FIREBASE_SECONDARY_PROJECT_ID'));
    expect(env, contains('PUSH_FIREBASE_SLOT'));
  });

  test('polling remains a 30-second independent fallback', () {
    final shell = File('lib/features/shell.dart').readAsStringSync();
    expect(shell, contains('Duration(seconds: 30)'));
    expect(shell, contains('_notificationController.pollOnce()'));
    expect(shell, contains('foregroundEventIds'));
    expect(shell, contains('openedEventIds'));
  });
}
