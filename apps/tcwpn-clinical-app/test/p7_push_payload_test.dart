import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/core/notifications/push_attention_message.dart';
import 'package:r26_ds012_app/domain/contracts/device_token_registration.dart';

void main() {
  test('accepts only minimal attention_event routing payload', () {
    final message = PushAttentionMessage.tryParse({
      'type': 'attention_event',
      'event_id': ' evt-123 ',
    });

    expect(message, isNotNull);
    expect(message!.eventId, 'evt-123');
    expect(message.type, 'attention_event');
  });

  test('rejects unrelated, blank and malformed payloads', () {
    expect(
      PushAttentionMessage.tryParse({'type': 'chat', 'event_id': 'evt-1'}),
      isNull,
    );
    expect(
      PushAttentionMessage.tryParse(
          {'type': 'attention_event', 'event_id': '  '}),
      isNull,
    );
    expect(
      PushAttentionMessage.tryParse({'type': 'attention_event'}),
      isNull,
    );
  });

  test('rejects PHI or clinical detail in push payload', () {
    for (final key in <String>[
      'patient_name',
      'mrn',
      'note_text',
      'current_score',
      'forecast_score',
      'fusion_score',
      'composite_score',
      'model_output',
    ]) {
      expect(
        PushAttentionMessage.tryParse({
          'type': 'attention_event',
          'event_id': 'evt-123',
          key: 'forbidden',
        }),
        isNull,
        reason: 'Push transport must reject $key.',
      );
    }
  });

  test('device token registration serializes routing metadata only', () {
    const registration = DeviceTokenRegistration(
      provider: 'fcm',
      providerProjectId: 'clinanx-primary',
      platform: 'android',
      pushToken: 'token-123',
      active: true,
    );

    expect(registration.toJson(), {
      'provider': 'fcm',
      'provider_project_id': 'clinanx-primary',
      'platform': 'android',
      'push_token': 'token-123',
      'active': true,
    });
    expect(registration.toJson().keys, isNot(contains('patient_name')));
    expect(registration.toJson().keys, isNot(contains('mrn')));
    expect(registration.toJson().keys, isNot(contains('note_text')));
  });
}
