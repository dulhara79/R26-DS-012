import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/core/notifications/push_attention_message.dart';
import 'package:r26_ds012_app/domain/contracts/device_token_registration.dart';

void main() {
  test('accepts only attention_event payload with nonblank event_id', () {
    final message = PushAttentionMessage.tryParse({
      'type': 'attention_event',
      'event_id': ' evt-123 ',
      'severity': 'high',
      'patient_name': 'must-not-be-used',
      'mrn': 'must-not-be-used',
    });

    expect(message, isNotNull);
    expect(message!.eventId, 'evt-123');
    expect(message.type, 'attention_event');
  });

  test('rejects unrelated, blank and malformed payloads', () {
    expect(PushAttentionMessage.tryParse({'type': 'chat', 'event_id': 'evt-1'}), isNull);
    expect(PushAttentionMessage.tryParse({'type': 'attention_event', 'event_id': '  '}), isNull);
    expect(PushAttentionMessage.tryParse({'type': 'attention_event'}), isNull);
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
