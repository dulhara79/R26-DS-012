import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/local/attention_notification_store.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  test('delivery dedupe survives store recreation for the same clinician scope', () async {
    const first = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR001');
    await first.markDelivered('evt-1');

    const second = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR001');
    expect(await second.wasDelivered('evt-1'), isTrue);
  });

  test('delivery dedupe is isolated between clinician scopes', () async {
    const dr1 = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR001');
    const dr2 = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR002');

    await dr1.markDelivered('evt-1');

    expect(await dr1.wasDelivered('evt-1'), isTrue);
    expect(await dr2.wasDelivered('evt-1'), isFalse);
  });

  test('pending notification open is consumed exactly once', () async {
    const store = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR001');
    await store.savePendingOpen('evt-9');

    expect(await store.takePendingOpen(), 'evt-9');
    expect(await store.takePendingOpen(), isNull);
  });
}
