import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/api/session.dart';
import 'package:r26_ds012_app/data/local/stores.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    FlutterSecureStorage.setMockInitialValues({});
    Session.clear();
  });

  test('sign out clears both persisted and in-memory session state', () async {
    await SecureStore.saveSession(
      clinicianId: 'DR001',
      clinicianName: 'Dr Test',
      token: 'session-token',
    );
    Session.set(token: 'session-token', clinicianId: 'DR001');

    await Session.signOut();

    expect(Session.isActive, isFalse);
    expect(Session.token, isNull);
    expect(Session.clinicianId, isNull);
    expect(await SecureStore.hasSession(), isFalse);
    expect(await SecureStore.token(), isNull);
    expect(await SecureStore.clinicianId(), isNull);
    expect(await SecureStore.clinicianName(), isNull);
  });
}
