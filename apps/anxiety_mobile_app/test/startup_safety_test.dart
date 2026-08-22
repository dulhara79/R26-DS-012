import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('background collector cannot auto-start from an unsafe context', () {
    final service = File(
      'lib/services/background/background_service.dart',
    ).readAsStringSync();
    final dashboard = File('lib/pages/dashboard_page.dart').readAsStringSync();
    final manifest = File(
      'android/app/src/main/AndroidManifest.xml',
    ).readAsStringSync();
    final powerReceiver = File(
      'android/app/src/main/kotlin/com/example/anxiety_mobile_app/PowerConnectedReceiver.kt',
    );

    expect(service, contains('autoStart: false'));
    expect(service, contains('autoStartOnBoot: false'));
    expect(service, contains('Permission.locationWhenInUse.status'));
    expect(service, contains('Geolocator.isLocationServiceEnabled()'));
    expect(service, contains('AppLifecycleState.resumed'));
    expect(
      dashboard,
      isNot(contains('FlutterBackgroundService().startService()')),
    );
    expect(manifest, isNot(contains('PowerConnectedReceiver')));
    expect(powerReceiver.existsSync(), isFalse);
    expect(manifest, contains('android:foregroundServiceType="location"'));
    expect(manifest, isNot(contains('dataSync')));
    expect(manifest, contains('android:exported="false"'));
  });
}
