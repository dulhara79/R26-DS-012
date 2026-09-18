import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/core/notifications/device_token_registration_coordinator.dart';
import 'package:r26_ds012_app/domain/contracts/device_token_registration.dart';
import 'package:r26_ds012_app/domain/repositories/device_token_repository.dart';

void main() {
  test('register sends active token with provider project routing', () async {
    final repo = _Repo();
    final coordinator = DeviceTokenRegistrationCoordinator(
      repository: repo,
      providerProjectId: 'primary-project',
      platform: 'android',
    );

    await coordinator.register('token-a');

    expect(repo.writes.single, const DeviceTokenRegistration(
      provider: 'fcm',
      providerProjectId: 'primary-project',
      platform: 'android',
      pushToken: 'token-a',
      active: true,
    ));
  });

  test('rotation deactivates previous token before activating replacement', () async {
    final repo = _Repo();
    final coordinator = DeviceTokenRegistrationCoordinator(
      repository: repo,
      providerProjectId: 'primary-project',
      platform: 'android',
    );

    await coordinator.register('token-a');
    await coordinator.rotate('token-b');

    expect(repo.writes.map((e) => '${e.pushToken}:${e.active}').toList(), [
      'token-a:true',
      'token-a:false',
      'token-b:true',
    ]);
  });

  test('revoke deactivates last registered token and is idempotent', () async {
    final repo = _Repo();
    final coordinator = DeviceTokenRegistrationCoordinator(
      repository: repo,
      providerProjectId: 'primary-project',
      platform: 'ios',
    );

    await coordinator.register('token-a');
    await coordinator.revoke();
    await coordinator.revoke();

    expect(repo.writes.map((e) => '${e.pushToken}:${e.active}').toList(), [
      'token-a:true',
      'token-a:false',
    ]);
  });
}

class _Repo implements DeviceTokenRepository {
  final List<DeviceTokenRegistration> writes = [];

  @override
  Future<void> upsert(DeviceTokenRegistration registration) async {
    writes.add(registration);
  }
}
