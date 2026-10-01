import '../../domain/contracts/device_token_registration.dart';
import '../../domain/repositories/device_token_repository.dart';

class DeviceTokenRegistrationCoordinator {
  DeviceTokenRegistrationCoordinator({
    required this.repository,
    required this.providerProjectId,
    required this.platform,
  });

  final DeviceTokenRepository repository;
  final String providerProjectId;
  final String platform;

  String? _registeredToken;

  Future<void> register(String token) async {
    final normalized = token.trim();
    if (normalized.isEmpty) return;
    await repository.upsert(_registration(normalized, active: true));
    _registeredToken = normalized;
  }

  Future<void> rotate(String token) async {
    final normalized = token.trim();
    if (normalized.isEmpty || normalized == _registeredToken) return;

    final previous = _registeredToken;
    if (previous != null) {
      try {
        await repository.upsert(_registration(previous, active: false));
      } catch (_) {
        // A stale token can also be expired server-side. Do not prevent the
        // replacement token from becoming active because cleanup failed.
      }
    }

    await repository.upsert(_registration(normalized, active: true));
    _registeredToken = normalized;
  }

  Future<void> revoke() async {
    final token = _registeredToken;
    if (token == null) return;
    await repository.upsert(_registration(token, active: false));
    _registeredToken = null;
  }

  DeviceTokenRegistration _registration(String token, {required bool active}) =>
      DeviceTokenRegistration(
        provider: 'fcm',
        providerProjectId: providerProjectId,
        platform: platform,
        pushToken: token,
        active: active,
      );
}
