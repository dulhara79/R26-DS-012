import '../../core/config/env.dart';
import '../../domain/contracts/device_token_registration.dart';
import '../../domain/repositories/device_token_repository.dart';
import '../api/api_client.dart';

class CentralBackendDeviceTokenRepository implements DeviceTokenRepository {
  CentralBackendDeviceTokenRepository([ApiClient? api])
      : _api = api ?? ApiClient(Env.backendBase);

  static const String path = '/v1/device-tokens';
  final ApiClient _api;

  @override
  Future<void> upsert(DeviceTokenRegistration registration) async {
    await _api.post(path, registration.toJson());
  }
}
