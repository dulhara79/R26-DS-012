import '../contracts/device_token_registration.dart';

abstract interface class DeviceTokenRepository {
  Future<void> upsert(DeviceTokenRegistration registration);
}
