abstract interface class AuthRepository {
  Future<void> validateCurrentSession();
  Future<void> expireCurrentSession();
}
