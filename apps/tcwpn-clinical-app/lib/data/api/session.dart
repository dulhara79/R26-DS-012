// lib/data/api/session.dart

import '../local/clinician_storage_scope.dart';
import '../local/stores.dart';

typedef SessionSignOutHook = Future<void> Function();

class Session {
  Session._();

  static String? _token;
  static String? _clinicianId;
  static DateTime? _expiresAt;
  static SessionSignOutHook? _beforeSignOut;

  static String? get token => _token;
  static String? get clinicianId => _clinicianId;
  static DateTime? get expiresAt => _expiresAt;
  static bool get isExpired =>
      _expiresAt != null && !DateTime.now().toUtc().isBefore(_expiresAt!);
  static bool get isActive => (_token ?? '').isNotEmpty && !isExpired;

  static void set({required String token, String? clinicianId, DateTime? expiresAt}) {
    _token = token;
    _clinicianId = clinicianId;
    _expiresAt = expiresAt?.toUtc();

    final id = clinicianId?.trim() ?? '';
    if (id.isEmpty) {
      ClinicianStorageScope.clear();
    } else {
      ClinicianStorageScope.bind(id);
    }
  }

  static void installBeforeSignOutHook(SessionSignOutHook? hook) {
    _beforeSignOut = hook;
  }

  static void clear() {
    _token = null;
    _clinicianId = null;
    _expiresAt = null;
    ClinicianStorageScope.clear();
  }

  static Future<void> signOut() async {
    final hook = _beforeSignOut;
    if (hook != null) {
      try {
        await hook();
      } catch (_) {
        // Push revocation is best-effort. Never trap a clinician in a session
        // because the network/push provider is unavailable.
      }
    }
    await SecureStore.signOut();
    clear();
  }
}
