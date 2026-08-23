// lib/data/api/session.dart
//
// The clinician's bearer token, held in memory for the life of the process.
//
// WHY THIS EXISTS
// ---------------
// ApiClient was sending `Env.hfToken` — the HuggingFace token baked in at build
// time — as its Authorization header. That is a deploy credential, not a user
// credential. The clinician's session JWT sat in SecureStore and was never
// attached to anything, so /predict arrived unauthenticated and the Space
// rejected it with 401.
//
// Reading SecureStore on every request would mean a platform-channel round trip
// per call, so the token is cached here and kept in step at three points:
//
//   • app start   — primed from SecureStore
//   • sign-in     — set from the login response
//   • sign-out    — cleared
//
// It is deliberately NOT persisted here. SecureStore remains the only place the
// token is written to disk.

class Session {
  Session._();

  static String? _token;
  static String? _clinicianId;

  static String? get token => _token;
  static String? get clinicianId => _clinicianId;
  static bool get isActive => (_token ?? '').isNotEmpty;

  static void set({required String token, String? clinicianId}) {
    _token = token;
    _clinicianId = clinicianId;
  }

  static void clear() {
    _token = null;
    _clinicianId = null;
  }
}
