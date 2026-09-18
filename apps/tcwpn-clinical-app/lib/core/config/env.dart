// lib/core/config/env.dart
//
// Central build-time configuration for ClinAnx. Values come from --dart-define;
// no reusable privileged credential is compiled into the mobile application.

class Env {
  const Env._();

  static const String appName = 'ClinAnx';

  // ── Research build identity ───────────────────────────────────────────────
  // Non-secret identifiers make a study/demo build reproducible and prevent
  // accidental endpoint/build confusion. They are safe to display in Settings.

  static const String appVersion = String.fromEnvironment(
    'APP_VERSION',
    defaultValue: '1.0.0+1',
  );
  static const String buildEnvironment = String.fromEnvironment(
    'BUILD_ENVIRONMENT',
    defaultValue: 'development',
  );
  static const String buildRevision = String.fromEnvironment(
    'BUILD_REVISION',
    defaultValue: 'unversioned',
  );

  // ── Central Backend ────────────────────────────────────────────────────────
  // Central Backend requests use the authenticated clinician Session bearer.
  // Service-to-service/shared privileged credentials belong on the backend and
  // must not be compiled into ClinAnx.

  static const String backendBase = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: '',
  );
  static const String tcwpnBase = String.fromEnvironment('TCWPN_BASE');

  // ── Clinician authentication ──────────────────────────────────────────────

  static const String authBase = String.fromEnvironment('AUTH_BASE');
  static const String authSalt = String.fromEnvironment(
    'AUTH_SALT',
    defaultValue: 'r26-ds012-local-salt',
  );
  static const String authLocalAccounts = String.fromEnvironment('AUTH_LOCAL');
  static bool get hasRemoteAuth => authBase.isNotEmpty;

  // ── Phase 7 push / disaster-recovery Firebase slots ───────────────────────
  // Only one slot is active in an installed build. FlutterFire Messaging does
  // not support runtime switching between multiple messaging FirebaseApp
  // instances. The second slot is for a separately built DR APK/app package;
  // runtime resilience remains server persistence + polling fallback.

  static const Set<String> validPushSlots = <String>{'primary', 'secondary'};

  static const String pushFirebaseSlot = String.fromEnvironment(
    'PUSH_FIREBASE_SLOT',
    defaultValue: 'primary',
  );

  static const String firebasePrimaryApiKey =
      String.fromEnvironment('FIREBASE_PRIMARY_API_KEY');
  static const String firebasePrimaryAppId =
      String.fromEnvironment('FIREBASE_PRIMARY_APP_ID');
  static const String firebasePrimarySenderId =
      String.fromEnvironment('FIREBASE_PRIMARY_SENDER_ID');
  static const String firebasePrimaryProjectId =
      String.fromEnvironment('FIREBASE_PRIMARY_PROJECT_ID');
  static const String firebasePrimaryIosBundleId =
      String.fromEnvironment('FIREBASE_PRIMARY_IOS_BUNDLE_ID');

  static const String firebaseSecondaryApiKey =
      String.fromEnvironment('FIREBASE_SECONDARY_API_KEY');
  static const String firebaseSecondaryAppId =
      String.fromEnvironment('FIREBASE_SECONDARY_APP_ID');
  static const String firebaseSecondarySenderId =
      String.fromEnvironment('FIREBASE_SECONDARY_SENDER_ID');
  static const String firebaseSecondaryProjectId =
      String.fromEnvironment('FIREBASE_SECONDARY_PROJECT_ID');
  static const String firebaseSecondaryIosBundleId =
      String.fromEnvironment('FIREBASE_SECONDARY_IOS_BUNDLE_ID');

  static String get normalizedPushFirebaseSlot =>
      pushFirebaseSlot.trim().toLowerCase();
  static bool get hasValidPushFirebaseSlot =>
      validPushSlots.contains(normalizedPushFirebaseSlot);
  static bool get useSecondaryFirebase => normalizedPushFirebaseSlot == 'secondary';

  static String get activeFirebaseApiKey => useSecondaryFirebase
      ? firebaseSecondaryApiKey
      : firebasePrimaryApiKey;
  static String get activeFirebaseAppId => useSecondaryFirebase
      ? firebaseSecondaryAppId
      : firebasePrimaryAppId;
  static String get activeFirebaseSenderId => useSecondaryFirebase
      ? firebaseSecondarySenderId
      : firebasePrimarySenderId;
  static String get activeFirebaseProjectId => useSecondaryFirebase
      ? firebaseSecondaryProjectId
      : firebasePrimaryProjectId;
  static String get activeFirebaseIosBundleId => useSecondaryFirebase
      ? firebaseSecondaryIosBundleId
      : firebasePrimaryIosBundleId;

  static bool get hasPushFirebase =>
      hasValidPushFirebaseSlot &&
      activeFirebaseApiKey.isNotEmpty &&
      activeFirebaseAppId.isNotEmpty &&
      activeFirebaseSenderId.isNotEmpty &&
      activeFirebaseProjectId.isNotEmpty;

  // ── Timeouts ───────────────────────────────────────────────────────────────

  static const Duration inferenceTimeout = Duration(seconds: 180);
  static const Duration quickTimeout = Duration(seconds: 25);

  // ── Derived ────────────────────────────────────────────────────────────────

  static bool get hasBackend => backendBase.isNotEmpty;
  static bool get hasTcwpnWarmup => tcwpnBase.isNotEmpty;
  static bool get isConfigured => hasBackend;

  /// Handbook Phase 8 transport rule: participant/clinical traffic uses HTTPS.
  /// Plain HTTP is tolerated only for loopback development, never remote hosts.
  static bool get isBackendTransportSafe {
    if (!hasBackend) return true;
    final uri = Uri.tryParse(backendBase.trim());
    if (uri == null || uri.host.isEmpty) return false;
    if (uri.scheme.toLowerCase() == 'https') return true;
    if (uri.scheme.toLowerCase() != 'http') return false;
    final host = uri.host.toLowerCase();
    return host == 'localhost' || host == '127.0.0.1' || host == '::1';
  }

  static const Duration stalenessThreshold = Duration(hours: 72);

  static const bool demoData = bool.fromEnvironment(
    'DEMO_DATA',
    defaultValue: false,
  );
}
