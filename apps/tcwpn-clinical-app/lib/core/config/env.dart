// lib/core/config/env.dart
//
// Central build-time configuration for ClinAnx. Values come from --dart-define;
// no reusable credential should ever be committed as a source literal.

class Env {
  const Env._();

  static const String appName = 'ClinAnx';

  // ── Central Backend ────────────────────────────────────────────────────────

  static const String backendBase = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: '',
  );
  static const String backendToken = String.fromEnvironment('BACKEND_TOKEN');
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

  static bool get useSecondaryFirebase =>
      pushFirebaseSlot.trim().toLowerCase() == 'secondary';

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

  static const Duration stalenessThreshold = Duration(hours: 72);

  static const bool demoData = bool.fromEnvironment(
    'DEMO_DATA',
    defaultValue: false,
  );
}
