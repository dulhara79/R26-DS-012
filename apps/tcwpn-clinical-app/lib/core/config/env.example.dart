// Example/reference build-time configuration for ClinAnx.
//
// Do not add reusable backend/model credentials here. Central Backend calls use
// the authenticated clinician Session bearer. Service credentials belong on the
// Central Backend, never in the mobile APK/app bundle.

class EnvExample {
  const EnvExample._();

  // Research build identity (safe to display in Settings).
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

  // Application-facing services. Remote clinical endpoints must use HTTPS.
  static const String backendBase = String.fromEnvironment('BACKEND_BASE');
  static const String authBase = String.fromEnvironment('AUTH_BASE');
  static const String tcwpnBase = String.fromEnvironment('TCWPN_BASE');

  // Local auth is demo/development only. Real participant builds use AUTH_BASE.
  static const String authSalt = String.fromEnvironment('AUTH_SALT');
  static const String authLocalAccounts = String.fromEnvironment('AUTH_LOCAL');

  // Firebase push transport. Only one slot is active per installed build;
  // primary/secondary are separately built DR configurations.
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
  static const String firebaseSecondaryApiKey =
      String.fromEnvironment('FIREBASE_SECONDARY_API_KEY');
  static const String firebaseSecondaryAppId =
      String.fromEnvironment('FIREBASE_SECONDARY_APP_ID');
  static const String firebaseSecondarySenderId =
      String.fromEnvironment('FIREBASE_SECONDARY_SENDER_ID');
  static const String firebaseSecondaryProjectId =
      String.fromEnvironment('FIREBASE_SECONDARY_PROJECT_ID');

  // Never seed synthetic participants unless the build explicitly opts in.
  static const bool demoData = bool.fromEnvironment(
    'DEMO_DATA',
    defaultValue: false,
  );
}
