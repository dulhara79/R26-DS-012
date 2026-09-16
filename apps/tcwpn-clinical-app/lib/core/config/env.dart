// lib/core/config/env.dart
//
// Central build-time configuration for ClinAnx. Values come from --dart-define;
// no reusable credential should ever be committed as a source literal.

class Env {
  const Env._();

  static const String appName = 'ClinAnx';

  // ── Central Backend ────────────────────────────────────────────────────────

  /// R26-DS-012 Central Backend. Authoritative clinical operations go through
  /// this service: enrolment, note ingestion, fusion, timeline, evidence and
  /// server-owned AttentionEvent operations as those backend routes become
  /// available.
  static const String backendBase = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: '',
  );

  /// Transitional Central Backend service credential used by verified legacy
  /// gateway contracts. It is not a clinician identity. New clinician-scoped
  /// contracts use the authenticated [Session] bearer via ApiClient's default
  /// bearer callback and must not treat this shared token as authorization.
  static const String backendToken = String.fromEnvironment('BACKEND_TOKEN');

  /// TC-WPN Space. Retained only for unauthenticated GET /health warm-up.
  /// Authoritative note inference goes through the Central Backend.
  static const String tcwpnBase = String.fromEnvironment('TCWPN_BASE');

  // ── Clinician authentication ──────────────────────────────────────────────

  /// Separate clinician-auth service. Empty selects local demo/development mode.
  static const String authBase = String.fromEnvironment('AUTH_BASE');

  static const String authSalt = String.fromEnvironment(
    'AUTH_SALT',
    defaultValue: 'r26-ds012-local-salt',
  );

  /// Local-mode account table. Format is AuthService's; see that file.
  static const String authLocalAccounts = String.fromEnvironment('AUTH_LOCAL');

  static bool get hasRemoteAuth => authBase.isNotEmpty;

  // ── Timeouts ───────────────────────────────────────────────────────────────

  static const Duration inferenceTimeout = Duration(seconds: 180);
  static const Duration quickTimeout = Duration(seconds: 25);

  // ── Derived ────────────────────────────────────────────────────────────────

  static bool get hasBackend => backendBase.isNotEmpty;
  static bool get hasTcwpnWarmup => tcwpnBase.isNotEmpty;
  static bool get isConfigured => hasBackend;

  /// Display-only fallback threshold. Authoritative modality freshness is the
  /// server-provided value when available.
  static const Duration stalenessThreshold = Duration(hours: 72);

  /// Demo fixtures are opt-in. Omitting the flag must never seed a demonstration
  /// patient into a build that may be used with participant data.
  static const bool demoData = bool.fromEnvironment(
    'DEMO_DATA',
    defaultValue: false,
  );
}
