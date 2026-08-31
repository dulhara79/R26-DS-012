// lib/core/config/env.dart
//
// COMMIT THIS FILE. Remove `lib/core/config/env.dart` from .gitignore and
// delete `lib/core/config/env.example.dart`.
//
// WHY THE GITIGNORE HAD TO GO
// ---------------------------
// Twenty-one files under lib/ import '../../core/config/env.dart'. The file was
// gitignored, so a clean clone of develop/integration cannot resolve that
// import: `flutter analyze`, `flutter test` and `flutter build apk` all fail on
// the first file that touches it. Four acceptance criteria in section 33 fail
// for a reason that has nothing to do with the code.
//
// The file was presumably gitignored because an earlier version held an
// HF_TOKEN literal. It no longer does. Every value below is
// `String.fromEnvironment` / `bool.fromEnvironment`, which reads a --dart-define
// at COMPILE time and defaults to empty. There is nothing secret in this source
// file; the secrets live in the build command and in CI, which is exactly where
// section 16 wants them.
//
// If you ever feel the urge to paste a literal token in here: don't. Anything
// compiled into an APK is recoverable with `apktool` in about a minute.

class Env {
  const Env._();

  static const String appName = 'ClinAnx';

  // ── The one service this app talks to ──────────────────────────────────────

  /// R26-DS-012 Central Backend. Everything clinical goes through here:
  /// enrolment, note ingestion, gate, fusion, timeline, evidence, verdict.
  ///
  /// Android emulator reaching a backend on the host machine: use
  /// `http://10.0.2.2:8000`, not `localhost`.
  static const String backendBase = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: '',
  );

  /// The backend's shared bearer token (`BACKEND_API_TOKEN` server-side).
  ///
  /// Injected at build time, never written into source. This is a SINGLE SHARED
  /// APP CREDENTIAL, not a per-clinician one. Clinician attribution therefore
  /// travels in each request body's `author` field, not in the token.
  ///
  /// State this limitation in the viva. It is adequate for a research prototype
  /// and is NOT adequate for clinical deployment, which needs per-user
  /// authentication and authorisation. Section 16 asks for the code to be
  /// structured so this can later be replaced by a clinician JWT: it is —
  /// `ApiClient`'s `bearer` callback is evaluated per request, so swapping the
  /// source of the credential is a one-line change at the gateway's call site.
  static const String backendToken = String.fromEnvironment('BACKEND_TOKEN');

  /// TC-WPN Space. Retained ONLY for the unauthenticated `GET /health` warm-up,
  /// so the first note analysis does not pay the full cold start (section 9).
  /// The clinician workflow must never call `/predict` here — inference belongs
  /// to the backend.
  ///
  /// Leave it empty and the app simply skips the warm-up. Nothing else changes.
  static const String tcwpnBase = String.fromEnvironment('TCWPN_BASE');

  // ── Clinician authentication ──────────────────────────────────────────────
  //
  // MOVED HERE from AuthService, which read these three defines directly. That
  // put four config knobs in two files and made section 17's "centralize
  // configuration" false in a way that was easy to miss: `grep -rn
  // fromEnvironment lib/` was the only way to find them.
  //
  // This is a SEPARATE service from the Central Backend. The backend's own auth
  // is the shared `backendToken` above; `authBase` is the clinician sign-in
  // service that issues the session shown in the app bar. Do not point one at
  // the other.

  /// ClinAnx auth service. Empty selects LOCAL mode: credentials compiled into
  /// the build, registration and password reset disabled, and a red banner in
  /// release builds. Development and demos only.
  static const String authBase = String.fromEnvironment('AUTH_BASE');

  /// Salt for the local-mode password hash. Overriding it is pointless security
  /// theatre — local mode is not secure — but it keeps two demo builds from
  /// sharing a hash table.
  static const String authSalt = String.fromEnvironment(
    'AUTH_SALT',
    defaultValue: 'r26-ds012-local-salt',
  );

  /// Local-mode account table. Format is AuthService's; see that file.
  static const String authLocalAccounts = String.fromEnvironment('AUTH_LOCAL');

  static bool get hasRemoteAuth => authBase.isNotEmpty;

  // ── Timeouts ───────────────────────────────────────────────────────────────

  /// A cold Space behind the backend can push one note analysis past a minute;
  /// COMPONENT_TIMEOUT_S is 60s server-side and the backend adds its own work
  /// on top.
  static const Duration inferenceTimeout = Duration(seconds: 180);
  static const Duration quickTimeout = Duration(seconds: 25);

  // ── Derived ────────────────────────────────────────────────────────────────

  static bool get hasBackend => backendBase.isNotEmpty;
  static bool get hasTcwpnWarmup => tcwpnBase.isNotEmpty;

  /// True for a build that can actually reach the clinical path. Use this to
  /// gate the UI rather than letting screens fail one call at a time.
  static bool get isConfigured => hasBackend;

  /// A modality reading older than this is called out in the chart.
  ///
  /// DISPLAY THRESHOLD ONLY. The authoritative freshness decision is the
  /// backend's `modalities[*].fresh`, computed from gate.MAX_AGE_MINUTES, which
  /// differs per modality — minutes for physiological, months for notes. Where
  /// the two disagree, show the server's (section 20).
  static const Duration stalenessThreshold = Duration(hours: 72);

  /// False for any build that touches real patients. Removes the seeded
  /// demonstration patient and the example note library.
  static const bool demoData = bool.fromEnvironment(
    'DEMO_DATA',
    defaultValue: true,
  );
}
