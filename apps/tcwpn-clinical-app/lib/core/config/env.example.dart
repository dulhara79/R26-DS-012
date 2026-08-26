// lib/core/config/env.dart
//
// REPLACES lib/core/config/env.example.dart (rename to env.dart when you copy
// it in — the rest of the app imports '../../core/config/env.dart').
//
// WHAT CHANGED FROM THE PREVIOUS VERSION, AND WHICH FEEDBACK ITEM IT CLOSES
// =========================================================================
//
// • `hfToken` DELETED — feedback section 23.
//   It was still read by ApiClient._defaultBearer as a fallback bearer, so a
//   release APK shipped a HuggingFace deploy credential that any `apktool`
//   run recovers in about a minute. Section 23 says not to put a privileged
//   model token in the APK; it was in there. The TC-WPN warm-up ping is
//   unauthenticated /health, which does not need it. If your Space is private,
//   make the BACKEND proxy the warm-up rather than shipping the token.
//
// • `c3Base` DELETED — feedback section 26, "No old /v3/risk/classify".
//   C3Gateway.classify posted to /v3/risk/classify, the retired intervention
//   route. It was dead code: nothing in lib/features called it. Removing the
//   URL removes the temptation to wire it back up, and settings_screen no
//   longer advertises a service the app does not use.
//
// • `authBase` DELETED.
//   It described a clinician login endpoint the backend does not implement.
//   main.py::_auth compares the Authorization header against one static
//   BACKEND_API_TOKEN — there is no /auth/login, no JWT, no refresh token.
//   Feedback section 3 asks for that contract; the honest answer is that it
//   does not exist yet. Keeping a config knob for an imaginary endpoint makes
//   the app look more secure than it is.
//
// WHAT SURVIVES, AND WHY IT IS NOT A CONTRADICTION
// ------------------------------------------------
// `tcwpnBase` stays. Section 21 allows "a separate TC-WPN development URL only
// for isolated development/testing", and that is precisely its remaining use:
// a GET /health warm-up so the clinician's first note analysis does not pay a
// full cold start. The clinician workflow must never call /predict here.

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
  /// authentication and authorisation.
  static const String backendToken = String.fromEnvironment('BACKEND_TOKEN');

  /// TC-WPN Space. Retained ONLY for the unauthenticated /health warm-up, so
  /// the first note analysis does not pay the full cold start. The clinician
  /// workflow must not call /predict here — that path belongs to the backend.
  static const String tcwpnBase = String.fromEnvironment('TCWPN_BASE');

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
  /// the two disagree, show the server's.
  static const Duration stalenessThreshold = Duration(hours: 72);

  /// False for any build that touches real patients. Removes the seeded
  /// demonstration patient and the example note library.
  static const bool demoData =
      bool.fromEnvironment('DEMO_DATA', defaultValue: true);
}
