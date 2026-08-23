// lib/core/config/env.dart
//
// Build-time configuration.
//
// CHANGED FOR CENTRAL-BACKEND INTEGRATION
// ---------------------------------------
// ClinAnx now talks to ONE service: the R26-DS-012 Central Backend. The backend
// calls TC-WPN and the other component services on its behalf. The app no longer
// posts /predict to the TC-WPN Space in the clinician workflow, and no longer
// holds fusion weights of any kind.
//
// `Env.defaultWeights` was deleted. The server derives its weights from each
// component's validation AUROC above chance (fusion_service/fusion.py,
// base_weights()), which produces a materially different vector from the fixed
// 0.25/0.20/0.15/0.40 this file used to carry. Two divergent weight sets, one of
// them shown to a clinician, is a correctness hazard rather than a fallback.

class Env {
  const Env._();

  static const String appName = 'ClinAnx';

  // ── The one service this app talks to ────────────────────────────────────

  /// R26-DS-012 Central Backend. Everything clinical goes through here.
  ///
  /// Android emulator reaching a backend on the host machine: use
  /// `http://10.0.2.2:8000`, not `localhost`.
  static const String backendBase = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: '',
  );

  /// The backend's shared bearer token (`BACKEND_API_TOKEN` server-side).
  ///
  /// Injected at build time, never written into source. This is a single shared
  /// app credential, not a per-clinician one: `main.py::_auth` compares the
  /// Authorization header against one static value. Clinician attribution
  /// therefore travels in the request body's `author` field, not in the token.
  /// That limitation is stated in the README and must be stated in the viva —
  /// it is adequate for a research prototype and is NOT adequate for a clinical
  /// deployment, which needs per-user authentication and authorisation.
  static const String backendToken = String.fromEnvironment('BACKEND_TOKEN');

  // ── Component services ───────────────────────────────────────────────────

  /// TC-WPN Space. Retained ONLY for the /health warm-up ping, so the first
  /// note analysis does not pay the full cold-start. The clinician workflow
  /// must not call /predict here — that path belongs to the backend now.
  static const String tcwpnBase = String.fromEnvironment('TCWPN_BASE');

  /// Component 3 — the Personalised Intervention Framework.
  ///
  /// NOT a fusion modality. The backend's four modalities are physiological,
  /// behavioural, clinical-NLP and demographic; intervention is not among them.
  /// When configured, this is called directly and displayed as its own section
  /// of the chart. It never enters the composite.
  static const String c3Base = String.fromEnvironment('C3_BASE');

  /// Clinician authentication. When empty, the app falls back to build-time
  /// local credentials and marks itself a development build.
  static const String authBase = String.fromEnvironment('AUTH_BASE');

  static const String hfToken = String.fromEnvironment('HF_TOKEN');

  /// A cold HuggingFace Space behind the backend can push a single note
  /// analysis past a minute; COMPONENT_TIMEOUT_S is 60s server-side and the
  /// backend adds its own work on top.
  static const Duration inferenceTimeout = Duration(seconds: 180);
  static const Duration quickTimeout = Duration(seconds: 25);

  static bool get hasBackend => backendBase.isNotEmpty;
  static bool get hasC3 => c3Base.isNotEmpty;
  static bool get hasTcwpnWarmup => tcwpnBase.isNotEmpty;

  /// A modality reading older than this is called out in the chart. This is a
  /// DISPLAY threshold only — the authoritative freshness decision is the
  /// backend's `modalities[*].fresh`, computed from gate.MAX_AGE_MINUTES, which
  /// differs per modality (15 min for physiological, 90 days for notes). Where
  /// the two disagree, show the server's.
  static const Duration stalenessThreshold = Duration(hours: 72);

  /// False for any build that touches real patients. Removes the seeded
  /// demonstration patient and the example note library.
  static const bool demoData =
      bool.fromEnvironment('DEMO_DATA', defaultValue: true);
}
