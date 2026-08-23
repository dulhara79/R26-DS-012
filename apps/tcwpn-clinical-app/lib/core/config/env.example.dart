// lib/core/config/env.dart

class Env {
  const Env._();

  static const String appName = 'ClinAnx';

  /// Component 4 — TC-WPN few-shot clinical NLP. This project's component.
  static const String tcwpnBase = String.fromEnvironment(
    'TCWPN_BASE',
    defaultValue: '<<space_url>>',
  );

  /// Late-fusion orchestration service (proposal §5.1). ClinAnx posts its C4
  /// contribution here and reads the composite back.
  static const String fusionBase = String.fromEnvironment('FUSION_BASE');

  /// Component 3 — intervention engine. Optional: when unset, the Intervention
  /// section of the chart explains that it is not yet wired.
  static const String c3Base = String.fromEnvironment('C3_BASE');

  /// Clinician authentication. When empty, the app falls back to build-time
  /// local credentials and marks itself a development build.
  static const String authBase = String.fromEnvironment('AUTH_BASE');

  static const String hfToken = String.fromEnvironment('HF_TOKEN');

  /// HuggingFace Spaces cold-start on free CPU tiers routinely exceeds a minute.
  static const Duration inferenceTimeout = Duration(seconds: 180);
  static const Duration quickTimeout = Duration(seconds: 25);

  static bool get hasFusion => fusionBase.isNotEmpty;
  static bool get hasC3 => c3Base.isNotEmpty;

  /// Fusion weights from proposal §5.1.
  ///
  /// Held client-side ONLY so a provisional composite can be shown when the
  /// fusion service is unreachable. Any composite computed with these is
  /// labelled provisional everywhere it appears. The server's own weights
  /// always win when it responds.
  static const Map<String, double> defaultWeights = {
    'c1_physiological': 0.25,
    'c2_behavioral': 0.20,
    'c3_intervention': 0.15,
    'c4_clinical_nlp': 0.40,
  };

  /// A modality reading older than this is stale: it still counts, but the
  /// chart says how old it is so a clinician is not misled by a three-week-old
  /// wearable score sitting next to a note written this morning.
  static const Duration stalenessThreshold = Duration(hours: 72);

  /// False for any build that touches real patients. Removes the seeded
  /// demonstration patient and the example note library.
  static const bool demoData =
      bool.fromEnvironment('DEMO_DATA', defaultValue: true);
}
