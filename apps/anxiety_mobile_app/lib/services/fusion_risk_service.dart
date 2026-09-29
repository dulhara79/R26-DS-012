import 'dart:async';

import 'package:flutter/foundation.dart';

import 'api_service.dart';
import 'participant_identity_service.dart';

/// A patient-safe C1 forecast, separate from the multimodal assessment.
class PatientForecast {
  const PatientForecast({
    required this.scope,
    required this.score,
    required this.tier,
    required this.horizonMinutes,
    required this.predicted,
    required this.generatedAt,
    required this.validUntil,
  });

  final String scope;
  final double score;
  final String tier;
  final int horizonMinutes;
  final bool predicted;
  final DateTime generatedAt;
  final DateTime validUntil;

  double get scoreOutOf100 => score * 100;
  bool isValidAt(DateTime time) =>
      !time.toUtc().isBefore(generatedAt.toUtc()) &&
      time.toUtc().isBefore(validUntil.toUtc());

  static PatientForecast? parse(Object? value) {
    if (value is! Map) return null;
    final score = value['score'];
    final tier = value['tier'];
    final horizon = value['horizon_minutes'];
    final predicted = value['predicted'];
    final generatedAt = DateTime.tryParse(
      value['generated_at']?.toString() ?? '',
    );
    final validUntil = DateTime.tryParse(
      value['valid_until']?.toString() ?? '',
    );
    if (value['scope'] != 'physiological' ||
        score is! num ||
        !score.isFinite ||
        score < 0 ||
        score > 1 ||
        tier is! String ||
        !const ['Low', 'Medium', 'High'].contains(tier) ||
        horizon is! int ||
        horizon <= 0 ||
        predicted is! bool ||
        generatedAt == null ||
        validUntil == null ||
        !validUntil.isAfter(generatedAt)) {
      return null;
    }
    return PatientForecast(
      scope: 'physiological',
      score: score.toDouble(),
      tier: tier,
      horizonMinutes: horizon,
      predicted: predicted,
      generatedAt: generatedAt,
      validUntil: validUntil,
    );
  }
}

/// The patient-facing view of the fusion result.
///
/// This is deliberately thin. The backend serves two different views of the
/// same fusion row: the clinician gets per-modality contributions, gate
/// reasons and conformal sets; the patient gets only a composite, a band and
/// a plain-language message. We do not ask for more than that here, and we
/// must not display anything the backend did not send.
class FusionRisk {
  /// Backend composite, on its native 0..1 scale.
  final double? composite;

  /// Authoritative server-side fusion assessment identifier shared with the clinician view.
  final int? fusionResultId;

  /// GREEN / AMBER / RED / GREY. GREY means the fusion gate refused to
  /// produce a score (for example only one modality was available), and it
  /// must never be rendered as if it were a low score.
  final String band;

  /// Backend-defined tier; the app never computes a clinical tier from score.
  final String? tier;
  final PatientForecast? forecast;

  /// Plain-language message written by the backend for the patient.
  final String? message;

  final DateTime? updatedAt;

  const FusionRisk({
    required this.composite,
    required this.band,
    this.tier,
    this.forecast,
    this.message,
    this.updatedAt,
    this.fusionResultId,
  });

  /// True only when the backend actually produced a usable score.
  bool get hasScore =>
      composite != null &&
      composite!.isFinite &&
      composite! >= 0 &&
      composite! <= 1 &&
      const ['Low', 'Medium', 'High'].contains(tier) &&
      const ['GREEN', 'AMBER', 'RED'].contains(band.toUpperCase());

  String get displayTier => hasScore ? tier! : 'Unavailable';

  /// The gauge on the home page works on a 0..100 scale, but the backend
  /// composite is 0..1. Converting here, once, keeps the mistake from being
  /// repeated at each call site.
  double? get scoreOutOf100 =>
      !hasScore ? null : (composite! * 100).clamp(0.0, 100.0);

  factory FusionRisk.fromJson(Map<String, dynamic> json) {
    final rawComposite = json['composite'];
    DateTime? parsedUpdatedAt;
    final rawUpdatedAt = json['updated_at'];
    if (rawUpdatedAt is String && rawUpdatedAt.isNotEmpty) {
      parsedUpdatedAt = DateTime.tryParse(rawUpdatedAt);
    }
    return FusionRisk(
      composite: rawComposite is num ? rawComposite.toDouble() : null,
      band: json['band']?.toString().toUpperCase() ?? 'GREY',
      tier: json['tier']?.toString(),
      forecast: PatientForecast.parse(json['forecast']),
      message: json['message']?.toString(),
      updatedAt: parsedUpdatedAt,
      fusionResultId: json['fusion_result_id'] is num
          ? (json['fusion_result_id'] as num).toInt()
          : null,
    );
  }
}

/// Returns the only score that is allowed to represent the app-wide
/// overall risk. A missing composite or GREY fusion result is unavailable,
/// never a client-side estimate.
double? officialOverallRisk(FusionRisk? risk) {
  if (risk == null || !risk.hasScore) return null;
  return risk.scoreOutOf100;
}

/// Reads the composite risk produced by the fusion engine.
///
/// The request uses the subject-bound patient session. If authentication or
/// the network fails, the result is null: missing data is "Unavailable",
/// never a client-generated low score.
class FusionRiskService {
  FusionRiskService._();

  static final FusionRiskService instance = FusionRiskService._();

  final ValueNotifier<FusionRisk?> latest = ValueNotifier(null);

  Timer? _pollTimer;
  int _generation = 0;

  static const Duration _pollInterval = Duration(minutes: 5);
  static const Duration _timeout = Duration(seconds: 10);

  /// Fetches once. Returns null when unpaired, unreachable, or on any
  /// non-200 response.
  Future<FusionRisk?> fetch() async {
    final generation = _generation;
    final subjectId = await ParticipantIdentityService.getCentralSubjectId();
    if (generation != _generation) return null;
    if (subjectId == null || subjectId.isEmpty) {
      latest.value = null;
      debugPrint('FusionRiskService: not paired with the central backend yet.');
      return null;
    }

    try {
      final decoded = await ApiService.getPatientRisk(
        subjectId,
      ).timeout(_timeout);
      if (generation != _generation) return null;
      if (decoded == null) {
        latest.value = null;
        debugPrint('FusionRiskService: authenticated risk is unavailable.');
        return null;
      }

      final risk = FusionRisk.fromJson(decoded);
      latest.value = risk;
      return risk;
    } catch (error) {
      if (generation == _generation) latest.value = null;
      debugPrint('FusionRiskService: fetch failed: $error');
      return null;
    }
  }

  void startPolling() {
    _pollTimer?.cancel();
    fetch();
    _pollTimer = Timer.periodic(_pollInterval, (_) => fetch());
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  /// Discard a prior participant's assessment, including pending responses.
  void clear() {
    stopPolling();
    _generation++;
    latest.value = null;
  }
}
