import 'contract_enums.dart';
import 'contract_parsing.dart';

class CurrentAssessment {
  final double? score;
  final RiskTier tier;
  final String? band;

  const CurrentAssessment({
    required this.score,
    required this.tier,
    required this.band,
  });

  factory CurrentAssessment.fromJson(Map<String, dynamic> json) =>
      CurrentAssessment(
        score: contractDouble(json['score']),
        tier: RiskTier.fromWire(json['tier']),
        band: contractString(json['band']),
      );
}

class ForecastResult {
  final String? forecastResultId;
  final ForecastScope scope;
  final int? horizonMinutes;
  final double? score;
  final RiskTier tier;
  final double? escalationProbability;
  final bool? escalationPredicted;
  final DateTime? generatedAt;
  final DateTime? validUntil;

  const ForecastResult({
    required this.forecastResultId,
    required this.scope,
    required this.horizonMinutes,
    required this.score,
    required this.tier,
    required this.escalationProbability,
    required this.escalationPredicted,
    required this.generatedAt,
    required this.validUntil,
  });

  factory ForecastResult.fromJson(Map<String, dynamic> json) => ForecastResult(
        forecastResultId: contractString(json['forecast_result_id']),
        scope: ForecastScope.fromWire(json['scope']),
        horizonMinutes: contractInt(json['horizon_minutes']),
        score: contractDouble(json['score']),
        tier: RiskTier.fromWire(json['tier']),
        escalationProbability: contractDouble(json['escalation_probability']),
        escalationPredicted: contractBool(json['escalation_predicted']),
        generatedAt: contractDateTime(json['generated_at']),
        validUntil: contractDateTime(json['valid_until']),
      );
}

class ModalityStatus {
  final String componentId;
  final double? score;
  final bool? available;
  final bool? includedInFusion;
  final ModalityState state;
  final double? confidence;
  final double? coverage;
  final DateTime? capturedAt;
  final double? contribution;

  const ModalityStatus({
    required this.componentId,
    required this.score,
    required this.available,
    required this.includedInFusion,
    required this.state,
    required this.confidence,
    required this.coverage,
    required this.capturedAt,
    required this.contribution,
  });

  bool get isExperimentalExcluded =>
      state == ModalityState.notValidated && includedInFusion == false;

  factory ModalityStatus.fromJson(Map<String, dynamic> json) => ModalityStatus(
        componentId: contractString(json['component_id']) ?? '',
        score: contractDouble(json['score']),
        available: contractBool(json['available']),
        includedInFusion: contractBool(json['included_in_fusion']),
        state: ModalityState.fromWire(json['status']),
        confidence: contractDouble(json['confidence']),
        coverage: contractDouble(json['coverage']),
        capturedAt: contractDateTime(json['captured_at']),
        contribution: contractDouble(json['contribution']),
      );
}

class AssessmentSummary {
  final String subjectId;
  final int? fusionResultId;
  final CurrentAssessment currentAssessment;
  final ForecastResult? forecast;
  final double? confidence;
  final AssessmentStatus assessmentStatus;
  final List<ModalityStatus> modalities;
  final DateTime? computedAt;
  final String? modelVersion;

  const AssessmentSummary({
    required this.subjectId,
    required this.fusionResultId,
    required this.currentAssessment,
    required this.forecast,
    required this.confidence,
    required this.assessmentStatus,
    required this.modalities,
    required this.computedAt,
    required this.modelVersion,
  });

  factory AssessmentSummary.fromJson(Map<String, dynamic> json) {
    final current = contractMap(json['current_assessment']) ?? const {};
    final forecast = contractMap(json['forecast']);
    return AssessmentSummary(
      subjectId: contractString(json['subject_id']) ?? '',
      fusionResultId: contractInt(json['fusion_result_id']),
      currentAssessment: CurrentAssessment.fromJson(current),
      forecast: forecast == null ? null : ForecastResult.fromJson(forecast),
      confidence: contractDouble(json['confidence'] ?? json['uncertainty']),
      assessmentStatus: AssessmentStatus.fromWire(json['assessment_status']),
      modalities: contractMapList(json['modalities'])
          .map(ModalityStatus.fromJson)
          .toList(growable: false),
      computedAt: contractDateTime(json['computed_at']),
      modelVersion: contractString(json['model_version']),
    );
  }
}
