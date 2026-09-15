import 'assessment_summary.dart';
import 'contract_enums.dart';
import 'contract_parsing.dart';

class PatientSummary {
  final String subjectId;
  final String? displayId;
  final int? fusionResultId;
  final CurrentAssessment? currentAssessment;
  final ForecastResult? forecast;
  final AssessmentStatus assessmentStatus;
  final DateTime? lastUpdated;
  final int? openEventCount;

  const PatientSummary({
    required this.subjectId,
    required this.displayId,
    required this.fusionResultId,
    required this.currentAssessment,
    required this.forecast,
    required this.assessmentStatus,
    required this.lastUpdated,
    required this.openEventCount,
  });

  factory PatientSummary.fromJson(Map<String, dynamic> json) {
    final current = contractMap(json['current'] ?? json['current_assessment']);
    final forecast = contractMap(json['forecast']);
    return PatientSummary(
      subjectId: contractString(json['subject_id']) ?? '',
      displayId: contractString(json['display_id']),
      fusionResultId: contractInt(json['fusion_result_id']),
      currentAssessment:
          current == null ? null : CurrentAssessment.fromJson(current),
      forecast: forecast == null ? null : ForecastResult.fromJson(forecast),
      assessmentStatus: AssessmentStatus.fromWire(json['assessment_status']),
      lastUpdated: contractDateTime(json['last_updated']),
      openEventCount: contractInt(json['open_event_count']),
    );
  }
}
