import 'contract_enums.dart';
import 'contract_parsing.dart';

class AttentionEvent {
  final String id;
  final String subjectId;
  final int? fusionResultId;
  final String? forecastResultId;
  final String? eventType;
  final AttentionSeverity severity;
  final String? reason;
  final int? forecastHorizon;
  final AttentionEventStatus status;
  final DateTime? createdAt;
  final DateTime? acknowledgedAt;
  final String? acknowledgedBy;
  final DateTime? resolvedAt;
  final String? resolvedBy;
  final String? policyVersion;

  const AttentionEvent({
    required this.id,
    required this.subjectId,
    required this.fusionResultId,
    required this.forecastResultId,
    required this.eventType,
    required this.severity,
    required this.reason,
    required this.forecastHorizon,
    required this.status,
    required this.createdAt,
    required this.acknowledgedAt,
    required this.acknowledgedBy,
    required this.resolvedAt,
    required this.resolvedBy,
    required this.policyVersion,
  });

  factory AttentionEvent.fromJson(Map<String, dynamic> json) => AttentionEvent(
        id: contractString(json['id']) ?? '',
        subjectId: contractString(json['subject_id']) ?? '',
        fusionResultId: contractInt(json['fusion_result_id']),
        forecastResultId: contractString(json['forecast_result_id']),
        eventType: contractString(json['event_type']),
        severity: AttentionSeverity.fromWire(json['severity']),
        reason: contractString(json['reason']),
        forecastHorizon: contractInt(json['forecast_horizon']),
        status: AttentionEventStatus.fromWire(json['status']),
        createdAt: contractDateTime(json['created_at']),
        acknowledgedAt: contractDateTime(json['acknowledged_at']),
        acknowledgedBy: contractString(json['acknowledged_by']),
        resolvedAt: contractDateTime(json['resolved_at']),
        resolvedBy: contractString(json['resolved_by']),
        policyVersion: contractString(json['policy_version']),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'subject_id': subjectId,
        'fusion_result_id': fusionResultId,
        'forecast_result_id': forecastResultId,
        'event_type': eventType,
        'severity': severity.name,
        'reason': reason,
        'forecast_horizon': forecastHorizon,
        'status': status.name.toUpperCase(),
        'created_at': createdAt?.toUtc().toIso8601String(),
        'acknowledged_at': acknowledgedAt?.toUtc().toIso8601String(),
        'acknowledged_by': acknowledgedBy,
        'resolved_at': resolvedAt?.toUtc().toIso8601String(),
        'resolved_by': resolvedBy,
        'policy_version': policyVersion,
      };
}
