import 'attention_event.dart';
import 'contract_parsing.dart';
import 'patient_summary.dart';

class DashboardSnapshot {
  final List<AttentionEvent> openEvents;
  final List<PatientSummary> assignedPatients;
  final DateTime fetchedAt;
  final bool isFromCache;

  const DashboardSnapshot({
    required this.openEvents,
    required this.assignedPatients,
    required this.fetchedAt,
    this.isFromCache = false,
  });

  bool get isEmpty => openEvents.isEmpty && assignedPatients.isEmpty;

  DashboardSnapshot asCached() => DashboardSnapshot(
        openEvents: openEvents,
        assignedPatients: assignedPatients,
        fetchedAt: fetchedAt,
        isFromCache: true,
      );

  Map<String, dynamic> toJson() => {
        'open_attention_events': openEvents.map((e) => e.toJson()).toList(),
        'patient_summaries': assignedPatients.map((p) => p.toJson()).toList(),
        'fetched_at': fetchedAt.toUtc().toIso8601String(),
      };

  factory DashboardSnapshot.fromJson(Map<String, dynamic> json) {
    final eventRows = contractMapList(
      json['open_attention_events'] ?? json['events'],
    );
    final patientRows = contractMapList(
      json['patient_summaries'] ?? json['patients'],
    );
    return DashboardSnapshot(
      openEvents:
          eventRows.map(AttentionEvent.fromJson).toList(growable: false),
      assignedPatients:
          patientRows.map(PatientSummary.fromJson).toList(growable: false),
      fetchedAt: contractDateTime(json['fetched_at']) ?? DateTime.fromMillisecondsSinceEpoch(0),
      isFromCache: true,
    );
  }
}
