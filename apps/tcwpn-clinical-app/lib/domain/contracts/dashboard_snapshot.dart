import 'attention_event.dart';
import 'contract_parsing.dart';
import 'patient_summary.dart';
import 'clinician_principal.dart';

class DashboardSnapshot {
  final List<AttentionEvent> openEvents;
  final List<PatientSummary> assignedPatients;
  final DateTime fetchedAt;
  final bool isFromCache;
  final ClinicianPrincipal? clinician;
  final int? authoritativeAssignedCount;

  const DashboardSnapshot({
    required this.openEvents,
    required this.assignedPatients,
    required this.fetchedAt,
    this.isFromCache = false,
    this.clinician,
    this.authoritativeAssignedCount,
  });

  int get assignedCount => authoritativeAssignedCount ?? assignedPatients.length;

  bool get isEmpty => openEvents.isEmpty && assignedPatients.isEmpty;

  DashboardSnapshot asCached() => DashboardSnapshot(
        openEvents: openEvents,
        assignedPatients: assignedPatients,
        fetchedAt: fetchedAt,
        isFromCache: true,
        clinician: clinician,
        authoritativeAssignedCount: authoritativeAssignedCount,
      );

  Map<String, dynamic> toJson() => {
        'open_attention_events': openEvents.map((e) => e.toJson()).toList(),
        'patient_summaries': assignedPatients.map((p) => p.toJson()).toList(),
        'fetched_at': fetchedAt.toUtc().toIso8601String(),
        'clinician': clinician == null ? null : {
          'clinician_id': clinician!.clinicianId,
          'display_name': clinician!.displayName,
          'role': clinician!.role,
        },
        'assigned_count': assignedCount,
      };

  factory DashboardSnapshot.fromJson(
    Map<String, dynamic> json, {
    bool isFromCache = true,
  }) {
    final eventRows = contractMapList(
      json['open_attention_events'] ?? json['events'],
    );
    final patientRows = contractMapList(
      json['patient_summaries'] ?? json['patients'],
    );
    final clinicianRow = contractMap(json['clinician']);
    return DashboardSnapshot(
      openEvents:
          eventRows.map(AttentionEvent.fromJson).toList(growable: false),
      assignedPatients:
          patientRows.map(PatientSummary.fromJson).toList(growable: false),
      fetchedAt: contractDateTime(json['fetched_at']) ?? DateTime.fromMillisecondsSinceEpoch(0),
      isFromCache: isFromCache,
      clinician: clinicianRow == null
          ? null
          : ClinicianPrincipal.fromJson(clinicianRow),
      authoritativeAssignedCount: contractInt(json['assigned_count']),
    );
  }
}
