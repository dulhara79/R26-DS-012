import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/data/repositories/composite_dashboard_repository.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/contracts/patient_summary.dart';
import 'package:r26_ds012_app/domain/repositories/attention_event_repository.dart';
import 'package:r26_ds012_app/domain/repositories/patient_repository.dart';

class _Patients implements PatientRepository {
  final List<PatientSummary> value;
  _Patients(this.value);

  @override
  Future<List<PatientSummary>> assignedPatients() async => value;
}

class _Events implements AttentionEventRepository {
  final List<AttentionEvent> value;
  _Events(this.value);

  @override
  Future<List<AttentionEvent>> openEvents() async => value;

  @override
  Future<List<AttentionEvent>> activity({String? subjectId}) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent?> eventById(String eventId) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent> acknowledge(String eventId) =>
      throw UnimplementedError();

  @override
  Future<AttentionEvent> resolve(String eventId) => throw UnimplementedError();
}

void main() {
  test('combines repository results without changing their order', () async {
    const patient = PatientSummary(
      subjectId: 'S1',
      displayId: 'P1',
      fusionResultId: 7,
      currentAssessment: null,
      forecast: null,
      assessmentStatus: AssessmentStatus.unavailable,
      lastUpdated: null,
      openEventCount: 1,
    );
    const event = AttentionEvent(
      id: 'E1',
      subjectId: 'S1',
      fusionResultId: 7,
      forecastResultId: null,
      eventType: null,
      severity: AttentionSeverity.high,
      reason: null,
      forecastHorizon: null,
      status: AttentionEventStatus.open,
      createdAt: null,
      acknowledgedAt: null,
      acknowledgedBy: null,
      resolvedAt: null,
      resolvedBy: null,
      policyVersion: null,
    );
    final at = DateTime.utc(2026, 9, 16);
    final repo = CompositeDashboardRepository(
      patients: _Patients(const [patient]),
      attentionEvents: _Events(const [event]),
      now: () => at,
    );

    final result = await repo.loadDashboard();

    expect(result.assignedPatients.single.subjectId, 'S1');
    expect(result.openEvents.single.id, 'E1');
    expect(result.fetchedAt, at);
    expect(result.isFromCache, isFalse);
  });
}
