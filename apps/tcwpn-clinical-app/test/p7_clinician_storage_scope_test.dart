import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:r26_ds012_app/data/api/session.dart';
import 'package:r26_ds012_app/data/local/attention_notification_store.dart';
import 'package:r26_ds012_app/data/local/dashboard_cache.dart';
import 'package:r26_ds012_app/data/local/stores.dart';
import 'package:r26_ds012_app/domain/contracts/dashboard_snapshot.dart';
import 'package:r26_ds012_app/domain/contracts/patient_summary.dart';
import 'package:r26_ds012_app/domain/contracts/assessment_summary.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';
import 'package:r26_ds012_app/domain/models.dart';

Patient _patient(String mrn, String name) => Patient(
      mrn: mrn,
      name: name,
      age: 24,
      gender: 'Female',
      referredOn: DateTime.utc(2026, 9, 1),
    );

DashboardSnapshot _snapshot(String subjectId, int fusionId) => DashboardSnapshot(
      openEvents: const [],
      assignedPatients: [
        PatientSummary(
          subjectId: subjectId,
          displayId: subjectId,
          fusionResultId: fusionId,
          currentAssessment: const CurrentAssessment(
            score: null,
            tier: RiskTier.unknown,
            band: null,
          ),
          forecast: null,
          assessmentStatus: AssessmentStatus.partial,
          lastUpdated: DateTime.utc(2026, 9, 17),
          openEventCount: 0,
        ),
      ],
      fetchedAt: DateTime.utc(2026, 9, 17),
    );

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
    Session.clear();
  });

  tearDown(Session.clear);

  test('RecordStore does not expose DR001 roster after DR002 signs in', () async {
    Session.set(token: 'token-a', clinicianId: 'DR001');
    await RecordStore.saveRoster([_patient('A-001', 'Patient A')]);

    Session.set(token: 'token-b', clinicianId: 'DR002');
    await RecordStore.saveRoster([_patient('B-001', 'Patient B')]);
    final drB = await RecordStore.loadRoster();

    Session.set(token: 'token-a2', clinicianId: 'DR001');
    final drA = await RecordStore.loadRoster();

    expect(drB.map((p) => p.mrn), ['B-001']);
    expect(drA.map((p) => p.mrn), ['A-001']);

    final keys = (await SharedPreferences.getInstance()).getKeys();
    expect(keys, isNot(contains('roster_v2')),
        reason: 'Legacy unscoped clinical storage must not remain authoritative.');
  });

  test('dashboard cache is isolated between clinicians', () async {
    const store = DashboardCacheStore();

    Session.set(token: 'token-a', clinicianId: 'DR001');
    await store.save(_snapshot('SUBJECT-A', 101));

    Session.set(token: 'token-b', clinicianId: 'DR002');
    await store.save(_snapshot('SUBJECT-B', 202));
    final drB = await store.load();

    Session.set(token: 'token-a2', clinicianId: 'DR001');
    final drA = await store.load();

    expect(drB!.assignedPatients.single.subjectId, 'SUBJECT-B');
    expect(drA!.assignedPatients.single.subjectId, 'SUBJECT-A');
  });

  test('pending notification open is scoped to the clinician', () async {
    const drA = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR001');
    const drB = SharedPreferencesAttentionNotificationStore(deliveryScope: 'DR002');

    await drA.savePendingOpen('event-for-a');

    expect(await drB.takePendingOpen(), isNull);
    expect(await drA.takePendingOpen(), 'event-for-a');
  });
}
