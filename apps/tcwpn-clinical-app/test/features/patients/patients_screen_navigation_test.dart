import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/features/patients/patients_screen.dart';
import 'package:r26_ds012_app/state/controllers.dart';

Patient _patient(String id) => Patient(
      mrn: id,
      name: 'Patient A',
      age: 24,
      gender: 'Female',
      referredOn: DateTime.utc(2026, 9, 1),
    );

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  testWidgets(
      'legacy Patients resolves canonical subject before opening Patient Overview',
      (tester) async {
    final roster = RosterController();
    final patient = _patient('P_65DC4002E7863773');
    await roster.addPatient(patient);
    String? resolvedInput;
    String? openedSubject;
    Patient? openedPatient;
    var legacyOpened = false;

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: roster,
        child: MaterialApp(
          home: PatientsScreen(
            resolveSubjectId: (value) async {
              resolvedInput = value;
              return 'subject-001';
            },
            onOpenOverview: (context, patient, subjectId) {
              openedPatient = patient;
              openedSubject = subjectId;
            },
            onOpenLegacyChart: (context, patient) {
              legacyOpened = true;
            },
          ),
        ),
      ),
    );

    await tester.tap(find.text('Patient A'));
    await tester.pumpAndSettle();

    expect(resolvedInput, patient.mrn);
    expect(openedSubject, 'subject-001');
    expect(openedPatient, same(patient));
    expect(legacyOpened, isFalse);
  });

  testWidgets('unresolved legacy patient falls back to existing chart path',
      (tester) async {
    final roster = RosterController();
    final patient = _patient('LOCAL-001');
    await roster.addPatient(patient);
    var overviewOpened = false;
    Patient? legacyPatient;

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: roster,
        child: MaterialApp(
          home: PatientsScreen(
            resolveSubjectId: (_) async => null,
            onOpenOverview: (context, patient, subjectId) {
              overviewOpened = true;
            },
            onOpenLegacyChart: (context, patient) {
              legacyPatient = patient;
            },
          ),
        ),
      ),
    );

    await tester.tap(find.text('Patient A'));
    await tester.pumpAndSettle();

    expect(overviewOpened, isFalse);
    expect(legacyPatient, same(patient));
  });

  testWidgets('resolver failure never fabricates a canonical subject id',
      (tester) async {
    final roster = RosterController();
    final patient = _patient('LOCAL-002');
    await roster.addPatient(patient);
    String? openedSubject;
    Patient? legacyPatient;

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: roster,
        child: MaterialApp(
          home: PatientsScreen(
            resolveSubjectId: (_) async => throw StateError('offline'),
            onOpenOverview: (context, patient, subjectId) {
              openedSubject = subjectId;
            },
            onOpenLegacyChart: (context, patient) {
              legacyPatient = patient;
            },
          ),
        ),
      ),
    );

    await tester.tap(find.text('Patient A'));
    await tester.pumpAndSettle();

    expect(openedSubject, isNull);
    expect(legacyPatient, same(patient));
  });
}
