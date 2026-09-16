import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  const p5bRepository = 'lib/data/repositories/p5b_repositories.dart';
  const timelineScreen = 'lib/features/patients/timeline_screen.dart';
  const notesScreen = 'lib/features/patients/clinical_notes_screen.dart';
  const timelineController = 'lib/state/timeline_controller.dart';
  const notesController = 'lib/state/clinical_notes_controller.dart';
  const patientsScreen = 'lib/features/patients/patients_screen.dart';

  test('P5B uses only verified timeline and clinical-note network routes', () {
    final source = File(p5bRepository).readAsStringSync();

    expect(source, contains('/v1/doctor/patients/'));
    expect(source, contains('/timeline?limit='));
    expect(source, isNot(contains('/clinical-notes/history')));
    expect(source, isNot(contains('/notes/history')));
    expect(source, isNot(contains("'/predict'")));
    expect(source, isNot(contains('"/predict"')));
  });

  test('P5B views and controllers do not derive authoritative risk locally', () {
    final source = [
      File(timelineScreen).readAsStringSync(),
      File(notesScreen).readAsStringSync(),
      File(timelineController).readAsStringSync(),
      File(notesController).readAsStringSync(),
    ].join('\n');

    expect(source, isNot(contains('AlertBandX.fromScore')));
    expect(source, isNot(contains('weight * score')));
    expect(source, isNot(contains('FusionResult')));
    expect(source, isNot(contains("'/predict'")));
    expect(source, isNot(contains('"/predict"')));
    expect(source, isNot(contains('largest weight')));
  });

  test('P5B clinical notes preserve subordinate C3 wording', () {
    final source = File(notesScreen).readAsStringSync();

    expect(source, contains('Clinical NLP / TC-WPN signal'));
    expect(source, contains('device-local'));
    expect(source, isNot(contains('TC-WPN Risk')));
    expect(source, isNot(contains('Overall risk')));
  });

  test('Patients navigation keeps canonical subject and local record identity separate', () {
    final source = File(patientsScreen).readAsStringSync();

    expect(source, contains('subjectId: subjectId!'));
    expect(source, contains('localRecordId: patient.mrn'));
  });
}
