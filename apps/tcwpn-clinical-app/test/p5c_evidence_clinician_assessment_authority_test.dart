import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

String _read(String path) => File(path).readAsStringSync();

void main() {
  test('P5C repositories use gateway boundaries and no direct CARE endpoint', () {
    final source = _read('lib/data/repositories/p5c_repositories.dart');
    expect(source, contains('_gateway.evidence'));
    expect(source, contains('_gateway.submitVerdict'));
    expect(source, isNot(contains('/v1/evidence/ask')));
    expect(source, isNot(contains('/v1/ask')));
    expect(source, isNot(contains('Care-AnxRAG')));
    expect(source, isNot(contains('/predict')));
  });

  test('P5C controllers never derive or mutate model authority', () {
    final evidence = _read('lib/state/supporting_evidence_controller.dart');
    final assessment = _read('lib/state/clinician_assessment_controller.dart');
    final combined = '$evidence\n$assessment';

    expect(combined, isNot(contains('DateTime.now')),
        reason: 'P5C must not fabricate evidence or verdict timestamps');
    expect(combined, isNot(contains('runFusion')));
    expect(combined, isNot(contains('composite')));
    expect(combined, isNot(contains('weight * score')));
    expect(combined, isNot(contains('AlertBandX.fromScore')));
  });

  test('P5C screens contain no guessed evidence or verdict history route', () {
    final evidence = _read('lib/features/patients/supporting_evidence_screen.dart');
    final assessment = _read('lib/features/patients/clinician_assessment_screen.dart');
    final combined = '$evidence\n$assessment';

    expect(combined, isNot(contains('/verdicts')));
    expect(combined, isNot(contains('/verdict/history')));
    expect(combined, isNot(contains('/evidence/history')));
    expect(combined, isNot(contains('/predict')));
    expect(combined, isNot(contains('/fuse')));
    expect(combined, isNot(contains('DateTime.now')));
  });

  test('patient-scoped evidence transport sends question only', () {
    final gateway = _read('lib/data/api/gateways.dart');
    final start = gateway.indexOf('Future<Map<String, dynamic>> evidence(');
    final end = gateway.indexOf('Future<EvidenceResult> askEvidence', start);
    expect(start, greaterThanOrEqualTo(0));
    expect(end, greaterThan(start));
    final method = gateway.substring(start, end);

    expect(method, contains('/v1/doctor/patients/'));
    expect(method, contains("{'question': trimmed}"));
    expect(method, isNot(contains('fusion_result_id')));
    expect(method, isNot(contains('note_text')));
    expect(method, isNot(contains('patient_data')));
  });
}
