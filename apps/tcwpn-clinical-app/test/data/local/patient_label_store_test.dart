import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/local/patient_label_store.dart';

class _Labels implements PatientLabelStore {
  final values = <String, String>{};
  String key(String clinician, String subject) => '$clinician::$subject';

  @override
  Future<String?> read({required String clinicianId, required String subjectId}) async =>
      values[key(clinicianId, subjectId)];

  @override
  Future<void> write({
    required String clinicianId,
    required String subjectId,
    required String label,
  }) async {
    final error = PatientLabelStore.validateLabel(label);
    if (error != null) throw ArgumentError(error);
    values[key(clinicianId, subjectId)] = PatientLabelStore.normalizeLabel(label);
  }

  @override
  Future<void> delete({required String clinicianId, required String subjectId}) async {
    values.remove(key(clinicianId, subjectId));
  }
}

void main() {
  test('only clinician-entered labels are stored and can be removed', () async {
    final s = _Labels();
    await s.write(clinicianId: 'DR001', subjectId: 'subject-a', label: '  Participant   Anna  ');
    expect(await s.read(clinicianId: 'DR001', subjectId: 'subject-a'), 'Participant Anna');
    expect(await s.read(clinicianId: 'DR002', subjectId: 'subject-a'), isNull);
    expect(await s.read(clinicianId: 'DR001', subjectId: 'subject-b'), isNull);
    await s.delete(clinicianId: 'DR001', subjectId: 'subject-a');
    expect(await s.read(clinicianId: 'DR001', subjectId: 'subject-a'), isNull);
  });

  test('empty, too long and control-character labels cannot be saved', () {
    expect(PatientLabelStore.validateLabel('   '), isNotNull);
    expect(PatientLabelStore.validateLabel('x' * 81), isNotNull);
    expect(PatientLabelStore.validateLabel('a\nb'), isNotNull);
    expect(PatientLabelStore.validateLabel('participant A'), isNull);
  });
}
