import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/contracts/clinician_assessment.dart';

void main() {
  group('ClinicianAssessmentDraft', () {
    test('accepts only the verified Low Medium High tier values', () {
      for (final tier in const ['Low', 'Medium', 'High']) {
        final draft = ClinicianAssessmentDraft.validated(
          fusionResultId: 37,
          tierLabel: tier,
          author: 'dr-1',
          note: 'Observed during review.',
        );
        expect(draft.tierLabel, tier);
      }
    });

    test('rejects unknown tier values instead of normalising them', () {
      expect(
        () => ClinicianAssessmentDraft.validated(
          fusionResultId: 37,
          tierLabel: 'Critical',
          author: 'dr-1',
        ),
        throwsArgumentError,
      );
    });

    test('preserves the exact fusion row and optional fields', () {
      const draft = ClinicianAssessmentDraft(
        fusionResultId: 91,
        tierLabel: 'Medium',
        author: 'dr-9',
        note: 'Review note',
      );

      expect(draft.fusionResultId, 91);
      expect(draft.author, 'dr-9');
      expect(draft.note, 'Review note');
    });
  });

  group('ClinicianAssessmentReceipt', () {
    test('parses only the verified server receipt fields', () {
      final receipt = ClinicianAssessmentReceipt.fromJson(
        const {
          'verdict_id': 12,
          'subject_id': 'subject-1',
          'agrees_with_model': false,
          'calibration_labels_total': 47,
          'conformal_calibrated': true,
        },
        request: const ClinicianAssessmentDraft(
          fusionResultId: 37,
          tierLabel: 'High',
          author: 'dr-1',
          note: 'Clinical observation',
        ),
      );

      expect(receipt.verdictId, 12);
      expect(receipt.subjectId, 'subject-1');
      expect(receipt.fusionResultId, 37);
      expect(receipt.tierLabel, 'High');
      expect(receipt.agreesWithModel, isFalse);
      expect(receipt.calibrationLabelsTotal, 47);
      expect(receipt.conformalCalibrated, isTrue);
      expect(receipt.author, 'dr-1');
      expect(receipt.note, 'Clinical observation');
    });

    test('keeps optional server comparison fields nullable', () {
      final receipt = ClinicianAssessmentReceipt.fromJson(
        const {
          'verdict_id': 13,
          'subject_id': 'subject-2',
        },
        request: const ClinicianAssessmentDraft(
          fusionResultId: 38,
          tierLabel: 'Low',
        ),
      );

      expect(receipt.agreesWithModel, isNull);
      expect(receipt.calibrationLabelsTotal, isNull);
      expect(receipt.conformalCalibrated, isNull);
      expect(receipt.author, isNull);
      expect(receipt.note, isNull);
    });

    test('has no fabricated generated or created timestamp field', () {
      final receipt = ClinicianAssessmentReceipt.fromJson(
        const {
          'verdict_id': 14,
          'subject_id': 'subject-3',
        },
        request: const ClinicianAssessmentDraft(
          fusionResultId: 39,
          tierLabel: 'Medium',
        ),
      );

      expect(receipt.toJson().containsKey('created_at'), isFalse);
      expect(receipt.toJson().containsKey('generated_at'), isFalse);
    });
  });
}
