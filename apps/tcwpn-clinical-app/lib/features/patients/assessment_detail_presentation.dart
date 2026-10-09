import 'package:intl/intl.dart';

import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/contract_enums.dart';

const p5aComponentOrder = <String>[
  'c1_physiological',
  'c2_behavioral',
  'c3_clinical_nlp',
  'c4_demographic',
];

String p5aModalityLabel(String componentId) => switch (componentId) {
      'c1_physiological' => 'Physiological',
      'c2_behavioral' => 'Behavioural',
      'c3_clinical_nlp' => 'Clinical NLP / TC-WPN',
      'c4_demographic' => 'Contextual',
      _ => componentId,
    };

String p5aAssessmentStatusLabel(AssessmentStatus status) => switch (status) {
      AssessmentStatus.complete => 'Complete assessment',
      AssessmentStatus.partial => 'Partial assessment',
      AssessmentStatus.unavailable => 'Assessment unavailable',
      AssessmentStatus.unknown => 'Assessment status unknown',
    };

String p5aModalityStateLabel(ModalityStatus? modality) {
  if (modality == null) return 'Unavailable';
  if (modality.isExperimentalExcluded) {
    return 'Experimental — not included in fusion';
  }
  return switch (modality.state) {
    ModalityState.ok => 'Available',
    ModalityState.stale => 'Stale',
    ModalityState.notValidated => 'Experimental / not validated',
    ModalityState.unavailable => 'Unavailable',
    ModalityState.error => 'Service error',
    ModalityState.unknown => 'Status unknown',
  };
}

String p5aInclusionLabel(bool? includedInFusion) => switch (includedInFusion) {
      true => 'Included in fusion',
      false => 'Not included in fusion',
      null => 'Inclusion not reported',
    };

String p5aScoreLabel(double? value) =>
    value == null || !value.isFinite ? '—' : value.toStringAsFixed(2);

String p5aReportedDecimal(double? value) =>
    value == null || !value.isFinite ? 'Not reported' : value.toStringAsFixed(2);

String p5aReportedTime(DateTime? value) => value == null
    ? 'Not reported'
    : DateFormat('d MMM y, HH:mm').format(value.toLocal());
