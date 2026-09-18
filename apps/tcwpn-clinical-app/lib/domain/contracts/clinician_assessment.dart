import 'package:flutter/foundation.dart';

const Set<String> clinicianAssessmentTierLabels = {
  'Low',
  'Medium',
  'High',
};

@immutable
class ClinicianAssessmentDraft {
  final int fusionResultId;
  final String tierLabel;
  final String? author;
  final String? note;

  const ClinicianAssessmentDraft({
    required this.fusionResultId,
    required this.tierLabel,
    this.author,
    this.note,
  }) : assert(
          tierLabel == 'Low' || tierLabel == 'Medium' || tierLabel == 'High',
          'tierLabel must be Low, Medium, or High',
        );

  factory ClinicianAssessmentDraft.validated({
    required int fusionResultId,
    required String tierLabel,
    String? author,
    String? note,
  }) {
    if (!clinicianAssessmentTierLabels.contains(tierLabel)) {
      throw ArgumentError.value(
        tierLabel,
        'tierLabel',
        'must be one of Low, Medium, High',
      );
    }
    return ClinicianAssessmentDraft(
      fusionResultId: fusionResultId,
      tierLabel: tierLabel,
      author: author,
      note: note,
    );
  }

  Map<String, dynamic> toRequestJson() => {
        'fusion_result_id': fusionResultId,
        'tier_label': tierLabel,
        if ((author ?? '').trim().isNotEmpty) 'author': author!.trim(),
        if ((note ?? '').trim().isNotEmpty) 'note': note!.trim(),
      };
}

@immutable
class ClinicianAssessmentReceipt {
  final int? verdictId;
  final String? subjectId;
  final int fusionResultId;
  final String tierLabel;
  final bool? agreesWithModel;
  final int? calibrationLabelsTotal;
  final bool? conformalCalibrated;
  final String? author;
  final String? note;

  const ClinicianAssessmentReceipt({
    required this.verdictId,
    required this.subjectId,
    required this.fusionResultId,
    required this.tierLabel,
    required this.agreesWithModel,
    required this.calibrationLabelsTotal,
    required this.conformalCalibrated,
    required this.author,
    required this.note,
  });

  factory ClinicianAssessmentReceipt.fromJson(
    Map<String, dynamic> json, {
    required ClinicianAssessmentDraft request,
  }) =>
      ClinicianAssessmentReceipt(
        verdictId: _intOrNull(json['verdict_id']),
        subjectId: json['subject_id']?.toString(),
        fusionResultId: request.fusionResultId,
        tierLabel: request.tierLabel,
        agreesWithModel: json['agrees_with_model'] is bool
            ? json['agrees_with_model'] as bool
            : null,
        calibrationLabelsTotal: _intOrNull(json['calibration_labels_total']),
        conformalCalibrated: json['conformal_calibrated'] is bool
            ? json['conformal_calibrated'] as bool
            : null,
        author: request.author,
        note: request.note,
      );

  Map<String, dynamic> toJson() => {
        'verdict_id': verdictId,
        'subject_id': subjectId,
        'fusion_result_id': fusionResultId,
        'tier_label': tierLabel,
        'agrees_with_model': agreesWithModel,
        'calibration_labels_total': calibrationLabelsTotal,
        'conformal_calibrated': conformalCalibrated,
        'author': author,
        'note': note,
      };
}

int? _intOrNull(dynamic value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  return int.tryParse(value?.toString() ?? '');
}
