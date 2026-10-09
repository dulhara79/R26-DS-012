import 'package:flutter/material.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../domain/contracts/assessment_summary.dart';
import 'assessment_detail_presentation.dart';

class DataQualityScreen extends StatelessWidget {
  final AssessmentSummary assessment;
  final String? displayId;

  const DataQualityScreen({
    super.key,
    required this.assessment,
    this.displayId,
  });

  @override
  Widget build(BuildContext context) {
    final byId = {
      for (final modality in assessment.modalities)
        modality.componentId: modality,
    };

    return Scaffold(
      appBar: AppBar(title: const Text('Data quality')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          Text(
            displayId ?? assessment.subjectId,
            style: AppTheme.display(size: 18),
          ),
          if (displayId != null) ...[
            const SizedBox(height: Ds.s1),
            Text(
              assessment.subjectId,
              style: AppTheme.data(size: 11, color: Ds.inkFaint),
            ),
          ],
          const SizedBox(height: Ds.s5),
          const SectionLabel('Assessment data quality'),
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Assessment: ${p5aAssessmentStatusLabel(assessment.assessmentStatus)}',
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    color: Ds.ink,
                  ),
                ),
                const SizedBox(height: Ds.s2),
                Text(
                  assessment.fusionResultId == null
                      ? 'Fusion result: Not reported'
                      : 'Fusion result: #${assessment.fusionResultId}',
                  style: const TextStyle(color: Ds.inkMuted),
                ),
                const SizedBox(height: Ds.s1),
                Text(
                  'Computed: ${p5aReportedTime(assessment.computedAt)}',
                  style: const TextStyle(color: Ds.inkMuted),
                ),
                const SizedBox(height: Ds.s1),
                Text(
                  assessment.modelVersion == null
                      ? 'Model: Not reported'
                      : 'Model: ${assessment.modelVersion}',
                  style: const TextStyle(color: Ds.inkMuted),
                ),
              ],
            ),
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Modality quality'),
          for (final componentId in p5aComponentOrder)
            Padding(
              padding: const EdgeInsets.only(bottom: Ds.s3),
              child: _DataQualityCard(
                componentId: componentId,
                modality: byId[componentId],
              ),
            ),
          const SizedBox(height: Ds.s3),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }
}

class _DataQualityCard extends StatelessWidget {
  final String componentId;
  final ModalityStatus? modality;

  const _DataQualityCard({
    required this.componentId,
    required this.modality,
  });

  @override
  Widget build(BuildContext context) {
    final value = modality;
    return Panel(
      padding: const EdgeInsets.all(Ds.s3),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            p5aModalityLabel(componentId),
            style: const TextStyle(
              fontWeight: FontWeight.w700,
              color: Ds.ink,
            ),
          ),
          const SizedBox(height: Ds.s2),
          Text(
            'Availability: ${_availabilityLabel(value)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Status: ${p5aModalityStateLabel(value)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Score: ${p5aScoreLabel(value?.score)}',
            style: AppTheme.data(size: 12.5, color: Ds.ink),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Confidence: ${p5aReportedDecimal(value?.confidence)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Coverage: ${p5aReportedDecimal(value?.coverage)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Captured: ${p5aReportedTime(value?.capturedAt)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Fusion inclusion: ${p5aInclusionLabel(value?.includedInFusion)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          const Text(
            'Reason detail: Not reported by backend',
            style: TextStyle(fontSize: 11.5, color: Ds.inkFaint),
          ),
        ],
      ),
    );
  }
}

String _availabilityLabel(ModalityStatus? modality) {
  if (modality == null) return 'Unavailable';
  return switch (modality.available) {
    true => 'Available',
    false => 'Unavailable',
    null => 'Availability not reported',
  };
}
