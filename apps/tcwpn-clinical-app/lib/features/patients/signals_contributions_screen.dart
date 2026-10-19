import 'package:flutter/material.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../domain/contracts/assessment_summary.dart';
import 'assessment_detail_presentation.dart';

class SignalsContributionsScreen extends StatelessWidget {
  final AssessmentSummary assessment;
  final String? displayId;

  const SignalsContributionsScreen({
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
      appBar: AppBar(title: const Text('Signals & contributions')),
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
          const SectionLabel('Assessment provenance'),
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  assessment.fusionResultId == null
                      ? 'Fusion result ID not reported'
                      : 'Fusion result #${assessment.fusionResultId}',
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    color: Ds.ink,
                  ),
                ),
                const SizedBox(height: Ds.s1),
                Text(
                  p5aAssessmentStatusLabel(assessment.assessmentStatus),
                  style: const TextStyle(color: Ds.inkMuted),
                ),
                const SizedBox(height: Ds.s1),
                Text(
                  'Computed ${p5aReportedTime(assessment.computedAt)}',
                  style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                ),
                const SizedBox(height: Ds.s1),
                Text(
                  assessment.modelVersion == null
                      ? 'Model: Not reported'
                      : 'Model ${assessment.modelVersion}',
                  style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                ),
              ],
            ),
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Signals & contributions'),
          for (final componentId in p5aComponentOrder)
            Padding(
              padding: const EdgeInsets.only(bottom: Ds.s3),
              child: _SignalContributionCard(
                componentId: componentId,
                modality: byId[componentId],
              ),
            ),
          const SizedBox(height: Ds.s2),
          const InlineNotice(
            icon: Icons.info_outline_rounded,
            text:
                "Contribution reflects the value reported by the fusion backend for this assessment. It does not establish that a modality caused the patient's state.",
          ),
          const SizedBox(height: Ds.s5),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }
}

class _SignalContributionCard extends StatelessWidget {
  final String componentId;
  final ModalityStatus? modality;

  const _SignalContributionCard({
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
            'Score: ${p5aScoreLabel(value?.score)}',
            style: AppTheme.data(size: 12.5, color: Ds.ink),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Status: ${p5aModalityStateLabel(value)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Fusion: ${p5aInclusionLabel(value?.includedInFusion)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Contribution: ${p5aReportedDecimal(value?.contribution)}',
            style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'Captured: ${p5aReportedTime(value?.capturedAt)}',
            style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
          ),
        ],
      ),
    );
  }
}
