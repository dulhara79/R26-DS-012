import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';
import '../widgets/risk_badge.dart';
import '../services/share_service.dart';
import '../services/pdf_service.dart';

class AssessmentResultScreen extends StatelessWidget {
  final Patient patient;
  final String noteText;
  final String noteType;
  final PredictionResult result;

  const AssessmentResultScreen({
    super.key,
    required this.patient,
    required this.noteText,
    required this.noteType,
    required this.result,
  });

  Color get _headerColor {
    switch (result.riskLevel) {
      case RiskLevel.veryHigh: return AppColors.riskVeryHigh;
      case RiskLevel.high:     return AppColors.riskHigh;
      case RiskLevel.moderate: return AppColors.riskModerate;
      case RiskLevel.low:      return AppColors.riskLow;
    }
  }

  Color get _headerBg {
    switch (result.riskLevel) {
      case RiskLevel.veryHigh: return AppColors.riskVeryHighBg;
      case RiskLevel.high:     return AppColors.riskHighBg;
      case RiskLevel.moderate: return AppColors.riskModerateBg;
      case RiskLevel.low:      return AppColors.riskLowBg;
    }
  }

  IconData get _riskIcon {
    switch (result.riskLevel) {
      case RiskLevel.veryHigh: return Icons.report_problem_rounded;
      case RiskLevel.high:     return Icons.warning_amber_rounded;
      case RiskLevel.moderate: return Icons.info_outline_rounded;
      case RiskLevel.low:      return Icons.check_circle_outline_rounded;
    }
  }

  Assessment get _tempAssessment => Assessment(
    id: 'TEMP_${DateTime.now().millisecondsSinceEpoch}',
    patientId: patient.id,
    timestamp: DateTime.now(),
    noteText: noteText,
    noteType: noteType,
    clinicianId: 'DR001',
    result: result,
  );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(
        title: const Text('Assessment result'),
        leading: IconButton(
          icon: const Icon(Icons.close_rounded),
          onPressed: () {
            // Pop back to patient detail
            Navigator.of(context)
              ..pop()
              ..pop();
          },
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_rounded),
            onPressed: () {
              ShareService.shareAssessment(patient, _tempAssessment);
              context.read<PatientProvider>().addNotification(
                title: 'Assessment Shared',
                body: 'Clinical summary for ${patient.name} was shared.',
                type: NotificationType.info,
                patientId: patient.id,
                patientName: patient.name,
              );
            },
            tooltip: 'Share',
          ),
          IconButton(
            icon: const Icon(Icons.picture_as_pdf_rounded),
            onPressed: () async {
              try {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Preparing PDF report...')),
                );
                await PdfService.generateAndSavePdf(patient, _tempAssessment);
                if (context.mounted) {
                  context.read<PatientProvider>().addNotification(
                    title: 'PDF Report Generated',
                    body: 'A medical-standard PDF was generated for ${patient.name}.',
                    type: NotificationType.info,
                    patientId: patient.id,
                    patientName: patient.name,
                  );
                }
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Error generating PDF: $e')),
                );
              }
            },
            tooltip: 'Download PDF',
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Main result hero card
            Container(
              decoration: BoxDecoration(
                color: _headerBg,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: _headerColor.withOpacity(0.3),
                  width: 1,
                ),
              ),
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(_riskIcon, color: _headerColor, size: 24),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          result.prediction,
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                            color: _headerColor,
                          ),
                        ),
                      ),
                      RiskBadge(risk: result.riskLevel, large: true),
                    ],
                  ),
                  const SizedBox(height: 20),
                  RiskScoreBar(
                    score:     result.riskScore,
                    threshold: result.threshold,
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      _StatPill(
                        label: 'Confidence',
                        value:
                            '${(result.confidence * 100).toStringAsFixed(1)}%',
                      ),
                      const SizedBox(width: 8),
                      _StatPill(
                        label: 'Latency',
                        value: '${result.latencyMs} ms',
                      ),
                      const SizedBox(width: 8),
                      _StatPill(
                        label: 'Support K',
                        value: '${result.supportK}',
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Patient context
            _SectionCard(
              title: 'Patient context',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _Row('Patient', patient.name),
                  _Row('Note type', noteType),
                  if (result.temporalContext.isNotEmpty)
                    _Row('Visit context', result.temporalContext),
                  _Row('Threshold', result.threshold.toStringAsFixed(4)),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // Key phrases (XAI)
            _SectionCard(
              title: 'Key phrases — attention-based explanation',
              subtitle:
                  'Phrases highlighted by ClinicalBERT attention weights',
              child: result.keyPhrases.isEmpty
                  ? Text(
                      'No key phrases identified',
                      style: Theme.of(context).textTheme.bodySmall,
                    )
                  : Wrap(
                      spacing: 8, runSpacing: 8,
                      children: result.keyPhrases
                          .asMap()
                          .entries
                          .map((e) => _PhraseChip(
                                phrase: e.value,
                                prominence: 1.0 -
                                    (e.key /
                                        (result.keyPhrases.length + 1)),
                              ))
                          .toList(),
                    ),
            ),
            const SizedBox(height: 12),

            // TC-WPN model info
            _SectionCard(
              title: 'Model information',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _Row('Model', 'TC-WPN v1.0'),
                  _Row('Backbone', 'Bio_ClinicalBERT'),
                  _Row('Val AUROC', '0.9671'),
                  _Row('Test AUROC', '0.9635 (K=5, high-conf)'),
                  _Row('Training', 'MIMIC-IV + MIMIC-III'),
                  _Row('Adaptation', 'NHSL finetuning pending'),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // Note preview
            _SectionCard(
              title: 'Submitted note',
              child: Text(
                noteText.length > 300
                    ? '${noteText.substring(0, 300)}...'
                    : noteText,
                style: Theme.of(context).textTheme.bodySmall
                    ?.copyWith(height: 1.6),
              ),
            ),
            const SizedBox(height: 16),

            // Mandatory clinical disclaimer
            Container(
              decoration: BoxDecoration(
                color: AppColors.riskHighBg,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                    color: AppColors.riskHigh.withOpacity(0.3)),
              ),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.shield_outlined,
                          size: 18, color: AppColors.riskHigh),
                      const SizedBox(width: 8),
                      Text(
                        'Clinical safety notice',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: AppColors.riskHigh,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'This prediction is generated by an AI model for clinical '
                    'decision support purposes only. It does not constitute a '
                    'formal diagnosis. All clinical decisions — including '
                    'diagnosis, treatment, and referral — remain the sole '
                    'responsibility of the responsible psychiatrist.\n\n'
                    'Confidence below 60% or predictions conflicting with '
                    'clinical judgment should be reviewed manually.',
                    style: TextStyle(
                      fontSize: 13,
                      color: AppColors.riskHigh,
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Actions
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.of(context)
                        ..pop()
                        ..pop();
                    },
                    icon: const Icon(Icons.arrow_back_rounded, size: 18),
                    label: const Text('Back to patient'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {},
                    icon: const Icon(Icons.edit_note_rounded, size: 18),
                    label: const Text('Add comment'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget child;

  const _SectionCard({
    required this.title,
    this.subtitle,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border, width: 0.8),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleLarge),
          if (subtitle != null) ...[
            const SizedBox(height: 2),
            Text(subtitle!,
                style: Theme.of(context).textTheme.bodySmall),
          ],
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  const _Row(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Expanded(
            flex: 2,
            child: Text(label,
                style: const TextStyle(
                    fontSize: 13, color: AppColors.textSecondary)),
          ),
          Expanded(
            flex: 3,
            child: Text(value,
                style: const TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w500)),
          ),
        ],
      ),
    );
  }
}

class _StatPill extends StatelessWidget {
  final String label;
  final String value;
  const _StatPill({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.6),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.white.withOpacity(0.4)),
        ),
        child: Column(
          children: [
            Text(value,
                style: const TextStyle(
                    fontSize: 14, fontWeight: FontWeight.w700)),
            Text(label,
                style: const TextStyle(
                    fontSize: 10, color: AppColors.textSecondary)),
          ],
        ),
      ),
    );
  }
}

class _PhraseChip extends StatelessWidget {
  final String phrase;
  final double prominence;  // 0.0 to 1.0
  const _PhraseChip({required this.phrase, required this.prominence});

  @override
  Widget build(BuildContext context) {
    final intensity = (prominence * 0.6 + 0.2).clamp(0.2, 0.8);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: AppColors.riskModerateBg.withOpacity(intensity + 0.4),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppColors.riskModerate.withOpacity(intensity),
        ),
      ),
      child: Text(
        phrase,
        style: TextStyle(
          fontSize: 12,
          fontWeight: prominence > 0.7 ? FontWeight.w700 : FontWeight.w500,
          color: AppColors.riskModerate,
        ),
      ),
    );
  }
}
