import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';
import '../widgets/risk_badge.dart';
import '../services/share_service.dart';
import '../services/pdf_service.dart';

class AssessmentDetailScreen extends StatelessWidget {
  final Assessment assessment;
  final Patient patient;

  const AssessmentDetailScreen({
    super.key,
    required this.assessment,
    required this.patient,
  });

  @override
  Widget build(BuildContext context) {
    final r = assessment.result;

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(
        title: const Text('Assessment Detail'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_rounded),
            onPressed: () {
              ShareService.shareAssessment(patient, assessment);
              context.read<PatientProvider>().addNotification(
                title: 'Assessment Shared',
                body: 'Clinical summary for ${patient.name} was shared.',
                type: NotificationType.info,
                patientId: patient.id,
                patientName: patient.name,
              );
            },
          ),
          IconButton(
            icon: const Icon(Icons.picture_as_pdf_rounded),
            onPressed: () async {
              try {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Preparing PDF report...')),
                );
                await PdfService.generateAndSavePdf(patient, assessment);
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
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Patient Header
            _buildPatientHeader(),
            const SizedBox(height: 20),

            // Note Content
            _SectionHeader('Clinical Note', Icons.notes_rounded),
            const SizedBox(height: 10),
            _buildNoteCard(),
            const SizedBox(height: 24),

            // Prediction Result
            if (r != null) ...[
              _SectionHeader('AI Analysis', Icons.analytics_rounded),
              const SizedBox(height: 10),
              _buildResultCard(context, r),
              const SizedBox(height: 24),

              _SectionHeader('Key Indicators', Icons.psychology_rounded),
              const SizedBox(height: 10),
              _buildKeyPhrases(r),
            ],

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _buildPatientHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.primary,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.3),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 24,
            backgroundColor: Colors.white.withOpacity(0.2),
            child: Text(
              patient.initials,
              style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  patient.name,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'Assessment from ${_formatDateTime(assessment.timestamp)}',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.8),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          if (assessment.result != null)
            RiskBadge(risk: assessment.result!.riskLevel, large: true),
        ],
      ),
    ).animate().fadeIn().slideY(begin: 0.1);
  }

  Widget _buildNoteCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                assessment.noteType,
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.primary,
                ),
              ),
              const Spacer(),
              const Icon(Icons.verified_user_rounded, size: 14, color: AppColors.info),
              const SizedBox(width: 4),
              Text(
                'Clinician: ${assessment.clinicianId}',
                style: const TextStyle(fontSize: 11, color: AppColors.textHint),
              ),
            ],
          ),
          const Divider(height: 24),
          Text(
            assessment.noteText,
            style: const TextStyle(
              fontSize: 15,
              height: 1.6,
              color: AppColors.textPrimary,
            ),
          ),
        ],
      ),
    ).animate().fadeIn(delay: 100.ms).slideY(begin: 0.1);
  }

  Widget _buildResultCard(BuildContext context, PredictionResult r) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.8),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Prediction', style: TextStyle(fontSize: 12, color: AppColors.textHint)),
                  Text(r.prediction, style: Theme.of(context).textTheme.headlineSmall),
                ],
              ),
              _buildMetricCircle(r.confidence, 'Confidence'),
            ],
          ),
          const SizedBox(height: 20),
          _ResultRow('Risk Score', r.riskScore.toStringAsFixed(4)),
          _ResultRow('Decision Threshold', r.threshold.toStringAsFixed(4)),
          _ResultRow('Processing Latency', '${r.latencyMs}ms'),
          if (r.temporalContext.isNotEmpty)
            _ResultRow('Temporal Context', r.temporalContext),
        ],
      ),
    ).animate().fadeIn(delay: 200.ms).slideY(begin: 0.1);
  }

  Widget _buildMetricCircle(double value, String label) {
    return Column(
      children: [
        SizedBox(
          width: 50, height: 50,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CircularProgressIndicator(
                value: value,
                strokeWidth: 4,
                backgroundColor: AppColors.border,
                valueColor: const AlwaysStoppedAnimation<Color>(AppColors.primary),
              ),
              Text('${(value * 100).toInt()}%',
                  style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
            ],
          ),
        ),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 10, color: AppColors.textHint)),
      ],
    );
  }

  Widget _buildKeyPhrases(PredictionResult r) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: r.keyPhrases.map((phrase) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: AppColors.primarySurface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: AppColors.primary.withOpacity(0.1)),
        ),
        child: Text(
          phrase,
          style: const TextStyle(
            fontSize: 13,
            color: AppColors.primary,
            fontWeight: FontWeight.w500,
          ),
        ),
      )).toList(),
    ).animate().fadeIn(delay: 300.ms).slideY(begin: 0.1);
  }

  String _formatDateTime(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year} at ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  final IconData icon;
  const _SectionHeader(this.title, this.icon);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 20, color: AppColors.textSecondary),
        const SizedBox(width: 8),
        Text(
          title.toUpperCase(),
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.bold,
            letterSpacing: 1.2,
            color: AppColors.textSecondary,
          ),
        ),
      ],
    );
  }
}

class _ResultRow extends StatelessWidget {
  final String label;
  final String value;
  const _ResultRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 13, color: AppColors.textSecondary)),
          Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}
