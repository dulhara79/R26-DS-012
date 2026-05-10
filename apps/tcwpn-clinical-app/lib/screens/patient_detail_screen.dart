import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';
import '../widgets/risk_badge.dart';
import 'assessment_input_screen.dart';

class PatientDetailScreen extends StatelessWidget {
  final Patient patient;

  const PatientDetailScreen({super.key, required this.patient});

  @override
  Widget build(BuildContext context) {
    final updated = context
        .watch<PatientProvider>()
        .patients
        .firstWhere((p) => p.id == patient.id, orElse: () => patient);

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 160,
            pinned: true,
            backgroundColor: AppColors.primary,
            leading: IconButton(
              icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
            flexibleSpace: FlexibleSpaceBar(
              titlePadding: const EdgeInsets.fromLTRB(56, 0, 20, 16),
              title: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    updated.name,
                    style: const TextStyle(
                      fontSize: 17, color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    '${updated.age}y · ${updated.gender} · ${updated.ward}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withOpacity(0.75),
                    ),
                  ),
                ],
              ),
            ),
            actions: [
              Container(
                margin: const EdgeInsets.only(right: 16, top: 8, bottom: 8),
                child: RiskBadge(risk: updated.latestRisk, large: true),
              ),
            ],
          ),

          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([

                // Summary row
                Row(
                  children: [
                    _InfoChip(label: 'ID', value: updated.id),
                    const SizedBox(width: 8),
                    _InfoChip(label: 'Visits', value: '${updated.totalVisits}'),
                    const SizedBox(width: 8),
                    _InfoChip(label: 'Since', value: updated.referralDate),
                  ],
                ),
                const SizedBox(height: 16),

                // Risk trend chart
                if (updated.assessments.length >= 2) ...[
                  _SectionHeader('Risk score over time'),
                  const SizedBox(height: 10),
                  _RiskTrendChart(assessments: updated.assessments),
                  const SizedBox(height: 16),
                ],

                // Latest result card
                if (updated.latestAssessment?.result != null) ...[
                  _SectionHeader('Latest assessment'),
                  const SizedBox(height: 10),
                  _LatestResultCard(
                    assessment: updated.latestAssessment!,
                  ),
                  const SizedBox(height: 16),
                ],

                // Assessment history
                _SectionHeader(
                    'Assessment history (${updated.assessments.length})'),
                const SizedBox(height: 10),
                ...updated.assessments.reversed.map((a) =>
                    _AssessmentHistoryItem(assessment: a)),
                const SizedBox(height: 80),
              ]),
            ),
          ),
        ],
      ),

      // New assessment FAB
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => AssessmentInputScreen(patient: updated),
          ),
        ),
        backgroundColor: AppColors.primary,
        icon: const Icon(Icons.add_rounded, color: Colors.white),
        label: const Text(
          'New assessment',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
        ),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader(this.title);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 3, height: 16,
          decoration: BoxDecoration(
            color: AppColors.primary,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 8),
        Text(title, style: Theme.of(context).textTheme.titleLarge),
      ],
    );
  }
}

class _InfoChip extends StatelessWidget {
  final String label;
  final String value;
  const _InfoChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppColors.border, width: 0.8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label,
                style: const TextStyle(fontSize: 11, color: AppColors.textHint)),
            const SizedBox(height: 2),
            Text(value,
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}

class _RiskTrendChart extends StatelessWidget {
  final List<Assessment> assessments;
  const _RiskTrendChart({required this.assessments});

  @override
  Widget build(BuildContext context) {
    final spots = assessments.asMap().entries.map((e) {
      final score = e.value.result?.riskScore ?? 0.0;
      return FlSpot(e.key.toDouble(), score);
    }).toList();

    return Container(
      height: 160,
      padding: const EdgeInsets.fromLTRB(8, 16, 16, 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border, width: 0.8),
      ),
      child: LineChart(
        LineChartData(
          minY: 0, maxY: 1,
          gridData: FlGridData(
            show: true,
            getDrawingHorizontalLine: (_) => FlLine(
              color: AppColors.border, strokeWidth: 0.8,
            ),
            drawVerticalLine: false,
          ),
          borderData: FlBorderData(show: false),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true, reservedSize: 32,
                getTitlesWidget: (v, _) => Text(
                  v.toStringAsFixed(1),
                  style: const TextStyle(
                      fontSize: 10, color: AppColors.textHint),
                ),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true, reservedSize: 20,
                getTitlesWidget: (v, _) => Text(
                  'V${v.toInt() + 1}',
                  style: const TextStyle(
                      fontSize: 10, color: AppColors.textHint),
                ),
              ),
            ),
            rightTitles:  AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles:    AxisTitles(sideTitles: SideTitles(showTitles: false)),
          ),
          extraLinesData: ExtraLinesData(
            horizontalLines: [
              HorizontalLine(
                y: 0.4036,
                color: AppColors.textSecondary.withOpacity(0.5),
                strokeWidth: 1,
                dashArray: [4, 4],
                label: HorizontalLineLabel(
                  show: true,
                  alignment: Alignment.topRight,
                  style: const TextStyle(
                      fontSize: 9, color: AppColors.textSecondary),
                  labelResolver: (_) => 'threshold',
                ),
              ),
            ],
          ),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              color: AppColors.primary,
              barWidth: 2.5,
              dotData: FlDotData(
                getDotPainter: (spot, _, __, ___) => FlDotCirclePainter(
                  radius: 4,
                  color: AppColors.surface,
                  strokeWidth: 2,
                  strokeColor: AppColors.primary,
                ),
              ),
              belowBarData: BarAreaData(
                show: true,
                color: AppColors.primarySurface.withOpacity(0.6),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LatestResultCard extends StatelessWidget {
  final Assessment assessment;
  const _LatestResultCard({required this.assessment});

  @override
  Widget build(BuildContext context) {
    final r = assessment.result!;
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
          Row(
            children: [
              Expanded(
                child: Text(r.prediction,
                    style: Theme.of(context).textTheme.headlineSmall),
              ),
              RiskBadge(risk: r.riskLevel, large: true),
            ],
          ),
          const SizedBox(height: 14),
          _ResultRow('Risk score', r.riskScore.toStringAsFixed(4)),
          _ResultRow('Confidence', '${(r.confidence * 100).toStringAsFixed(1)}%'),
          _ResultRow('Latency', '${r.latencyMs} ms'),
          if (r.temporalContext.isNotEmpty)
            _ResultRow('Context', r.temporalContext),
          const SizedBox(height: 12),
          Text('Key phrases',
              style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          Wrap(
            spacing: 6, runSpacing: 6,
            children: r.keyPhrases.map((p) => Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: AppColors.riskModerateBg,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                    color: AppColors.riskModerate.withOpacity(0.3)),
              ),
              child: Text(p,
                  style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.riskModerate,
                      fontWeight: FontWeight.w500)),
            )).toList(),
          ),
        ],
      ),
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
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(
                  fontSize: 13, color: AppColors.textSecondary)),
          Text(value,
              style: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _AssessmentHistoryItem extends StatelessWidget {
  final Assessment assessment;
  const _AssessmentHistoryItem({required this.assessment});

  @override
  Widget build(BuildContext context) {
    final r = assessment.result;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border, width: 0.8),
      ),
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          Container(
            width: 40, height: 40,
            decoration: BoxDecoration(
              color: AppColors.primarySurface,
              borderRadius: BorderRadius.circular(10),
            ),
            alignment: Alignment.center,
            child: Text(
              r != null ? r.riskScore.toStringAsFixed(2) : '—',
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: AppColors.primary,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(assessment.noteType,
                    style: Theme.of(context).textTheme.titleSmall),
                const SizedBox(height: 2),
                Text(
                  _formatDateTime(assessment.timestamp),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          if (r != null) RiskBadge(risk: r.riskLevel),
        ],
      ),
    );
  }

  String _formatDateTime(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inDays == 0) return 'Today ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
    if (diff.inDays == 1) return 'Yesterday';
    return '${dt.day}/${dt.month}/${dt.year}';
  }
}
