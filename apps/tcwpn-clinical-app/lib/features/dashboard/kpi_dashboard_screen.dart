// lib/features/dashboard/kpi_dashboard_screen.dart
//
// Clinician-facing dashboard for the investor demo branch.
//
// IMPORTANT: this screen deliberately uses CLINICAL-NOTE assessment results,
// not fusion/composite results. The demo fusion path is currently being repaired,
// and presenting a single-modality clinical score as a multimodal composite would
// be misleading. Every label on this page therefore says clinical risk.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/local/stores.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import '../shell.dart' show PatientRow, openChart;

enum Trajectory { improving, worsening, steady, insufficient }

extension TrajectoryX on Trajectory {
  String get label => switch (this) {
        Trajectory.improving => 'Improving',
        Trajectory.worsening => 'Worsening',
        Trajectory.steady => 'Stable',
        Trajectory.insufficient => 'One assessment',
      };

  IconData get icon => switch (this) {
        Trajectory.improving => Icons.trending_down_rounded,
        Trajectory.worsening => Icons.trending_up_rounded,
        Trajectory.steady => Icons.trending_flat_rounded,
        Trajectory.insufficient => Icons.remove_rounded,
      };

  Color get tone => switch (this) {
        Trajectory.improving => Ds.green,
        Trajectory.worsening => Ds.red,
        Trajectory.steady => Ds.brand,
        Trajectory.insufficient => Ds.grey,
      };
}

const double kTrajectoryDeadband = 0.02;

class ClinicalDashboardAssessment {
  final double score;
  final DateTime at;
  const ClinicalDashboardAssessment({required this.score, required this.at});
  AlertBand get band => AlertBandX.fromScore(score);
}

class ClinicalDashboardPatient {
  final String patientMrn;
  final List<ClinicalDashboardAssessment> assessments;

  const ClinicalDashboardPatient({
    required this.patientMrn,
    this.assessments = const [],
  });

  factory ClinicalDashboardPatient.fromNotes(
    String patientMrn,
    List<ClinicalNote> notes,
  ) {
    final usable = notes
        .where((n) => n.isAnalysed && !n.resultIsStale)
        .map((n) => ClinicalDashboardAssessment(
              score: n.result!.calibratedProbability.clamp(0.0, 1.0),
              at: n.analysedAt ?? n.recordedAt,
            ))
        .toList()
      ..sort((a, b) => b.at.compareTo(a.at));

    return ClinicalDashboardPatient(
      patientMrn: patientMrn,
      assessments: usable,
    );
  }

  bool get hasAssessment => assessments.isNotEmpty;
  double? get latestScore => hasAssessment ? assessments.first.score : null;
  AlertBand? get latestBand =>
      latestScore == null ? null : AlertBandX.fromScore(latestScore!);
  DateTime? get latestAt => hasAssessment ? assessments.first.at : null;

  Trajectory? get trajectory {
    if (!hasAssessment) return null;
    if (assessments.length < 2) return Trajectory.insufficient;
    final delta = assessments[0].score - assessments[1].score;
    if (delta > kTrajectoryDeadband) return Trajectory.worsening;
    if (delta < -kTrajectoryDeadband) return Trajectory.improving;
    return Trajectory.steady;
  }
}

class ClinicalDashboardStats {
  final int total;
  final int scoredCount;
  final int awaitingAssessment;
  final int needsReview;
  final double? meanRisk;
  final Map<AlertBand, int> bandCounts;
  final Map<Trajectory, int> trajectoryCounts;
  final Map<String, ClinicalDashboardPatient> patients;

  const ClinicalDashboardStats({
    required this.total,
    required this.scoredCount,
    required this.awaitingAssessment,
    required this.needsReview,
    required this.meanRisk,
    required this.bandCounts,
    required this.trajectoryCounts,
    required this.patients,
  });

  factory ClinicalDashboardStats.fromSummaries({
    required List<String> patientMrns,
    required Map<String, ClinicalDashboardPatient> summaries,
  }) {
    final bands = <AlertBand, int>{};
    final trajectories = <Trajectory, int>{};
    var scored = 0;
    var review = 0;
    var sum = 0.0;

    for (final mrn in patientMrns) {
      final summary =
          summaries[mrn] ?? ClinicalDashboardPatient(patientMrn: mrn);
      final score = summary.latestScore;
      final band = summary.latestBand;
      if (score == null || band == null) continue;
      scored++;
      sum += score;
      bands[band] = (bands[band] ?? 0) + 1;
      if (band == AlertBand.red || band == AlertBand.darkRed) review++;
      final t = summary.trajectory;
      if (t != null) trajectories[t] = (trajectories[t] ?? 0) + 1;
    }

    return ClinicalDashboardStats(
      total: patientMrns.length,
      scoredCount: scored,
      awaitingAssessment: patientMrns.length - scored,
      needsReview: review,
      meanRisk: scored == 0 ? null : sum / scored,
      bandCounts: bands,
      trajectoryCounts: trajectories,
      patients: summaries,
    );
  }

  static Future<ClinicalDashboardStats> load(List<Patient> patients) async {
    final summaries = <String, ClinicalDashboardPatient>{};
    for (final p in patients) {
      final notes = await RecordStore.loadNotes(p.mrn);
      summaries[p.mrn] = ClinicalDashboardPatient.fromNotes(p.mrn, notes);
    }
    return ClinicalDashboardStats.fromSummaries(
      patientMrns: patients.map((p) => p.mrn).toList(),
      summaries: summaries,
    );
  }

  int countFor(AlertBand band) => bandCounts[band] ?? 0;
  int trajectoryCount(Trajectory t) => trajectoryCounts[t] ?? 0;
}

class KpiDashboardScreen extends StatefulWidget {
  const KpiDashboardScreen({super.key});

  @override
  State<KpiDashboardScreen> createState() => _KpiDashboardScreenState();
}

class _KpiDashboardScreenState extends State<KpiDashboardScreen> {
  ClinicalDashboardStats? _stats;
  bool _loading = true;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => _reload(silent: true),
    );
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _reload();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _reload({bool silent = false}) async {
    if (!mounted) return;
    final roster = context.read<RosterController>();
    if (!silent && _stats == null) setState(() => _loading = true);
    final next = await ClinicalDashboardStats.load(roster.patients);
    if (!mounted) return;
    setState(() {
      _stats = next;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();
    final stats = _stats;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dashboard'),
        actions: [
          IconButton(
            tooltip: 'Refresh clinical dashboard',
            onPressed: () => _reload(),
            icon: const Icon(Icons.refresh_rounded),
          ),
        ],
      ),
      body: roster.loading || (_loading && stats == null)
          ? const Center(child: CircularProgressIndicator(color: Ds.brand))
          : roster.patients.isEmpty
              ? const EmptyState(
                  icon: Icons.insights_rounded,
                  title: 'No patients yet',
                  body:
                      'Add a patient and analyse a clinical note to populate the dashboard.',
                )
              : _body(context, roster, stats ?? _empty(roster)),
    );
  }

  ClinicalDashboardStats _empty(RosterController roster) =>
      ClinicalDashboardStats.fromSummaries(
        patientMrns: roster.patients.map((p) => p.mrn).toList(),
        summaries: const {},
      );

  Widget _body(
    BuildContext context,
    RosterController roster,
    ClinicalDashboardStats stats,
  ) {
    return RefreshIndicator(
      onRefresh: () => _reload(),
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s3, Ds.s4, Ds.s10),
        children: [
          const InlineNotice(
            icon: Icons.description_outlined,
            tone: Ds.brand,
            text:
                'This dashboard currently summarises clinical-note assessments. '
                'It does not label these values as fusion or multimodal risk.',
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Clinical overview'),
          _overview(stats),
          const SizedBox(height: Ds.s6),
          const SectionLabel('Severity distribution'),
          _severity(stats),
          const SizedBox(height: Ds.s6),
          const SectionLabel('Direction of travel'),
          _trajectory(stats),
          const SizedBox(height: Ds.s6),
          const SectionLabel('Patients'),
          ...roster.patients.map(
            (p) => _patientCard(context, roster, p, stats),
          ),
          const SizedBox(height: Ds.s4),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }

  Widget _overview(ClinicalDashboardStats s) {
    final mean =
        s.meanRisk == null ? '—' : '${(s.meanRisk! * 100).toStringAsFixed(1)}%';
    return LayoutBuilder(
      builder: (context, c) {
        final width = (c.maxWidth - Ds.s3) / 2;
        return Wrap(
          spacing: Ds.s3,
          runSpacing: Ds.s3,
          children: [
            _tile(width, 'Active patients', '${s.total}',
                Icons.groups_2_outlined, Ds.brand, 'on this device'),
            _tile(width, 'Needs review', '${s.needsReview}',
                Icons.priority_high_rounded, Ds.red,
                'latest note is high / very high'),
            _tile(width, 'Mean clinical risk', mean,
                Icons.monitor_heart_outlined, Ds.brand,
                'over ${s.scoredCount} assessed patients'),
            _tile(width, 'Awaiting assessment', '${s.awaitingAssessment}',
                Icons.pending_actions_outlined, Ds.grey,
                'no current analysed note'),
          ],
        );
      },
    );
  }

  Widget _tile(
    double width,
    String label,
    String value,
    IconData icon,
    Color tone,
    String caption,
  ) {
    return SizedBox(
      width: width,
      child: Panel(
        padding: const EdgeInsets.all(Ds.s4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 19, color: tone),
            const SizedBox(height: Ds.s3),
            Text(value,
                style: AppTheme.data(
                    size: 24, weight: FontWeight.w700, color: Ds.ink)),
            const SizedBox(height: 2),
            Text(label,
                style: const TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: Ds.ink)),
            const SizedBox(height: 2),
            Text(caption,
                style: const TextStyle(fontSize: 10.5, color: Ds.inkFaint)),
          ],
        ),
      ),
    );
  }

  Widget _severity(ClinicalDashboardStats s) {
    if (s.scoredCount == 0) {
      return Panel(
        child: Text(
          '${s.total} active patients are on this device, but none has a current analysed clinical note yet.',
          style: const TextStyle(
              fontSize: 13, color: Ds.inkMuted, height: 1.45),
        ),
      );
    }

    const bands = [
      AlertBand.green,
      AlertBand.amber,
      AlertBand.red,
      AlertBand.darkRed,
    ];
    return Panel(
      child: Column(
        children: [
          for (final b in bands) ...[
            _distributionRow(
              label: b.severityLabel,
              count: s.countFor(b),
              total: s.scoredCount,
              tone: b.fg,
            ),
            if (b != bands.last) const SizedBox(height: Ds.s4),
          ],
          if (s.awaitingAssessment > 0) ...[
            const Divider(color: Ds.hairline, height: Ds.s6),
            Row(
              children: [
                const Icon(Icons.help_outline_rounded,
                    size: 17, color: Ds.grey),
                const SizedBox(width: Ds.s2),
                Text('${s.awaitingAssessment} awaiting assessment',
                    style: const TextStyle(
                        fontSize: 12, color: Ds.inkMuted)),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _distributionRow({
    required String label,
    required int count,
    required int total,
    required Color tone,
  }) {
    final fraction = total == 0 ? 0.0 : count / total;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(label,
                  style: const TextStyle(
                      fontSize: 13, fontWeight: FontWeight.w600)),
            ),
            Text('$count',
                style: AppTheme.data(size: 13, weight: FontWeight.w700)),
          ],
        ),
        const SizedBox(height: Ds.s2),
        ClipRRect(
          borderRadius: BorderRadius.circular(Ds.rPill),
          child: LinearProgressIndicator(
            value: fraction,
            minHeight: 8,
            backgroundColor: Ds.surfaceSunken,
            valueColor: AlwaysStoppedAnimation<Color>(tone),
          ),
        ),
      ],
    );
  }

  Widget _trajectory(ClinicalDashboardStats s) {
    if (s.scoredCount == 0) {
      return const Panel(
        child: Text(
          'Analyse a clinical note to begin tracking direction of travel.',
          style: TextStyle(fontSize: 13, color: Ds.inkMuted),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, c) {
        final width = (c.maxWidth - Ds.s3) / 2;
        return Wrap(
          spacing: Ds.s3,
          runSpacing: Ds.s3,
          children: [
            for (final t in Trajectory.values)
              _tile(
                width,
                t.label,
                '${s.trajectoryCount(t)}',
                t.icon,
                t.tone,
                t == Trajectory.insufficient
                    ? 'needs a second analysed note'
                    : 'latest versus previous note',
              ),
          ],
        );
      },
    );
  }

  Widget _patientCard(
    BuildContext context,
    RosterController roster,
    Patient patient,
    ClinicalDashboardStats stats,
  ) {
    final summary = stats.patients[patient.mrn];
    final score = summary?.latestScore;
    final band = summary?.latestBand;
    final trajectory = summary?.trajectory;

    return Padding(
      padding: const EdgeInsets.only(bottom: Ds.s3),
      child: Panel(
        padding: EdgeInsets.zero,
        child: InkWell(
          borderRadius: BorderRadius.circular(Ds.rMd),
          onTap: () => openChart(context, patient),
          child: Padding(
            padding: const EdgeInsets.all(Ds.s4),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        patient.name,
                        style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            color: Ds.ink),
                      ),
                      const SizedBox(height: 3),
                      Text(patient.mrn,
                          style: AppTheme.data(size: 11, color: Ds.inkFaint)),
                    ],
                  ),
                ),
                if (score == null || band == null)
                  const Text(
                    'Awaiting assessment',
                    style: TextStyle(
                        fontSize: 11.5,
                        color: Ds.grey,
                        fontWeight: FontWeight.w600),
                  )
                else
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        '${(score * 100).toStringAsFixed(1)}%',
                        style: AppTheme.data(
                            size: 16,
                            weight: FontWeight.w700,
                            color: band.fg),
                      ),
                      Text(
                        band.severityLabel,
                        style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: band.fg),
                      ),
                      if (trajectory != null)
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(trajectory.icon,
                                size: 13, color: trajectory.tone),
                            const SizedBox(width: 2),
                            Text(
                              trajectory.label,
                              style: TextStyle(
                                  fontSize: 10.5, color: trajectory.tone),
                            ),
                          ],
                        ),
                    ],
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
