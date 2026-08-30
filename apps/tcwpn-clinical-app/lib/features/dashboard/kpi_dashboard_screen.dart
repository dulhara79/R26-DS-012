// lib/features/dashboard/kpi_dashboard_screen.dart
//
// Caseload KPI dashboard — the sixth destination in the shell.
//
// WHAT THIS SCREEN IS ALLOWED TO DO
// ---------------------------------
// It reads RosterController and renders. It computes NO risk score, invents no
// composite, and adds no backend route. The only network call it makes is the
// one the chart screen already makes — GET /v1/doctor/patients/{id}/timeline —
// looped over the patients already on this device, so "Refresh all" is a
// re-read of the server's own answers and nothing more.
//
// It also deliberately does NOT enrol anybody. A patient with no local
// subject_id is reported as "not enrolled" rather than silently registered,
// because minting backend records as a side effect of opening a dashboard is
// how a roster quietly diverges from the study log.
//
// THE GREY RULE, RESTATED
// -----------------------
// tokens.dart already says it: GREY is the absence of an assessment, not the
// low end of the scale. Every aggregate on this screen honours that:
//
//   • the severity bar chart iterates AlertBandX.scored — four bars, never five;
//   • the mean composite is taken over scored patients only, and always prints
//     its own denominator so the number can never be read as a caseload mean;
//   • "Not scored" is a separate, neutral tile in Ds.grey, outside every chart;
//   • a patient with no current composite gets NO trajectory, even when their
//     history has one. Direction without a current reading is a claim about a
//     patient we cannot presently score.

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../core/config/env.dart';
import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/api_client.dart';
import '../../data/api/gateways.dart';
import '../../data/local/stores.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import '../shell.dart' show PatientRow, openChart;

// ─────────────────────────────────────────────────────────────────────────────
// Trajectory
// ─────────────────────────────────────────────────────────────────────────────

/// Which way a patient's composite has moved between the last two assessments
/// the server actually computed.
enum Trajectory { improving, worsening, steady, insufficient }

extension TrajectoryX on Trajectory {
  String get label => switch (this) {
        Trajectory.improving => 'Improving',
        Trajectory.worsening => 'Worsening',
        Trajectory.steady => 'Steady',
        Trajectory.insufficient => 'One assessment',
      };

  String get caption => switch (this) {
        Trajectory.improving => 'composite fell',
        Trajectory.worsening => 'composite rose',
        Trajectory.steady => 'within deadband',
        Trajectory.insufficient => 'no direction yet',
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

/// Movement smaller than this is reported as Steady.
///
/// A DISPLAY DEADBAND, NOT A CLINICAL THRESHOLD. The composite is a weighted
/// blend that is renormalised whenever the set of reporting modalities changes,
/// so a wearable dropping out for one cycle can shift it by a couple of points
/// with no change in the patient. Calling every such wobble a direction would
/// fill this screen with movement that means nothing. It is not derived from
/// any validation study, and nothing clinical should be hung on it.
const double kTrajectoryDeadband = 0.02;

/// The patient's direction of travel, or null when they have no current
/// composite at all. Null and `insufficient` are different states: null means
/// "we cannot score this patient right now", `insufficient` means "we can, but
/// only once, so there is nothing to compare against".
Trajectory? trajectoryOf(FusionResult? f) {
  if (f == null || !f.hasComposite) return null;

  // The backend returns `trend` oldest-first (main.py reverses the desc query),
  // so the last two entries are the two most recent assessments.
  final scored = f.trend.where((t) => t.composite != null).toList();
  if (scored.length < 2) return Trajectory.insufficient;

  final delta = scored.last.composite! - scored[scored.length - 2].composite!;
  if (delta > kTrajectoryDeadband) return Trajectory.worsening;
  if (delta < -kTrajectoryDeadband) return Trajectory.improving;
  return Trajectory.steady;
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregates
// ─────────────────────────────────────────────────────────────────────────────

/// One pure snapshot of the caseload, computed from the roster in a single
/// pass. Kept as a plain object rather than a pile of getters in build() so
/// every number on screen has exactly one definition that can be pointed at.
class CaseloadStats {
  final int total;

  /// Counts for the FOUR SCORED BANDS ONLY. GREY is never a key here.
  final Map<AlertBand, int> bandCounts;

  /// No composite for any reason.
  final int notScored;

  /// Of [notScored]: never assessed at all (no fusion row on this device).
  final int neverAssessed;

  /// Of [notScored]: a fusion row exists and the gate declined to produce a
  /// composite. Distinct from never-assessed — this one has a server reason.
  final int gateBlocked;

  final int scoredCount;
  final double? meanComposite;
  final int needsReview;

  final Map<Trajectory, int> trajectoryCounts;

  /// Most recent `updated_at` across the roster, and the oldest one, so the
  /// dashboard can state how current it actually is.
  final DateTime? newestComposite;
  final DateTime? oldestComposite;

  const CaseloadStats({
    required this.total,
    required this.bandCounts,
    required this.notScored,
    required this.neverAssessed,
    required this.gateBlocked,
    required this.scoredCount,
    required this.meanComposite,
    required this.needsReview,
    required this.trajectoryCounts,
    required this.newestComposite,
    required this.oldestComposite,
  });

  factory CaseloadStats.from(RosterController roster) {
    final bands = <AlertBand, int>{};
    final traj = <Trajectory, int>{};
    var notScored = 0;
    var neverAssessed = 0;
    var gateBlocked = 0;
    var sum = 0.0;
    var scored = 0;
    var review = 0;
    DateTime? newest;
    DateTime? oldest;

    for (final p in roster.patients) {
      final f = roster.fusionFor(p.mrn);

      if (f == null) {
        notScored++;
        neverAssessed++;
        continue;
      }
      if (!f.hasComposite) {
        notScored++;
        gateBlocked++;
        continue;
      }

      bands[f.band] = (bands[f.band] ?? 0) + 1;
      sum += f.compositeScore!;
      scored++;
      if (f.band == AlertBand.red || f.band == AlertBand.darkRed) review++;

      final t = trajectoryOf(f);
      if (t != null) traj[t] = (traj[t] ?? 0) + 1;

      final at = f.updatedAt;
      if (at != null) {
        if (newest == null || at.isAfter(newest)) newest = at;
        if (oldest == null || at.isBefore(oldest)) oldest = at;
      }
    }

    return CaseloadStats(
      total: roster.patients.length,
      bandCounts: bands,
      notScored: notScored,
      neverAssessed: neverAssessed,
      gateBlocked: gateBlocked,
      scoredCount: scored,
      // Mean over scored patients only. Null rather than 0 when nothing is
      // scored — a caseload with no assessments has no mean, and printing
      // 0.000 would put every unscored patient at the bottom of the scale.
      meanComposite: scored == 0 ? null : sum / scored,
      needsReview: review,
      trajectoryCounts: traj,
      newestComposite: newest,
      oldestComposite: oldest,
    );
  }

  int countFor(AlertBand b) => bandCounts[b] ?? 0;
  int trajectoryCount(Trajectory t) => trajectoryCounts[t] ?? 0;

  /// Patients with a current composite AND at least two assessments, i.e. the
  /// denominator the trajectory tiles are actually a fraction of.
  int get trajectoryDenominator =>
      trajectoryCounts.values.fold<int>(0, (a, b) => a + b);
}

/// One day's worth of assessments across the whole caseload.
class CohortDay {
  final DateTime day;
  final double mean;
  final int n;
  const CohortDay(this.day, this.mean, this.n);
}

/// Mean composite per calendar day, from every patient's `trend`.
///
/// This is a mean OF ASSESSMENTS, not of patients: a patient assessed three
/// times on one day contributes three values that day. The chart labels it as
/// such, and carries the per-day n, because the denominator moves.
List<CohortDay> cohortSeries(RosterController roster, {int days = 14}) {
  final cutoff = DateTime.now().subtract(Duration(days: days - 1));
  final floor = DateTime(cutoff.year, cutoff.month, cutoff.day);
  final buckets = <DateTime, List<double>>{};

  for (final p in roster.patients) {
    final f = roster.fusionFor(p.mrn);
    if (f == null) continue;
    for (final t in f.trend) {
      final at = t.computedAt;
      final c = t.composite;
      if (at == null || c == null) continue;
      final d = DateTime(at.year, at.month, at.day);
      if (d.isBefore(floor)) continue;
      buckets.putIfAbsent(d, () => <double>[]).add(c);
    }
  }

  final keys = buckets.keys.toList()..sort();
  return [
    for (final k in keys)
      CohortDay(
        k,
        buckets[k]!.reduce((a, b) => a + b) / buckets[k]!.length,
        buckets[k]!.length,
      ),
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter
// ─────────────────────────────────────────────────────────────────────────────

/// Exactly one filter is active at a time. Two orthogonal filters (band AND
/// trajectory) would let a clinician build a query whose empty result is
/// ambiguous — no RED-and-improving patients, or no RED patients at all?
class _Filter {
  final AlertBand? band;
  final bool notScored;
  final Trajectory? trajectory;

  const _Filter._({this.band, this.notScored = false, this.trajectory});

  const _Filter.all() : this._();
  const _Filter.byBand(AlertBand b) : this._(band: b);
  const _Filter.unscored() : this._(notScored: true);
  const _Filter.byTrajectory(Trajectory t) : this._(trajectory: t);

  bool get isAll => band == null && !notScored && trajectory == null;

  String get label {
    if (band != null) return band!.protocolName;
    if (notScored) return 'Not scored';
    if (trajectory != null) return trajectory!.label;
    return 'All patients';
  }

  bool accepts(FusionResult? f) {
    final scored = f != null && f.hasComposite;
    if (notScored) return !scored;
    if (band != null) return scored && f.band == band;
    if (trajectory != null) return trajectoryOf(f) == trajectory;
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen
// ─────────────────────────────────────────────────────────────────────────────

class KpiDashboardScreen extends StatefulWidget {
  const KpiDashboardScreen({super.key});

  @override
  State<KpiDashboardScreen> createState() => _KpiDashboardScreenState();
}

class _KpiDashboardScreenState extends State<KpiDashboardScreen> {
  final _backend = CentralBackendGateway();

  _Filter _filter = const _Filter.all();

  bool _refreshing = false;
  int _done = 0;
  int _target = 0;
  List<String> _failures = const [];

  @override
  void dispose() {
    _backend.dispose();
    super.dispose();
  }

  // ── Refresh all ───────────────────────────────────────────────────────────
  //
  // One existing endpoint, once per patient. No new route, no enrolment, no
  // fusion re-run. Failures are collected and shown rather than swallowed: a
  // dashboard that silently keeps displaying yesterday's composite after a
  // failed refresh is worse than one that says which patients it could not
  // reach.

  Future<void> _refreshAll(RosterController roster) async {
    if (_refreshing) return;

    if (!Env.hasBackend) {
      setState(() => _failures = const [
            'No backend address is compiled into this build, so nothing could '
                'be refreshed. Pass --dart-define=BACKEND_BASE=... when you run.'
          ]);
      return;
    }

    final patients = roster.patients;
    setState(() {
      _refreshing = true;
      _done = 0;
      _target = patients.length;
      _failures = const [];
    });

    final failures = <String>[];

    for (final p in patients) {
      try {
        final subjectId = await RecordStore.subjectId(p.mrn);
        if (subjectId == null || subjectId.isEmpty) {
          // Deliberately NOT enrolled here. See the header note.
          failures.add('${p.mrn} — not enrolled on the backend yet');
        } else {
          final result =
              await _backend.timeline(subjectId: subjectId, mrn: p.mrn);
          if (result == null) {
            failures.add('${p.mrn} — the backend has no record for this '
                'subject id');
          } else {
            await roster.refreshFusion(p.mrn, result);
          }
        }
      } on ApiException catch (e) {
        failures.add('${p.mrn} — ${e.message}');
      }

      if (!mounted) return;
      setState(() => _done++);
    }

    if (!mounted) return;
    setState(() {
      _refreshing = false;
      _failures = failures;
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();
    final stats = CaseloadStats.from(roster);
    final series = cohortSeries(roster);

    final matches = roster.patients
        .where((p) => _filter.accepts(roster.fusionFor(p.mrn)))
        .toList()
      ..sort((a, b) => _severityRank(roster.fusionFor(b.mrn))
          .compareTo(_severityRank(roster.fusionFor(a.mrn))));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dashboard'),
        actions: [
          IconButton(
            tooltip: 'Refresh every patient from the backend',
            icon: _refreshing
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Ds.brand),
                  )
                : const Icon(Icons.sync_rounded),
            onPressed: _refreshing ? null : () => _refreshAll(roster),
          ),
        ],
      ),
      body: roster.loading
          ? const Center(child: CircularProgressIndicator(color: Ds.brand))
          : stats.total == 0
              ? const EmptyState(
                  icon: Icons.insights_rounded,
                  title: 'No patients yet',
                  body: 'The dashboard summarises the patients on this device. '
                      'Add a patient from the Patients tab and the caseload '
                      'figures will appear here.',
                )
              : ListView(
                  padding:
                      const EdgeInsets.fromLTRB(Ds.s4, Ds.s3, Ds.s4, Ds.s10),
                  children: [
                    if (_refreshing) _progress(),
                    if (_failures.isNotEmpty) _failureNotice(),
                    _currency(stats),
                    const SizedBox(height: Ds.s5),
                    const SectionLabel('Caseload'),
                    _kpiGrid(stats),
                    const SizedBox(height: Ds.s6),
                    const SectionLabel('Severity distribution'),
                    _bandChartPanel(stats),
                    const SizedBox(height: Ds.s6),
                    const SectionLabel('Direction of travel'),
                    _trajectoryPanel(stats),
                    const SizedBox(height: Ds.s6),
                    const SectionLabel('Cohort composite, last 14 days'),
                    _cohortPanel(series),
                    const SizedBox(height: Ds.s6),
                    SectionLabel(_filter.isAll
                        ? 'All patients'
                        : 'Filtered · ${_filter.label}'),
                    _filterRow(stats),
                    const SizedBox(height: Ds.s3),
                    if (matches.isEmpty)
                      Panel(
                        child: Text(
                          'No patient matches ${_filter.label}.',
                          style: const TextStyle(
                              fontSize: 13, color: Ds.inkMuted),
                        ),
                      )
                    else
                      ...matches.map((p) => Padding(
                            padding: const EdgeInsets.only(bottom: Ds.s3),
                            child: PatientRow(
                              patient: p,
                              fusion: roster.fusionFor(p.mrn),
                              onTap: () => openChart(context, p),
                            ),
                          )),
                    const SizedBox(height: Ds.s4),
                    const DecisionSupportNotice(),
                  ],
                ),
    );
  }

  /// Sort key for the filtered list. Unscored patients sort to the bottom
  /// rather than the top — they are not the least severe, they are simply not
  /// on the scale, and burying them under RED is the right reading order.
  int _severityRank(FusionResult? f) {
    if (f == null || !f.hasComposite) return -1;
    return switch (f.band) {
      AlertBand.darkRed => 4,
      AlertBand.red => 3,
      AlertBand.amber => 2,
      AlertBand.green => 1,
      AlertBand.grey => -1,
    };
  }

  // ── Pieces ────────────────────────────────────────────────────────────────

  Widget _progress() => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: Panel(
          padding: const EdgeInsets.all(Ds.s4),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Refreshing $_done of $_target',
                  style: AppTheme.data(size: 12, weight: FontWeight.w600)),
              const SizedBox(height: Ds.s2),
              ClipRRect(
                borderRadius: BorderRadius.circular(Ds.rPill),
                child: LinearProgressIndicator(
                  value: _target == 0 ? null : _done / _target,
                  minHeight: 5,
                  backgroundColor: Ds.surfaceSunken,
                  valueColor: const AlwaysStoppedAnimation<Color>(Ds.brand),
                ),
              ),
            ],
          ),
        ),
      );

  Widget _failureNotice() => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: InlineNotice(
          icon: Icons.error_outline_rounded,
          tone: Ds.amber,
          text: 'Some patients could not be refreshed, so their figures below '
              'are the last known values:\n\u2022 ${_failures.join('\n\u2022 ')}',
        ),
      );

  /// How current this screen is. A dashboard that does not say when its numbers
  /// were computed invites them to be read as live.
  Widget _currency(CaseloadStats s) {
    final newest = s.newestComposite;
    final oldest = s.oldestComposite;
    final stale = oldest != null &&
        DateTime.now().difference(oldest) > Env.stalenessThreshold;

    return Panel(
      padding: const EdgeInsets.all(Ds.s4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(stale ? Icons.history_rounded : Icons.schedule_rounded,
              size: 16, color: stale ? Ds.amber : Ds.inkFaint),
          const SizedBox(width: Ds.s3),
          Expanded(
            child: Text(
              newest == null
                  ? 'No composite has been computed for any patient on this '
                      'device yet.'
                  : 'Composites shown were computed by the backend between '
                      '${DateFormat('d MMM, HH:mm').format(oldest!)} and '
                      '${DateFormat('d MMM, HH:mm').format(newest)}. '
                      'Use Refresh to re-read them.',
              style: const TextStyle(
                  fontSize: 12, color: Ds.inkMuted, height: 1.45),
            ),
          ),
        ],
      ),
    );
  }

  Widget _kpiGrid(CaseloadStats s) => LayoutBuilder(
        builder: (context, c) {
          final columns = c.maxWidth > 720 ? 4 : 2;
          final width =
              (c.maxWidth - (Ds.s3 * (columns - 1))) / columns;

          Widget tile({
            required String label,
            required String value,
            required IconData icon,
            String? caption,
            Color accent = Ds.brand,
            _Filter? filter,
          }) =>
              SizedBox(
                width: width,
                child: _KpiTile(
                  label: label,
                  value: value,
                  caption: caption,
                  icon: icon,
                  accent: accent,
                  selected: filter != null &&
                      filter.label == _filter.label &&
                      !_filter.isAll,
                  onTap: filter == null
                      ? null
                      : () => setState(() => _filter =
                          _filter.label == filter.label
                              ? const _Filter.all()
                              : filter),
                ),
              );

          return Wrap(
            spacing: Ds.s3,
            runSpacing: Ds.s3,
            children: [
              tile(
                label: 'Active patients',
                value: '${s.total}',
                icon: Icons.groups_2_outlined,
                caption: 'on this device',
                filter: const _Filter.all(),
              ),
              tile(
                label: 'Needs review',
                value: '${s.needsReview}',
                icon: Icons.priority_high_rounded,
                accent: Ds.red,
                caption: 'RED or DARK RED',
              ),
              tile(
                label: 'Mean composite',
                value: s.meanComposite == null
                    ? '\u2014'
                    : s.meanComposite!.toStringAsFixed(3),
                icon: Icons.functions_rounded,
                // The denominator travels with the number, always.
                caption: 'over ${s.scoredCount} of ${s.total} scored',
              ),
              // Neutral by construction. Ds.grey, no band colour, no position
              // on the severity axis, and its own caption explaining that this
              // is missing evidence.
              tile(
                label: 'Not scored',
                value: '${s.notScored}',
                icon: Icons.help_outline_rounded,
                accent: Ds.grey,
                caption: s.notScored == 0
                    ? 'every patient has a composite'
                    : '${s.neverAssessed} never assessed \u00b7 '
                        '${s.gateBlocked} gate-blocked',
                filter: const _Filter.unscored(),
              ),
            ],
          );
        },
      );

  // ── Severity distribution ─────────────────────────────────────────────────

  Widget _bandChartPanel(CaseloadStats s) {
    final counts = [for (final b in AlertBandX.scored) s.countFor(b)];
    final maxCount = counts.fold<int>(0, (a, b) => a > b ? a : b);

    if (s.scoredCount == 0) {
      return Panel(
        child: Row(
          children: [
            const Icon(Icons.bar_chart_rounded, size: 18, color: Ds.grey),
            const SizedBox(width: Ds.s3),
            Expanded(
              child: Text(
                'No patient currently has a composite, so there is no '
                'distribution to draw. ${s.notScored} '
                '${s.notScored == 1 ? 'patient is' : 'patients are'} awaiting '
                'an assessment.',
                style: const TextStyle(
                    fontSize: 13, color: Ds.inkMuted, height: 1.45),
              ),
            ),
          ],
        ),
      );
    }

    final maxY = (maxCount * 1.3).ceilToDouble().clamp(1.0, 100000.0).toDouble();

    // Patients are counted in whole people, so the y-axis must step in whole
    // numbers. Left to itself fl_chart picks a "nice" fractional interval and
    // the axis reads 0 / 0.5 / 1.0 patients.
    final yInterval = (maxY / 4).ceilToDouble().clamp(1.0, 100000.0).toDouble();

    return Panel(
      padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s5, Ds.s4, Ds.s4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('SCORED PATIENTS BY BAND', style: AppTheme.eyebrow),
          const SizedBox(height: Ds.s4),
          SizedBox(
            height: 190,
            child: BarChart(
              BarChartData(
                alignment: BarChartAlignment.spaceAround,
                maxY: maxY,
                minY: 0,
                barTouchData: BarTouchData(
                  enabled: true,
                  // Built-in tooltips are off: every count is already printed
                  // under its own bar, so a tooltip would only repeat it. The
                  // touch handler is kept for tap-to-filter.
                  handleBuiltInTouches: false,
                  touchCallback: (event, response) {
                    // FlTapUpEvent specifically, NOT
                    // `isInterestedForInteractions` — that getter returns true
                    // for FlPointerHoverEvent, so on Chrome simply moving the
                    // mouse across the chart would toggle the caseload filter.
                    if (event is! FlTapUpEvent) return;
                    final spot = response?.spot;
                    if (spot == null) return;
                    final band = AlertBandX.scored[spot.touchedBarGroupIndex];
                    setState(() => _filter = _filter.band == band
                        ? const _Filter.all()
                        : _Filter.byBand(band));
                  },
                ),
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: yInterval,
                  getDrawingHorizontalLine: (_) =>
                      const FlLine(color: Ds.hairline, strokeWidth: 1),
                ),
                borderData: FlBorderData(show: false),
                titlesData: FlTitlesData(
                  show: true,
                  topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 30,
                      interval: yInterval,
                      getTitlesWidget: (value, meta) {
                        if (value != value.roundToDouble()) {
                          return const SizedBox.shrink();
                        }
                        return Text(
                          '${value.toInt()}',
                          style: AppTheme.data(size: 10, color: Ds.inkFaint),
                        );
                      },
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 44,
                      getTitlesWidget: (value, meta) {
                        final i = value.toInt();
                        if (i < 0 || i >= AlertBandX.scored.length) {
                          return const SizedBox.shrink();
                        }
                        final b = AlertBandX.scored[i];
                        final selected = _filter.band == b;
                        return Padding(
                          padding: const EdgeInsets.only(top: Ds.s2),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                b.protocolName,
                                style: TextStyle(
                                  fontSize: 9.5,
                                  fontWeight: selected
                                      ? FontWeight.w700
                                      : FontWeight.w600,
                                  color: b.fg,
                                  letterSpacing: 0.2,
                                ),
                              ),
                              Text(
                                '${s.countFor(b)}',
                                style: AppTheme.data(
                                    size: 11,
                                    weight: FontWeight.w600,
                                    color: Ds.ink),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                ),
                barGroups: [
                  for (var i = 0; i < AlertBandX.scored.length; i++)
                    BarChartGroupData(
                      x: i,
                      barRods: [
                        BarChartRodData(
                          toY: s.countFor(AlertBandX.scored[i]).toDouble(),
                          // Band colour, straight from tokens.dart. Dimmed,
                          // never recoloured, when a different band is picked.
                          color: _filter.band == null ||
                                  _filter.band == AlertBandX.scored[i]
                              ? AlertBandX.scored[i].fg
                              : AlertBandX.scored[i].bg,
                          width: 28,
                          borderRadius: const BorderRadius.only(
                            topLeft: Radius.circular(Ds.rSm),
                            topRight: Radius.circular(Ds.rSm),
                          ),
                        ),
                      ],
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: Ds.s3),
          const Divider(color: Ds.hairline, height: Ds.s5),
          // The unscored count lives HERE — beneath the axis, in grey, outside
          // the chart entirely. It is not a fifth bar and never will be.
          Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                    color: Ds.grey, shape: BoxShape.circle),
              ),
              const SizedBox(width: Ds.s2),
              Expanded(
                child: Text(
                  '${s.notScored} not scored \u2014 shown off the scale '
                  'because a blocked assessment is missing evidence, not a '
                  'low score.',
                  style: const TextStyle(
                      fontSize: 11.5, color: Ds.inkFaint, height: 1.4),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ── Trajectory ────────────────────────────────────────────────────────────

  Widget _trajectoryPanel(CaseloadStats s) {
    if (s.trajectoryDenominator == 0) {
      return const Panel(
        child: Text(
          'Direction of travel needs at least one patient with a current '
          'composite. Nothing on this caseload qualifies yet.',
          style: TextStyle(fontSize: 13, color: Ds.inkMuted, height: 1.45),
        ),
      );
    }

    return Column(
      children: [
        LayoutBuilder(
          builder: (context, c) {
            final columns = c.maxWidth > 720 ? 4 : 2;
            final width = (c.maxWidth - (Ds.s3 * (columns - 1))) / columns;
            return Wrap(
              spacing: Ds.s3,
              runSpacing: Ds.s3,
              children: [
                for (final t in Trajectory.values)
                  SizedBox(
                    width: width,
                    child: _KpiTile(
                      label: t.label,
                      value: '${s.trajectoryCount(t)}',
                      caption: t.caption,
                      icon: t.icon,
                      accent: t.tone,
                      selected: _filter.trajectory == t,
                      onTap: () => setState(() => _filter =
                          _filter.trajectory == t
                              ? const _Filter.all()
                              : _Filter.byTrajectory(t)),
                    ),
                  ),
              ],
            );
          },
        ),
        const SizedBox(height: Ds.s3),
        InlineNotice(
          icon: Icons.info_outline_rounded,
          tone: Ds.inkMuted,
          text: 'Direction compares the last two assessments the backend '
              'computed, over ${s.trajectoryDenominator} of ${s.total} '
              'patients \u2014 only those with a current composite. Movement '
              'below ${kTrajectoryDeadband.toStringAsFixed(2)} is reported as '
              'Steady, a display deadband to suppress renormalisation noise, '
              'not a clinical threshold.',
        ),
      ],
    );
  }

  // ── Cohort line ───────────────────────────────────────────────────────────

  Widget _cohortPanel(List<CohortDay> series) {
    if (series.length < 2) {
      return Panel(
        child: Text(
          series.isEmpty
              ? 'No assessments were computed in the last 14 days, so there is '
                  'no cohort line to draw.'
              : 'Only one day in the last 14 has an assessment. A trend line '
                  'needs at least two.',
          style: const TextStyle(
              fontSize: 13, color: Ds.inkMuted, height: 1.45),
        ),
      );
    }

    final first = series.first.day;
    final spanDays =
        series.last.day.difference(first).inDays.toDouble().clamp(1.0, 400.0).toDouble();
    final totalN = series.fold<int>(0, (a, d) => a + d.n);

    return Panel(
      padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s5, Ds.s5, Ds.s4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('MEAN COMPOSITE PER DAY', style: AppTheme.eyebrow),
          const SizedBox(height: 2),
          Text(
            '$totalN assessments across ${series.length} days. This averages '
            'assessments, not patients \u2014 a patient assessed twice in a '
            'day counts twice.',
            style: const TextStyle(
                fontSize: 11.5, color: Ds.inkFaint, height: 1.4),
          ),
          const SizedBox(height: Ds.s4),
          SizedBox(
            height: 180,
            child: LineChart(
              LineChartData(
                minX: 0,
                maxX: spanDays,
                // The composite is defined on 0..1, so the axis is fixed to
                // 0..1. Auto-scaling to the data would make a flat caseload
                // look volatile.
                minY: 0,
                maxY: 1,
                lineTouchData: const LineTouchData(enabled: false),
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: 0.25,
                  getDrawingHorizontalLine: (_) =>
                      const FlLine(color: Ds.hairline, strokeWidth: 1),
                ),
                borderData: FlBorderData(show: false),
                titlesData: FlTitlesData(
                  show: true,
                  topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 34,
                      interval: 0.25,
                      getTitlesWidget: (value, meta) => Text(
                        value.toStringAsFixed(2),
                        style: AppTheme.data(size: 10, color: Ds.inkFaint),
                      ),
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 28,
                      interval: (spanDays / 3).clamp(1.0, 400.0).toDouble(),
                      getTitlesWidget: (value, meta) => Padding(
                        padding: const EdgeInsets.only(top: Ds.s2),
                        child: Text(
                          DateFormat('d MMM').format(
                              first.add(Duration(days: value.round()))),
                          style: AppTheme.data(size: 9.5, color: Ds.inkFaint),
                        ),
                      ),
                    ),
                  ),
                ),
                lineBarsData: [
                  LineChartBarData(
                    isCurved: false,
                    color: Ds.brand,
                    barWidth: 2,
                    dotData: const FlDotData(show: true),
                    belowBarData: BarAreaData(show: true, color: Ds.brandSoft),
                    spots: [
                      for (final d in series)
                        FlSpot(
                          d.day.difference(first).inDays.toDouble(),
                          d.mean,
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Filter chips ──────────────────────────────────────────────────────────

  Widget _filterRow(CaseloadStats s) => SizedBox(
        height: 34,
        child: ListView(
          scrollDirection: Axis.horizontal,
          children: [
            _chip('All', s.total, _filter.isAll, null,
                () => setState(() => _filter = const _Filter.all())),
            for (final b in AlertBandX.scored.reversed)
              _chip(b.protocolName, s.countFor(b), _filter.band == b, b.fg,
                  () {
                setState(() => _filter = _filter.band == b
                    ? const _Filter.all()
                    : _Filter.byBand(b));
              }),
            // Separated from the band chips by a rule, so the eye reads it as a
            // different kind of thing rather than the next step down the scale.
            const VerticalDivider(
                width: Ds.s4, indent: 8, endIndent: 8, color: Ds.hairline),
            _chip('Not scored', s.notScored, _filter.notScored, Ds.grey, () {
              setState(() => _filter = _filter.notScored
                  ? const _Filter.all()
                  : const _Filter.unscored());
            }),
          ],
        ),
      );

  Widget _chip(String label, int count, bool active, Color? tone,
          VoidCallback onTap) =>
      Padding(
        padding: const EdgeInsets.only(right: Ds.s2),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: onTap,
            borderRadius: BorderRadius.circular(Ds.rPill),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: Ds.s3),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: active
                    ? (tone ?? Ds.brand).withValues(alpha: 0.10)
                    : Ds.surface,
                borderRadius: BorderRadius.circular(Ds.rPill),
                border: Border.all(
                  color: active ? (tone ?? Ds.brand) : Ds.hairline,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (tone != null) ...[
                    Container(
                      width: 7,
                      height: 7,
                      decoration:
                          BoxDecoration(color: tone, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 6),
                  ],
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                      color: active ? (tone ?? Ds.brand) : Ds.inkMuted,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text('$count',
                      style: AppTheme.data(
                          size: 11,
                          weight: FontWeight.w600,
                          color: active ? (tone ?? Ds.brand) : Ds.inkFaint)),
                ],
              ),
            ),
          ),
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile
// ─────────────────────────────────────────────────────────────────────────────

/// A tappable KPI box.
///
/// Not `MetricTile` from components.dart: that one has no tap target, and the
/// whole point of these boxes is that a count is also a filter. It is built on
/// the same `Panel` so it stays visually identical to the rest of the product.
class _KpiTile extends StatelessWidget {
  final String label;
  final String value;
  final String? caption;
  final IconData icon;
  final Color accent;
  final bool selected;
  final VoidCallback? onTap;

  const _KpiTile({
    required this.label,
    required this.value,
    required this.icon,
    this.caption,
    this.accent = Ds.brand,
    this.selected = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) => Panel(
        onTap: onTap,
        padding: const EdgeInsets.all(Ds.s4),
        borderColor: selected ? accent : Ds.hairline,
        background: selected ? accent.withValues(alpha: 0.05) : Ds.surface,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Icon(icon, size: 17, color: accent),
                const Spacer(),
                if (onTap != null)
                  Icon(
                    selected
                        ? Icons.filter_alt_rounded
                        : Icons.filter_alt_outlined,
                    size: 13,
                    color: selected ? accent : Ds.inkFaint,
                  ),
              ],
            ),
            const SizedBox(height: Ds.s3),
            Text(value,
                style: AppTheme.data(
                    size: 24, weight: FontWeight.w600, color: Ds.ink)),
            const SizedBox(height: 2),
            Text(label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w600, color: Ds.ink)),
            if (caption != null)
              Text(caption!,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      fontSize: 10.5, color: Ds.inkFaint, height: 1.3)),
          ],
        ),
      );
}
