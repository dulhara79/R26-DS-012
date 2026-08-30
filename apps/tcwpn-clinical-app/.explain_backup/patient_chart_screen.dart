// lib/features/chart/patient_chart_screen.dart
//
// One patient, one screen. The four research components appear as sections of a
// single chart, never as separate apps.
//
//   Risk           — the fused composite and its four contributions
//   Clinical notes — TC-WPN analyses (Component 4)
//   Intervention   — GAD-7, tiering, coping history (Component 3)
//
// The ChartController is created here and holds this patient's MRN final for its
// entire lifetime, which is what guarantees no other patient's record can be
// read or written through it.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import '../fusion/fusion_detail_screen.dart';
import '../tcwpn/note_analysis_screen.dart';
import '../tcwpn/tcwpn_result_screen.dart';
import '../tcwpn/support_set_screen.dart';

class PatientChartScreen extends StatelessWidget {
  final Patient patient;
  const PatientChartScreen({super.key, required this.patient});

  @override
  Widget build(BuildContext context) => ChangeNotifierProvider(
        create: (_) => ChartController(
          patient: patient,
          roster: context.read<RosterController>(),
        )..load(),
        child: const _ChartBody(),
      );
}

class _ChartBody extends StatefulWidget {
  const _ChartBody();
  @override
  State<_ChartBody> createState() => _ChartBodyState();
}

class _ChartBodyState extends State<_ChartBody> {
  int _section = 0;

  @override
  Widget build(BuildContext context) {
    final chart = context.watch<ChartController>();
    final p = chart.patient;

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(p.name, style: AppTheme.display(size: 16.5)),
            Text('${p.mrn} · ${p.age}y · ${p.gender}',
                style: AppTheme.data(size: 10.5, color: Ds.inkFaint)),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Support set for this patient',
            icon: const Icon(Icons.dataset_outlined),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => ChangeNotifierProvider.value(
                  value: chart,
                  child: const SupportSetScreen(scope: SupportScope.patient),
                ),
              ),
            ),
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(58),
          child: Container(
            color: Ds.surface,
            padding: const EdgeInsets.fromLTRB(Ds.s4, 0, Ds.s4, Ds.s3),
            child: _SegmentedControl(
              labels: const ['Risk', 'Clinical notes', 'Intervention'],
              index: _section,
              onChanged: (i) => setState(() => _section = i),
            ),
          ),
        ),
      ),
      floatingActionButton: _section == 1
          ? FloatingActionButton.extended(
              backgroundColor: Ds.brand,
              foregroundColor: Colors.white,
              onPressed: () => _newNote(context, chart),
              icon: const Icon(Icons.add_rounded, size: 20),
              label: const Text('Analyse note'),
            )
          : null,
      body: chart.status == ChartStatus.idle
          ? const Center(child: CircularProgressIndicator(color: Ds.brand))
          : IndexedStack(
              index: _section,
              children: [
                _RiskSection(chart: chart),
                _NotesSection(chart: chart),
                const _InterventionSection(),
              ],
            ),
    );
  }

  Future<void> _newNote(BuildContext context, ChartController chart) =>
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ChangeNotifierProvider.value(
            value: chart,
            child: const NoteAnalysisScreen(),
          ),
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────

class _SegmentedControl extends StatelessWidget {
  final List<String> labels;
  final int index;
  final ValueChanged<int> onChanged;

  const _SegmentedControl({
    required this.labels,
    required this.index,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(3),
        decoration: BoxDecoration(
          color: Ds.surfaceSunken,
          borderRadius: BorderRadius.circular(Ds.rMd),
        ),
        child: Row(
          children: [
            for (var i = 0; i < labels.length; i++)
              Expanded(
                child: GestureDetector(
                  onTap: () => onChanged(i),
                  behavior: HitTestBehavior.opaque,
                  child: AnimatedContainer(
                    duration: Ds.fast,
                    curve: Curves.easeOut,
                    padding: const EdgeInsets.symmetric(vertical: 9),
                    decoration: BoxDecoration(
                      color: index == i ? Ds.surface : Colors.transparent,
                      borderRadius: BorderRadius.circular(Ds.rSm),
                      border: Border.all(
                        color: index == i ? Ds.hairline : Colors.transparent,
                      ),
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      labels[i],
                      style: TextStyle(
                        fontSize: 12.5,
                        fontWeight:
                            index == i ? FontWeight.w600 : FontWeight.w500,
                        color: index == i ? Ds.ink : Ds.inkMuted,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Risk — the fusion view
// ─────────────────────────────────────────────────────────────────────────────

class _RiskSection extends StatelessWidget {
  final ChartController chart;
  const _RiskSection({required this.chart});

  @override
  Widget build(BuildContext context) {
    final f = chart.fusion;

    return RefreshIndicator(
      color: Ds.brand,
      // Re-runs fusion server-side, then re-reads. Deliberately a re-RUN rather
      // than a re-read: the common reason a clinician pulls to refresh is that
      // a wearable reading arrived after the last composite was computed, and a
      // plain re-read would return the same stale row.
      onRefresh: () => chart.rerunFusion(),
      child: ListView(
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          if (f == null)
            const Panel(
              child: Text(
                'No assessment yet. The composite needs at least two usable '
                'modalities, and at least one of them time-varying — so a '
                'clinical note alone will not produce one until the patient app '
                'has also sent a wearable or intake reading.',
                style: TextStyle(fontSize: 13, color: Ds.inkMuted, height: 1.5),
              ),
            )
          else ...[
            Panel(
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => FusionDetailScreen(
                    fusion: f,
                    patient: chart.patient,
                    latestNote: chart.latestAnalysedNote,
                  ),
                ),
              ),
              padding: const EdgeInsets.all(Ds.s5),
              child: Column(
                children: [
                  FusionBar(
                    composite: f.compositeScore,
                    band: f.band,
                    renormalised: f.renormalised,
                    blockedReason: f.reason,
                    segments: [
                      for (final c in f.contributions)
                        FusionSegment(
                          key: c.key,
                          label: c.label,
                          contribution: c.contribution,
                          weight: c.weight,
                          score: c.score,
                          excluded: c.excluded,
                          unavailableReason: c.note,
                          color: FusionBar.palette[c.key] ?? Ds.brand,
                        ),
                    ],
                  ),
                  const SizedBox(height: Ds.s4),
                  const Divider(),
                  const SizedBox(height: Ds.s3),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          f.band.guidance,
                          style: const TextStyle(
                              fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
                        ),
                      ),
                      const SizedBox(width: Ds.s3),
                      const Icon(Icons.chevron_right_rounded,
                          size: 20, color: Ds.inkFaint),
                    ],
                  ),
                ],
              ),
            ),
            if (f.pendingReadings.isNotEmpty) ...[
              const SizedBox(height: Ds.s3),
              InlineNotice(
                icon: Icons.update_rounded,
                tone: Ds.amber,
                text:
                    '${f.pendingReadings.map((c) => c.label).join(', ')} arrived '
                    'after this composite was computed and are not included in '
                    'it. Pull down to re-run fusion.',
              ),
            ],
            if (chart.fusionFromCache) ...[
              const SizedBox(height: Ds.s3),
              InlineNotice(
                icon: Icons.history_rounded,
                tone: Ds.amber,
                text: 'Last result received from the backend'
                    '${f.updatedAt == null ? '' : ' on ${DateFormat('d MMM y, HH:mm').format(f.updatedAt!)}'}'
                    '. Not refreshed this session — pull down to retry.',
              ),
            ],
            if (chart.error != null) ...[
              const SizedBox(height: Ds.s3),
              InlineNotice(
                icon: Icons.cloud_off_rounded,
                tone: Ds.amber,
                text: chart.error!,
              ),
            ],
          ],
          const SizedBox(height: Ds.s6),
          SectionLabel('Where each signal comes from'),
          if (f == null)
            const Panel(
              child: Text(
                'Modality status appears once the fusion service has a record '
                'for this patient.',
                style:
                    TextStyle(fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
              ),
            )
          else
            ...f.contributions.map(_modalityRow),
          if (f != null && f.rejected.isNotEmpty) ...[
            const SizedBox(height: Ds.s3),
            Panel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('WHY MODALITIES WERE NOT USED', style: AppTheme.eyebrow),
                  const SizedBox(height: Ds.s3),
                  // The gate's own words. Not paraphrased, because the reason a
                  // reading was rejected is auditable information.
                  for (final e in f.rejected.entries)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Text(
                        '${Modality.labelFor(e.key)} — ${e.value}',
                        style: const TextStyle(
                            fontSize: 12, color: Ds.inkMuted, height: 1.45),
                      ),
                    ),
                ],
              ),
            ),
          ],
          const SizedBox(height: Ds.s3),
          const InlineNotice(
            icon: Icons.devices_rounded,
            text:
                'Physiological, behavioural and intake signals are collected by '
                'the patient-facing app. ClinAnx submits the clinical note to '
                'the Central Backend, which calls TC-WPN, applies the gate and '
                'the fusion weighting, and returns the result shown here.',
          ),
          const SizedBox(height: Ds.s6),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }

  Widget _modalityRow(ComponentContribution c) {
    final color = FusionBar.palette[c.key] ?? Ds.brand;
    final age = c.ageLabel;

    return Padding(
      padding: const EdgeInsets.only(bottom: Ds.s3),
      child: Panel(
        padding: const EdgeInsets.all(Ds.s4),
        child: Row(
          children: [
            Container(
              width: 34,
              height: 34,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.10),
                borderRadius: BorderRadius.circular(Ds.rSm),
              ),
              alignment: Alignment.center,
              child: Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: c.available ? color : Ds.hairlineStrong,
                  shape: BoxShape.circle,
                ),
              ),
            ),
            const SizedBox(width: Ds.s3),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Flexible(
                        child: Text(c.label,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                                fontSize: 13.5, fontWeight: FontWeight.w600)),
                      ),
                      if (c.isStale) ...[
                        const SizedBox(width: Ds.s2),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 1),
                          decoration: BoxDecoration(
                            color: Ds.amberSoft,
                            borderRadius: BorderRadius.circular(Ds.rSm),
                          ),
                          child: const Text('stale',
                              style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.w600,
                                  color: Ds.amber)),
                        ),
                      ],
                    ],
                  ),
                  Text(
                    c.available
                        ? '${c.origin}${age == null ? '' : ' · $age'}'
                        : (c.note ?? 'No reading recorded.'),
                    style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            if (c.available)
              Text(c.score!.toStringAsFixed(3),
                  style: AppTheme.data(size: 14, weight: FontWeight.w600))
            else
              Text('—', style: AppTheme.data(size: 14, color: Ds.inkFaint)),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Clinical notes — Component 4
// ─────────────────────────────────────────────────────────────────────────────

class _NotesSection extends StatelessWidget {
  final ChartController chart;
  const _NotesSection({required this.chart});

  @override
  Widget build(BuildContext context) {
    if (chart.notes.isEmpty) {
      return EmptyState(
        icon: Icons.description_outlined,
        title: 'No clinical notes yet',
        body:
            'Analyse a written note to produce a clinical-note risk score. This '
            'modality carries the largest weight in the composite.',
        actionLabel: 'Analyse a note',
        onAction: () => Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ChangeNotifierProvider.value(
              value: chart,
              child: const NoteAnalysisScreen(),
            ),
          ),
        ),
      );
    }

    return ListView(
      padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, 96),
      children: [
        Panel(
          background: Ds.brandSoft,
          borderColor: Ds.brandEdge,
          padding: const EdgeInsets.all(Ds.s4),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('SUPPORT SET', style: AppTheme.eyebrow),
                    const SizedBox(height: 2),
                    Text(
                      '${chart.effectiveSupportCount} labelled examples',
                      style: const TextStyle(
                          fontSize: 13.5, fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      chart.effectiveSupportCount == 0
                          ? 'Without labelled examples the model falls back to its meta-trained prototypes.'
                          : 'Prototypes are formed from these, weighted by recency and confidence.',
                      style: const TextStyle(
                          fontSize: 11.5, color: Ds.inkMuted, height: 1.35),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded, color: Ds.brand),
            ],
          ),
        ),
        const SizedBox(height: Ds.s5),
        SectionLabel('Note history'),
        ...chart.notes.map((n) => Padding(
              padding: const EdgeInsets.only(bottom: Ds.s3),
              child: _NoteTile(note: n, chart: chart),
            )),
      ],
    );
  }
}

class _NoteTile extends StatelessWidget {
  final ClinicalNote note;
  final ChartController chart;
  const _NoteTile({required this.note, required this.chart});

  @override
  Widget build(BuildContext context) {
    final r = note.result;
    return Panel(
      padding: const EdgeInsets.all(Ds.s4),
      // Every note is reachable. The previous build made a draft's tile inert
      // (`onTap: null` whenever there was no result), which is why a saved draft
      // could not be edited, deleted or analysed — there was no way back into it
      // at all. A note without an assessment opens the editor; a note with one
      // opens the assessment, which itself offers Edit.
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ChangeNotifierProvider.value(
            value: chart,
            child: note.hasBeenAnalysed
                ? TcwpnResultScreen(note: note)
                : NoteAnalysisScreen(existing: note),
          ),
        ),
      ),
      child: Row(
        children: [
          SizedBox(
            width: 44,
            child: Text(
              r == null ? '—' : r.calibratedProbability.toStringAsFixed(2),
              style: AppTheme.data(
                size: 15,
                weight: FontWeight.w600,
                color: r == null
                    ? Ds.inkFaint
                    : (r.isPositive ? Ds.red : Ds.green),
              ),
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Flexible(
                      child: Text(note.noteType,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                              fontSize: 13.5, fontWeight: FontWeight.w600)),
                    ),
                    if (note.clinicianVerdict != null) ...[
                      const SizedBox(width: Ds.s2),
                      Icon(
                        note.clinicianVerdict == 'agree'
                            ? Icons.verified_rounded
                            : Icons.flag_rounded,
                        size: 13,
                        color: note.clinicianVerdict == 'agree'
                            ? Ds.green
                            : Ds.amber,
                      ),
                    ],
                  ],
                ),
                Text(
                  DateFormat('d MMM y · HH:mm').format(note.recordedAt),
                  style: AppTheme.data(size: 10.5, color: Ds.inkFaint),
                ),
              ],
            ),
          ),
          if (r == null)
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: Ds.s3, vertical: 3),
              decoration: BoxDecoration(
                color: Ds.surfaceSunken,
                borderRadius: BorderRadius.circular(Ds.rPill),
              ),
              child: const Text('Draft',
                  style: TextStyle(fontSize: 11, color: Ds.inkMuted)),
            )
          else
            BandChip(band: AlertBandX.fromScore(r.calibratedProbability)),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Intervention — Component 3
// ─────────────────────────────────────────────────────────────────────────────

class _InterventionSection extends StatelessWidget {
  const _InterventionSection();

  @override
  Widget build(BuildContext context) => ListView(
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          const InlineNotice(
            icon: Icons.link_rounded,
            text: 'The intervention engine and its GAD-7 flow are owned by '
                'Component 3. This section renders its calibrated tier, conformal '
                'prediction set, and SHAP attribution once the C3 service address '
                'is configured in Settings.',
          ),
          const SizedBox(height: Ds.s5),
          SectionLabel('Expected from the C3 service'),
          const Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _Bullet('Calibrated risk tier with isotonic probabilities'),
                _Bullet(
                    'Adaptive Prediction Set at α = 0.10, with the singleton '
                    'rate surfaced so ambiguous cases are visible'),
                _Bullet('SHAP attribution over the 13-feature vector'),
                _Bullet(
                    'DiCE counterfactuals and FAISS-retrieved similar cases'),
                _Bullet('Session history and the composite reward that drives '
                    'per-patient adaptation'),
              ],
            ),
          ),
        ],
      );
}

class _Bullet extends StatelessWidget {
  final String text;
  const _Bullet(this.text);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s2),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 4,
              height: 4,
              margin: const EdgeInsets.only(top: 7, right: Ds.s3),
              decoration: const BoxDecoration(
                  color: Ds.hairlineStrong, shape: BoxShape.circle),
            ),
            Expanded(
              child: Text(text,
                  style: const TextStyle(
                      fontSize: 12.5, color: Ds.inkMuted, height: 1.45)),
            ),
          ],
        ),
      );
}
