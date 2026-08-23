// lib/features/fusion/fusion_detail_screen.dart
//
// The composite, expanded. Every number on this screen came off the wire from
// the Central Backend; none of it is recomputed here.
//
// WHAT CHANGED
// ------------
// The old screen printed a four-way band table at 0.25/0.50/0.75 as though it
// were the framework's. It is not: the fusion service bands into THREE tiers at
// 0.33/0.66 (fusion_service/fusion.py, BANDS). Printing a different table beside
// a server-produced tier invited a clinician to check one against the other and
// find them inconsistent. The table is now driven by the server's own tier.
//
// It also carried a "Provisional, calculated on this device" provenance line.
// There is no on-device fusion path any more, so that line is gone rather than
// left to describe a branch that cannot execute.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../domain/models.dart';

class FusionDetailScreen extends StatelessWidget {
  final FusionResult fusion;
  final Patient patient;
  final ClinicalNote? latestNote;

  const FusionDetailScreen({
    super.key,
    required this.fusion,
    required this.patient,
    this.latestNote,
  });

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Composite risk')),
        body: ListView(
          padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
          children: [
            Panel(
              padding: const EdgeInsets.all(Ds.s5),
              child: FusionBar(
                composite: fusion.compositeScore,
                band: fusion.band,
                renormalised: fusion.renormalised,
                blockedReason: fusion.reason,
                segments: [
                  for (final c in fusion.contributions)
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
            ),
            const SizedBox(height: Ds.s4),

            Panel(
              background: fusion.band.bg,
              borderColor: fusion.band.fg.withValues(alpha: 0.22),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    fusion.hasComposite
                        ? Icons.assignment_outlined
                        : Icons.block_flipped,
                    size: 17,
                    color: fusion.band.fg,
                  ),
                  const SizedBox(width: Ds.s3),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          fusion.hasComposite
                              ? '${fusion.band.protocolName} band'
                              : 'No composite computed',
                          style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: fusion.band.fg),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          // The server's own words first. Its gate reason is
                          // more specific than any generic band guidance.
                          fusion.reason ?? fusion.band.guidance,
                          style: const TextStyle(
                              fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // ── Gate ───────────────────────────────────────────────────────
            if (fusion.rejected.isNotEmpty ||
                fusion.usableModalities.isNotEmpty) ...[
              const SizedBox(height: Ds.s5),
              SectionLabel('Fusion gate'),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'A composite is produced only when at least two modalities '
                      'are usable and at least one of them varies over time. '
                      'Readings that failed this check are listed with the '
                      'reason the gate gave.',
                      style: TextStyle(
                          fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
                    ),
                    const SizedBox(height: Ds.s4),
                    if (fusion.usableModalities.isNotEmpty)
                      _kv(
                        'Used',
                        fusion.usableModalities
                            .map(Modality.labelFor)
                            .join(', '),
                      ),
                    for (final e in fusion.rejected.entries)
                      _kv(Modality.labelFor(e.key), e.value),
                  ],
                ),
              ),
            ],

            // ── Tier ───────────────────────────────────────────────────────
            const SizedBox(height: Ds.s5),
            SectionLabel('Tier'),
            Panel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'The tier below is assigned by the fusion service, not '
                    'derived on this device. It is shown as the service '
                    'reported it.',
                    style: TextStyle(
                        fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
                  ),
                  const SizedBox(height: Ds.s4),
                  for (final t in const ['Low', 'Medium', 'High'])
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 5),
                      child: Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: t == fusion.tier
                                  ? Ds.brand
                                  : Ds.hairlineStrong,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: Ds.s3),
                          Expanded(
                            child: Text(
                              t,
                              style: TextStyle(
                                fontSize: 12.5,
                                fontWeight: t == fusion.tier
                                    ? FontWeight.w700
                                    : FontWeight.w500,
                                color: t == fusion.tier ? Ds.ink : Ds.inkMuted,
                              ),
                            ),
                          ),
                          if (t == fusion.tier)
                            Text('assigned',
                                style: AppTheme.data(
                                    size: 11.5, color: Ds.inkMuted)),
                        ],
                      ),
                    ),
                  if (fusion.tier == null)
                    const Padding(
                      padding: EdgeInsets.only(top: Ds.s2),
                      child: Text(
                        'No tier was assigned, because no composite was computed.',
                        style: TextStyle(fontSize: 12, color: Ds.inkFaint),
                      ),
                    ),
                ],
              ),
            ),

            // ── Conformal ──────────────────────────────────────────────────
            if (fusion.conformal != null) ...[
              const SizedBox(height: Ds.s5),
              SectionLabel('Conformal prediction set'),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (final e in fusion.conformal!.entries)
                      _kv(e.key.replaceAll('_', ' '), '${e.value}'),
                    const SizedBox(height: Ds.s3),
                    const Text(
                      'Record your own tier judgement BEFORE reading this set. '
                      'A judgement made after seeing it is contaminated by the '
                      'prediction it exists to calibrate.',
                      style: TextStyle(
                          fontSize: 12, color: Ds.inkMuted, height: 1.45),
                    ),
                  ],
                ),
              ),
            ],

            // ── Provenance ─────────────────────────────────────────────────
            const SizedBox(height: Ds.s5),
            SectionLabel('Provenance'),
            Panel(
              child: Column(
                children: [
                  _kv(
                    'Computed',
                    fusion.updatedAt == null
                        ? 'Not yet'
                        : DateFormat('d MMM y, HH:mm')
                            .format(fusion.updatedAt!),
                  ),
                  _kv('Modalities',
                      '${fusion.modalitiesUsed} of ${Modality.all.length} used in the composite'),
                  _kv(
                      'Weights',
                      fusion.renormalised
                          ? 'Rescaled across the modalities that reported'
                          : 'Framework weights, unmodified'),
                  _kv('Source', 'R26-DS-012 Central Backend'),
                  if (fusion.fusionResultId != null)
                    _kv('Fusion record', '#${fusion.fusionResultId}'),
                  if (fusion.confidence != null)
                    _kv('Fusion confidence',
                        fusion.confidence!.toStringAsFixed(3)),
                ],
              ),
            ),

            // ── Largest contributor ────────────────────────────────────────
            if (latestNote?.result != null) ...[
              const SizedBox(height: Ds.s5),
              SectionLabel('Most recent clinical note'),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _clinicalNoteWeightNote(),
                      style: const TextStyle(
                          fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
                    ),
                    const SizedBox(height: Ds.s4),
                    Row(
                      children: [
                        Expanded(
                          child: Readout(
                            label: 'NOTE RISK',
                            value: latestNote!.result!.calibratedProbability
                                .toStringAsFixed(3),
                          ),
                        ),
                        Expanded(
                          child: Readout(
                            label: 'CONFIDENCE',
                            value:
                                '${(latestNote!.result!.confidence * 100).round()}%',
                          ),
                        ),
                        Expanded(
                          child: Readout(
                            label: 'SHOTS',
                            value: '${latestNote!.result!.supportK}',
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],

            const SizedBox(height: Ds.s5),
            const DecisionSupportNotice(),
          ],
        ),
      );

  /// Describes the clinical-note modality's actual share of THIS composite,
  /// rather than asserting a fixed ranking. The weights are renormalised per
  /// assessment over whichever modalities reported, so "clinical notes carry the
  /// heaviest weight" is not reliably true of any given patient.
  String _clinicalNoteWeightNote() {
    final c = fusion.contributions
        .where((x) => x.key == Modality.c3ClinicalNlp)
        .toList();
    if (c.isEmpty || !c.first.available) {
      return 'This note\u2019s score is not currently part of the composite.';
    }
    final pct = (c.first.weight * 100).round();
    return 'This note\u2019s score carries $pct% of the weight in the current '
        'composite, after rescaling across the modalities that reported.';
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 118,
              child: Text(k,
                  style: const TextStyle(fontSize: 12.5, color: Ds.inkMuted)),
            ),
            Expanded(
                child: Text(v, style: AppTheme.data(size: 11.5, height: 1.5))),
          ],
        ),
      );
}
