// lib/features/fusion/fusion_detail_screen.dart
//
// The late-fusion equation, expanded. Every number on the chart can be traced
// from here back to the modality that produced it, including the ones that
// didn't report.

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
                segments: [
                  for (final c in fusion.contributions)
                    FusionSegment(
                      key: c.key,
                      label: c.label,
                      contribution: c.contribution,
                      weight: c.weight,
                      score: c.score,
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
                  Icon(Icons.assignment_outlined,
                      size: 17, color: fusion.band.fg),
                  const SizedBox(width: Ds.s3),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('${fusion.band.protocolName} band',
                            style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: fusion.band.fg)),
                        const SizedBox(height: 3),
                        Text(fusion.band.guidance,
                            style: const TextStyle(
                                fontSize: 12.5, color: Ds.inkMuted, height: 1.45)),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: Ds.s5),

            SectionLabel('Band thresholds'),
            Panel(
              child: Column(
                children: [
                  for (final t in const [
                    (AlertBand.green, '0.00 – 0.24'),
                    (AlertBand.amber, '0.25 – 0.49'),
                    (AlertBand.red, '0.50 – 0.74'),
                    (AlertBand.darkRed, '0.75 – 1.00'),
                  ])
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 5),
                      child: Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                                color: t.$1.fg, shape: BoxShape.circle),
                          ),
                          const SizedBox(width: Ds.s3),
                          Expanded(
                            child: Text(
                              t.$1.protocolName,
                              style: TextStyle(
                                fontSize: 12.5,
                                fontWeight: t.$1 == fusion.band
                                    ? FontWeight.w700
                                    : FontWeight.w500,
                                color:
                                    t.$1 == fusion.band ? Ds.ink : Ds.inkMuted,
                              ),
                            ),
                          ),
                          Text(t.$2,
                              style:
                                  AppTheme.data(size: 11.5, color: Ds.inkMuted)),
                        ],
                      ),
                    ),
                ],
              ),
            ),
            const SizedBox(height: Ds.s5),

            SectionLabel('Provenance'),
            Panel(
              child: Column(
                children: [
                  _kv('Computed',
                      DateFormat('d MMM y, HH:mm').format(fusion.computedAt)),
                  _kv('Modalities',
                      '${fusion.modalitiesAvailable} of 4 reporting'),
                  _kv(
                      'Weights',
                      fusion.renormalised
                          ? 'Rescaled across reporting modalities'
                          : 'Framework weights, unmodified'),
                  _kv(
                      'Source',
                      fusion.computedLocally
                          ? 'Provisional, calculated on this device'
                          : 'Fusion service'),
                  if (fusion.confidence != null)
                    _kv('Fusion confidence',
                        fusion.confidence!.toStringAsFixed(3)),
                ],
              ),
            ),

            if (latestNote?.result != null) ...[
              const SizedBox(height: Ds.s5),
              SectionLabel('Largest contributor'),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Clinical notes carry the heaviest fusion weight, so this '
                      'score moves the composite more than any other single '
                      'modality.',
                      style: TextStyle(
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
