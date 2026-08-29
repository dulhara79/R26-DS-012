// lib/features/explain/explain_screen.dart
//
// Explainable AI — why THIS patient carries THIS tier.
//
// Distinct from the Ask CARE tab. That tab queries the literature; this screen
// accounts for one fusion decision on one patient. Everything above the
// narrative is computed server-side and merely rendered here: contributions,
// weights, gate verdicts, and the conformal set arrive from the backend and are
// never recomputed on-device, because a number the clinician reads must be the
// same number the audit log recorded.
//
// The narrative at the bottom is generated, and is therefore fenced off from
// the arithmetic above it and labelled as such.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/design/tokens.dart';
import '../../domain/evidence.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';

class ExplainScreen extends StatefulWidget {
  const ExplainScreen({super.key});

  @override
  State<ExplainScreen> createState() => _ExplainScreenState();
}

class _ExplainScreenState extends State<ExplainScreen> {
  EvidenceResult? _narrative;
  bool _loadingNarrative = false;

  /// Builds the question put to CARE-AnxRAG from the fusion result.
  ///
  /// Only the composite, band and the names of contributing modalities are
  /// included. No identifiers, no demographics, no note text: the evidence
  /// layer answers a clinical question and has no need of the patient behind
  /// it, so it is not given one.
  String _composeQuestion(FusionResult f) {
    final used = f.contributions
        .where((c) => c.available && !c.excluded)
        .map((c) => c.label)
        .join(', ');
    final band = f.band.protocolName;
    final composite = f.compositeScore?.toStringAsFixed(2) ?? 'unavailable';

    return 'A patient has been assigned a $band risk band with a composite '
        'score of $composite, derived from these assessment modalities: '
        '$used. What does current evidence recommend for clinical management '
        'at this level of anxiety risk, and what should be monitored?';
  }

  Future<void> _generateNarrative(ChartController chart) async {
    final fusion = chart.fusion;
    if (fusion == null || _loadingNarrative) return;

    setState(() => _loadingNarrative = true);
    final raw = await chart.askEvidence(_composeQuestion(fusion));
    if (!mounted) return;
    setState(() {
      _narrative = raw == null
          ? EvidenceResult.failure('Decision support did not respond.')
          : EvidenceResult.fromJson(raw);
      _loadingNarrative = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final chart = context.watch<ChartController>();
    final fusion = chart.fusion;

    return Scaffold(
      backgroundColor: Ds.canvas,
      appBar: AppBar(
        title: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Why this tier?',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            Text('Explainable assessment',
                style: TextStyle(fontSize: 11, color: Ds.inkFaint)),
          ],
        ),
      ),
      body: fusion == null
          ? const _NoAssessment()
          : ListView(
              padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
              children: [
                _VerdictHeader(fusion: fusion),
                const SizedBox(height: Ds.s5),
                _GateSection(fusion: fusion),
                const SizedBox(height: Ds.s5),
                if (fusion.hasComposite) ...[
                  _ContributionSection(fusion: fusion),
                  const SizedBox(height: Ds.s5),
                ],
                if (fusion.conformal != null) ...[
                  _ConformalSection(conformal: fusion.conformal!),
                  const SizedBox(height: Ds.s5),
                ],
                _NarrativeSection(
                  narrative: _narrative,
                  loading: _loadingNarrative,
                  onGenerate: () => _generateNarrative(chart),
                ),
              ],
            ),
    );
  }
}

class _NoAssessment extends StatelessWidget {
  const _NoAssessment();

  @override
  Widget build(BuildContext context) => const Center(
        child: Padding(
          padding: EdgeInsets.all(Ds.s6),
          child: Text(
            'No assessment has been produced for this patient yet. Run a '
            'fusion from the patient chart first.',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 13, color: Ds.inkMuted, height: 1.5),
          ),
        ),
      );
}

// ── Verdict ──────────────────────────────────────────────────────────────────

class _VerdictHeader extends StatelessWidget {
  final FusionResult fusion;

  const _VerdictHeader({required this.fusion});

  @override
  Widget build(BuildContext context) {
    // A blocked assessment is GREY: it has no composite and must not be
    // painted on the severity scale, so it falls back to neutral surfaces.
    final scored = fusion.hasComposite;
    final tone = scored ? fusion.band.fg : Ds.grey;
    final wash = scored ? fusion.band.bg : Ds.greySoft;

    return Container(
      padding: const EdgeInsets.all(Ds.s5),
      decoration: BoxDecoration(
        color: wash,
        borderRadius: BorderRadius.circular(Ds.rLg),
        border: Border.all(color: tone.withValues(alpha: 0.3), width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('COMPOSITE',
                      style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                          color: Ds.inkFaint)),
                  const SizedBox(height: 2),
                  Text(
                    fusion.compositeLabel,
                    style: TextStyle(
                        fontSize: 34,
                        fontWeight: FontWeight.w600,
                        color: tone,
                        height: 1.1),
                  ),
                ],
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: Ds.s3, vertical: Ds.s2),
                decoration: BoxDecoration(
                  color: tone,
                  borderRadius: BorderRadius.circular(Ds.rPill),
                ),
                child: Text(
                  scored ? fusion.band.protocolName : 'NOT SCORED',
                  style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: Colors.white),
                ),
              ),
            ],
          ),
          if ((fusion.reason ?? '').isNotEmpty) ...[
            const SizedBox(height: Ds.s3),
            Text(fusion.reason!,
                style: const TextStyle(
                    fontSize: 12.5, color: Ds.inkMuted, height: 1.5)),
          ],
          const SizedBox(height: Ds.s3),
          Wrap(
            spacing: Ds.s4,
            runSpacing: Ds.s2,
            children: [
              _Fact('Modalities used', '${fusion.modalitiesUsed}'),
              if (fusion.confidence != null)
                _Fact('Confidence',
                    '${(fusion.confidence! * 100).round()}%'),
              _Fact('Weights',
                  fusion.renormalised ? 'Renormalised' : 'Framework default'),
            ],
          ),
        ],
      ),
    );
  }
}

class _Fact extends StatelessWidget {
  final String label;
  final String value;

  const _Fact(this.label, this.value);

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label,
              style: const TextStyle(fontSize: 10.5, color: Ds.inkFaint)),
          Text(value,
              style: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w600, color: Ds.ink)),
        ],
      );
}

// ── Gate ─────────────────────────────────────────────────────────────────────

class _GateSection extends StatelessWidget {
  final FusionResult fusion;

  const _GateSection({required this.fusion});

  @override
  Widget build(BuildContext context) {
    final rejected = fusion.rejected;
    final usable = fusion.usableModalities;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _Label('EVIDENCE GATE'),
        const SizedBox(height: Ds.s2),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(Ds.s4),
          decoration: BoxDecoration(
            color: Ds.surface,
            borderRadius: BorderRadius.circular(Ds.rMd),
            border: Border.all(color: Ds.hairline, width: 0.5),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Fusion requires at least two usable modalities. The gate '
                'decides which readings qualify before any score is computed.',
                style:
                    TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
              ),
              const SizedBox(height: Ds.s3),
              for (final key in usable)
                _GateRow(
                  label: Modality.labelFor(key),
                  accepted: true,
                  detail: 'Accepted into the composite',
                ),
              for (final entry in rejected.entries)
                _GateRow(
                  label: Modality.labelFor(entry.key),
                  accepted: false,
                  detail: entry.value,
                ),
              if (usable.isEmpty && rejected.isEmpty)
                const Text('No gate detail reported.',
                    style: TextStyle(fontSize: 12, color: Ds.inkFaint)),
            ],
          ),
        ),
      ],
    );
  }
}

class _GateRow extends StatelessWidget {
  final String label;
  final bool accepted;
  final String detail;

  const _GateRow({
    required this.label,
    required this.accepted,
    required this.detail,
  });

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(
              accepted
                  ? Icons.check_circle_outline_rounded
                  : Icons.remove_circle_outline_rounded,
              size: 17,
              color: accepted ? Ds.green : Ds.inkFaint,
            ),
            const SizedBox(width: Ds.s2),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: const TextStyle(
                          fontSize: 13, fontWeight: FontWeight.w600)),
                  Text(detail,
                      style: const TextStyle(
                          fontSize: 11.5, color: Ds.inkFaint, height: 1.4)),
                ],
              ),
            ),
          ],
        ),
      );
}

// ── Contributions ────────────────────────────────────────────────────────────

class _ContributionSection extends StatelessWidget {
  final FusionResult fusion;

  const _ContributionSection({required this.fusion});

  @override
  Widget build(BuildContext context) {
    final contributing =
        fusion.contributions.where((c) => c.contribution != null).toList()
          ..sort((a, b) => (b.contribution ?? 0).compareTo(a.contribution ?? 0));

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _Label('WHAT DROVE THIS SCORE'),
        const SizedBox(height: Ds.s2),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(Ds.s4),
          decoration: BoxDecoration(
            color: Ds.surface,
            borderRadius: BorderRadius.circular(Ds.rMd),
            border: Border.all(color: Ds.hairline, width: 0.5),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Each modality contributes its score multiplied by a weight '
                'derived from that component\'s validation performance above '
                'chance, decayed by the age of the reading.',
                style:
                    TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
              ),
              const SizedBox(height: Ds.s4),
              for (final c in contributing) _ContributionRow(contribution: c),
              if (contributing.isEmpty)
                const Text('No modality contributed to this assessment.',
                    style: TextStyle(fontSize: 12, color: Ds.inkFaint)),
            ],
          ),
        ),
      ],
    );
  }
}

class _ContributionRow extends StatelessWidget {
  final ComponentContribution contribution;

  const _ContributionRow({required this.contribution});

  @override
  Widget build(BuildContext context) {
    final share = (contribution.contribution ?? 0).clamp(0.0, 1.0);
    final tone = FusionBarPalette.forKey(contribution.key);

    return Padding(
      padding: const EdgeInsets.only(bottom: Ds.s4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(contribution.label,
                    style: const TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w600)),
              ),
              Text(
                contribution.score == null
                    ? '—'
                    : contribution.score!.toStringAsFixed(3),
                style: const TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: Ds.ink),
              ),
            ],
          ),
          const SizedBox(height: Ds.s2),
          ClipRRect(
            borderRadius: BorderRadius.circular(Ds.rPill),
            child: LinearProgressIndicator(
              value: share,
              minHeight: 6,
              backgroundColor: Ds.surfaceSunken,
              valueColor: AlwaysStoppedAnimation(tone),
            ),
          ),
          const SizedBox(height: Ds.s1),
          Text(
            'weight ${contribution.weight.toStringAsFixed(3)}  ·  '
            'contributes ${(share * 100).toStringAsFixed(1)}%'
            '${contribution.ageLabel == null ? '' : '  ·  ${contribution.ageLabel}'}',
            style: const TextStyle(fontSize: 11, color: Ds.inkFaint),
          ),
        ],
      ),
    );
  }
}

/// Keeps the modality colour mapping in one place. Mirrors the palette used by
/// FusionBar so a modality is the same colour everywhere in the app.
class FusionBarPalette {
  static const _map = {
    'c1_physiological': Color(0xFF4C6EF5),
    'c2_behavioral': Color(0xFF2F9E68),
    'c3_clinical_nlp': Color(0xFF0F5B6E),
    'c4_demographic': Color(0xFFB5651D),
  };

  static Color forKey(String key) => _map[key] ?? Ds.brand;
}

// ── Conformal ────────────────────────────────────────────────────────────────

class _ConformalSection extends StatelessWidget {
  final Map<String, dynamic> conformal;

  const _ConformalSection({required this.conformal});

  @override
  Widget build(BuildContext context) {
    final raw = conformal['prediction_set'];
    final set = raw is List ? raw.map((e) => '$e').toList() : <String>[];
    final coverage = conformal['coverage'] ?? conformal['alpha'];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _Label('PREDICTION SET'),
        const SizedBox(height: Ds.s2),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(Ds.s4),
          decoration: BoxDecoration(
            color: Ds.brandSoft,
            borderRadius: BorderRadius.circular(Ds.rMd),
            border: Border.all(color: Ds.brandEdge, width: 0.5),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Conformal prediction reports every tier that cannot be ruled '
                'out at the target coverage level. A set with more than one '
                'tier is the model stating it cannot separate them, and that '
                'is a finding rather than a defect.',
                style:
                    TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
              ),
              const SizedBox(height: Ds.s3),
              if (set.isEmpty)
                const Text('No prediction set reported.',
                    style: TextStyle(fontSize: 12, color: Ds.inkFaint))
              else
                Wrap(
                  spacing: Ds.s2,
                  runSpacing: Ds.s2,
                  children: [
                    for (final tier in set)
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: Ds.s3, vertical: Ds.s2),
                        decoration: BoxDecoration(
                          color: Ds.surface,
                          borderRadius: BorderRadius.circular(Ds.rSm),
                          border:
                              Border.all(color: Ds.brandEdge, width: 0.5),
                        ),
                        child: Text(tier,
                            style: const TextStyle(
                                fontSize: 12.5,
                                fontWeight: FontWeight.w600,
                                color: Ds.brandDeep)),
                      ),
                  ],
                ),
              if (coverage != null) ...[
                const SizedBox(height: Ds.s3),
                Text('Target coverage: $coverage',
                    style:
                        const TextStyle(fontSize: 11, color: Ds.inkFaint)),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

// ── Generated narrative ──────────────────────────────────────────────────────

class _NarrativeSection extends StatelessWidget {
  final EvidenceResult? narrative;
  final bool loading;
  final VoidCallback onGenerate;

  const _NarrativeSection({
    required this.narrative,
    required this.loading,
    required this.onGenerate,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _Label('EVIDENCE-BASED GUIDANCE'),
        const SizedBox(height: Ds.s2),
        if (narrative == null && !loading)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(Ds.s4),
            decoration: BoxDecoration(
              color: Ds.surface,
              borderRadius: BorderRadius.circular(Ds.rMd),
              border: Border.all(color: Ds.hairline, width: 0.5),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Ask CARE-AnxRAG what the evidence recommends at this risk '
                  'level. Only the risk band and the contributing modality '
                  'names are sent — no identifiers and no note text.',
                  style: TextStyle(
                      fontSize: 12, color: Ds.inkMuted, height: 1.5),
                ),
                const SizedBox(height: Ds.s3),
                FilledButton.icon(
                  onPressed: onGenerate,
                  style: FilledButton.styleFrom(backgroundColor: Ds.brand),
                  icon: const Icon(Icons.auto_awesome_outlined, size: 17),
                  label: const Text('Generate guidance'),
                ),
              ],
            ),
          ),
        if (loading)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: Ds.s6),
            child: Column(
              children: [
                SizedBox(
                    width: 26,
                    height: 26,
                    child: CircularProgressIndicator(
                        strokeWidth: 2.5, color: Ds.brand)),
                SizedBox(height: Ds.s3),
                Text('Retrieving and appraising evidence…',
                    style: TextStyle(fontSize: 12.5, color: Ds.inkMuted)),
                SizedBox(height: 2),
                Text('Local generation can take up to a minute.',
                    style: TextStyle(fontSize: 11, color: Ds.inkFaint)),
              ],
            ),
          ),
        if (narrative != null && !loading) _NarrativeBody(result: narrative!),
      ],
    );
  }
}

class _NarrativeBody extends StatelessWidget {
  final EvidenceResult result;

  const _NarrativeBody({required this.result});

  @override
  Widget build(BuildContext context) {
    // The same four honesty states as the Ask CARE tab. An explainability
    // screen that invented guidance when the evidence layer declined would be
    // worse than one that showed nothing.
    switch (result.state) {
      case EvidenceState.crisisBypass:
        return _Notice(
          tone: Ds.darkRed,
          wash: Ds.darkRedSoft,
          icon: Icons.warning_amber_rounded,
          title: 'Crisis pre-screen matched',
          body: result.safetyMessage ??
              'Follow the ward\'s crisis protocol. The evidence layer was not '
                  'consulted.',
        );
      case EvidenceState.abstained:
        return _Notice(
          tone: Ds.amber,
          wash: Ds.amberSoft,
          icon: Icons.help_outline_rounded,
          title: 'Insufficient evidence',
          body: result.abstentionReason ??
              'CARE-AnxRAG declined to answer rather than generate an '
                  'unsupported recommendation.',
        );
      case EvidenceState.unavailable:
        return _Notice(
          tone: Ds.grey,
          wash: Ds.greySoft,
          icon: Icons.cloud_off_rounded,
          title: 'Decision support unavailable',
          body: result.error ?? 'CARE-AnxRAG could not be reached.',
        );
      case EvidenceState.answered:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(Ds.s4),
              decoration: BoxDecoration(
                color: Ds.surface,
                borderRadius: BorderRadius.circular(Ds.rMd),
                border: Border.all(color: Ds.hairline, width: 0.5),
              ),
              child: Text(result.answer ?? '',
                  style: const TextStyle(
                      fontSize: 13.5, color: Ds.ink, height: 1.65)),
            ),
            const SizedBox(height: Ds.s3),
            Row(
              children: [
                Icon(Icons.verified_outlined,
                    size: 15,
                    color: result.hasConflict ? Ds.amber : Ds.green),
                const SizedBox(width: Ds.s1),
                Text(
                  'Evidence confidence ${result.confidencePercent ?? '—'}%  ·  '
                  'conflict ${result.conflictLabel.toLowerCase()}  ·  '
                  '${result.citations.length} sources',
                  style:
                      const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                ),
              ],
            ),
            const SizedBox(height: Ds.s2),
            const Text(
              'Generated from retrieved evidence. It is guidance, not a '
              'diagnosis or a treatment decision.',
              style: TextStyle(fontSize: 11, color: Ds.inkFaint, height: 1.4),
            ),
          ],
        );
    }
  }
}

class _Notice extends StatelessWidget {
  final Color tone;
  final Color wash;
  final IconData icon;
  final String title;
  final String body;

  const _Notice({
    required this.tone,
    required this.wash,
    required this.icon,
    required this.title,
    required this.body,
  });

  @override
  Widget build(BuildContext context) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(Ds.s4),
        decoration: BoxDecoration(
          color: wash,
          borderRadius: BorderRadius.circular(Ds.rMd),
          border: Border.all(color: tone.withValues(alpha: 0.4), width: 0.5),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 18, color: tone),
                const SizedBox(width: Ds.s2),
                Expanded(
                  child: Text(title,
                      style: TextStyle(
                          fontSize: 13.5,
                          fontWeight: FontWeight.w700,
                          color: tone)),
                ),
              ],
            ),
            const SizedBox(height: Ds.s2),
            Text(body,
                style: const TextStyle(
                    fontSize: 12.5, color: Ds.ink, height: 1.5)),
          ],
        ),
      );
}

class _Label extends StatelessWidget {
  final String text;

  const _Label(this.text);

  @override
  Widget build(BuildContext context) => Text(
        text,
        style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.8,
            color: Ds.inkFaint),
      );
}
