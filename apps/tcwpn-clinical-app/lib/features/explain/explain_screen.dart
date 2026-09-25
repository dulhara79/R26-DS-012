// lib/features/explain/explain_screen.dart
//
// Explainable assessment for one patient. A real fusion result always takes
// precedence. In demo builds only, a blocked fusion can render a clearly marked
// display-only low-risk example so the investor walkthrough still has a useful
// explainability surface. The example is never written to patient state.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/config/env.dart';
import '../../core/design/tokens.dart';
import '../../domain/evidence.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';

const double demoLowRiskScore = 0.2033;

class ExplainScreen extends StatefulWidget {
  const ExplainScreen({super.key});

  @override
  State<ExplainScreen> createState() => _ExplainScreenState();
}

class _ExplainScreenState extends State<ExplainScreen> {
  EvidenceResult? _narrative;
  bool _loadingNarrative = false;

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

  String _composeDemoQuestion() =>
      'This is a display-only low-risk anxiety demo example with an overall '
      'risk score of ${demoLowRiskScore.toStringAsFixed(2)} (Low). What does '
      'current evidence recommend for routine clinical monitoring and follow-up '
      'at a low anxiety risk level?';

  Future<void> _generateNarrative(ChartController chart) async {
    final fusion = chart.fusion;
    if (fusion == null || _loadingNarrative) return;

    final question = fusion.hasComposite
        ? _composeQuestion(fusion)
        : Env.demoData
            ? _composeDemoQuestion()
            : null;
    if (question == null) return;

    setState(() => _loadingNarrative = true);
    final raw = await chart.askEvidence(question);
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
                if (fusion.conformal != null && fusion.hasComposite) ...[
                  _ConformalSection(conformal: fusion.conformal!),
                  const SizedBox(height: Ds.s5),
                ],
                _NarrativeSection(
                  narrative: _narrative,
                  loading: _loadingNarrative,
                  enabled: fusion.hasComposite || Env.demoData,
                  demo: !fusion.hasComposite && Env.demoData,
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

class _VerdictHeader extends StatelessWidget {
  final FusionResult fusion;

  const _VerdictHeader({required this.fusion});

  @override
  Widget build(BuildContext context) {
    final real = fusion.hasComposite;
    final demo = !real && Env.demoData;
    final tone = real ? fusion.band.fg : demo ? Ds.green : Ds.grey;
    final wash = real ? fusion.band.bg : demo ? Ds.greenSoft : Ds.greySoft;
    final score = real
        ? fusion.compositeLabel
        : demo
            ? demoLowRiskScore.toStringAsFixed(4)
            : '—';
    final band = real ? fusion.band.protocolName : demo ? 'LOW' : 'NOT SCORED';

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
          Text(real ? 'OVERALL ANXIETY RISK' : demo ? 'DEMO OVERALL ANXIETY RISK' : 'COMPOSITE',
              style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                  color: Ds.inkFaint)),
          const SizedBox(height: Ds.s2),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Text(score,
                    style: TextStyle(
                        fontSize: 34,
                        fontWeight: FontWeight.w600,
                        color: tone,
                        height: 1.1)),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: Ds.s3, vertical: Ds.s2),
                decoration: BoxDecoration(
                  color: tone,
                  borderRadius: BorderRadius.circular(Ds.rPill),
                ),
                child: Text(band,
                    style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        color: Colors.white)),
              ),
            ],
          ),
          const SizedBox(height: Ds.s3),
          if (real) ...[
            const Text('Real backend fusion result',
                style: TextStyle(
                    fontSize: 12.5,
                    color: Ds.green,
                    fontWeight: FontWeight.w600)),
            if ((fusion.reason ?? '').isNotEmpty) ...[
              const SizedBox(height: Ds.s2),
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
                _Fact('Assessment status', fusion.assessmentLabel),
                if (fusion.confidence != null)
                  _Fact(
                      'Confidence',
                      fusion.confidence!.isFinite
                          ? '${(fusion.confidence! * 100).round()}%'
                          : '—'),
                _Fact('Weights',
                    fusion.renormalised ? 'Renormalised' : 'Framework default'),
              ],
            ),
          ] else if (demo) ...[
            const Text('Demo low-risk example',
                style: TextStyle(
                    fontSize: 12.5,
                    color: Ds.green,
                    fontWeight: FontWeight.w700)),
            const SizedBox(height: Ds.s2),
            const Text(
              'Current demo example indicates a low level of anxiety risk. '
              'Continue routine monitoring and clinical review.',
              style: TextStyle(fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
            ),
            const SizedBox(height: Ds.s3),
            const Text(
              'A complete overall assessment is not available for this demo '
              'patient, so this display-only example is shown. It does not '
              'replace, save, or modify a fusion result.',
              style: TextStyle(fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
            ),
          ] else ...[
            if ((fusion.reason ?? '').isNotEmpty)
              Text(fusion.reason!,
                  style: const TextStyle(
                      fontSize: 12.5, color: Ds.inkMuted, height: 1.5)),
          ],
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
                style: TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
              ),
              const SizedBox(height: Ds.s3),
              for (final key in usable)
                _GateRow(
                    label: Modality.labelFor(key),
                    accepted: true,
                    detail: 'Accepted into the composite'),
              for (final entry in rejected.entries)
                _GateRow(
                    label: Modality.labelFor(entry.key),
                    accepted: false,
                    detail: entry.value),
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
  const _GateRow(
      {required this.label, required this.accepted, required this.detail});

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
                style: TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
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
          Row(children: [
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
                  fontSize: 12.5, fontWeight: FontWeight.w600, color: Ds.ink),
            ),
          ]),
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

class FusionBarPalette {
  static const _map = {
    'c1_physiological': Color(0xFF4C6EF5),
    'c2_behavioral': Color(0xFF2F9E68),
    'c3_clinical_nlp': Color(0xFF0F5B6E),
    'c4_demographic': Color(0xFFB5651D),
  };
  static Color forKey(String key) => _map[key] ?? Ds.brand;
}

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
                'tier means the model cannot separate them.',
                style: TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
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
                          border: Border.all(color: Ds.brandEdge, width: 0.5),
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
                    style: const TextStyle(fontSize: 11, color: Ds.inkFaint)),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _NarrativeSection extends StatelessWidget {
  final EvidenceResult? narrative;
  final bool loading;
  final bool enabled;
  final bool demo;
  final VoidCallback onGenerate;
  const _NarrativeSection({
    required this.narrative,
    required this.loading,
    required this.enabled,
    required this.demo,
    required this.onGenerate,
  });

  @override
  Widget build(BuildContext context) => Column(
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
                  Text(
                    demo
                        ? 'Generate evidence guidance for the display-only low-risk demo example. No patient identifiers or note text are sent.'
                        : 'Ask CARE-AnxRAG what the evidence recommends at this risk level. Only the risk band and contributing modality names are sent — no identifiers and no note text.',
                    style: const TextStyle(
                        fontSize: 12, color: Ds.inkMuted, height: 1.5),
                  ),
                  const SizedBox(height: Ds.s3),
                  FilledButton.icon(
                    onPressed: enabled ? onGenerate : null,
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
          if (narrative != null && !loading)
            _NarrativeBody(result: narrative!),
        ],
      );
}

class _NarrativeBody extends StatelessWidget {
  final EvidenceResult result;
  const _NarrativeBody({required this.result});

  @override
  Widget build(BuildContext context) {
    switch (result.state) {
      case EvidenceState.crisisBypass:
        return _Notice(
          tone: Ds.darkRed,
          wash: Ds.darkRedSoft,
          icon: Icons.warning_amber_rounded,
          title: 'Crisis pre-screen matched',
          body: result.safetyMessage ??
              'Follow the ward\'s crisis protocol. The evidence layer was not consulted.',
        );
      case EvidenceState.abstained:
        return _Notice(
          tone: Ds.amber,
          wash: Ds.amberSoft,
          icon: Icons.help_outline_rounded,
          title: 'Insufficient evidence',
          body: result.abstentionReason ??
              'CARE-AnxRAG declined to answer rather than generate an unsupported recommendation.',
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
            Row(children: [
              Icon(Icons.verified_outlined,
                  size: 15, color: result.hasConflict ? Ds.amber : Ds.green),
              const SizedBox(width: Ds.s1),
              Expanded(
                child: Text(
                  'Evidence confidence ${result.confidencePercent ?? '—'}%  ·  conflict ${result.conflictLabel.toLowerCase()}  ·  ${result.citations.length} sources',
                  style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                ),
              ),
            ]),
            const SizedBox(height: Ds.s2),
            const Text(
              'Generated from retrieved evidence. It is guidance, not a diagnosis or a treatment decision.',
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
            Row(children: [
              Icon(icon, size: 18, color: tone),
              const SizedBox(width: Ds.s2),
              Expanded(
                child: Text(title,
                    style: TextStyle(
                        fontSize: 13.5,
                        fontWeight: FontWeight.w700,
                        color: tone)),
              ),
            ]),
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
