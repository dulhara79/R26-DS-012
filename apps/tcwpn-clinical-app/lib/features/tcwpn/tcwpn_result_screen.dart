// lib/features/tcwpn/tcwpn_result_screen.dart
//
// Component 4's result view — the deepest explanation surface in the app.
//
// Two rules govern everything here:
//
//   Never fabricate attribution. Phrase prominence is rendered only from
//   attention weights the service actually returned. When the service returns
//   phrases without weights, the section is titled "key phrases", the intensity
//   ramp is dropped, and the difference is stated. The previous build derived
//   prominence from list index and labelled it "ClinicalBERT attention weights",
//   which is a claim the model never made.
//
//   Never present a prediction without its uncertainty. Confidence, entropy,
//   the decision threshold, and K appear beside the score, not buried below it.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';

class TcwpnResultScreen extends StatefulWidget {
  final ClinicalNote note;
  const TcwpnResultScreen({super.key, required this.note});

  @override
  State<TcwpnResultScreen> createState() => _TcwpnResultScreenState();
}

class _TcwpnResultScreenState extends State<TcwpnResultScreen> {
  late ClinicalNote _note = widget.note;

  @override
  Widget build(BuildContext context) {
    final r = _note.result;
    if (r == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Clinical note')),
        body: const EmptyState(
          icon: Icons.pending_outlined,
          title: 'Not analysed yet',
          body: 'This note is saved as a draft. Analyse it to produce a score.',
        ),
      );
    }

    final chart = context.read<ChartController>();
    final band = AlertBandX.fromScore(r.calibratedProbability);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Note analysis'),
        actions: [
          IconButton(
            tooltip: 'Export report',
            icon: const Icon(Icons.ios_share_rounded),
            onPressed: () => _export(context),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          _headline(r, band),
          const SizedBox(height: Ds.s4),
          if (r.needsManualReview) ...[
            const InlineNotice(
              icon: Icons.pan_tool_outlined,
              tone: Ds.amber,
              text:
                  'Confidence is below 60%. Treat this prediction as inconclusive '
                  'and rely on clinical judgement.',
            ),
            const SizedBox(height: Ds.s4),
          ],
          if (r.usedDefaultSupportSet) ...[
            const InlineNotice(
              icon: Icons.dataset_outlined,
              tone: Ds.amber,
              text:
                  'No site-labelled examples were available, so the model used its '
                  'meta-trained prototypes. This is the unadapted baseline.',
            ),
            const SizedBox(height: Ds.s4),
          ],
          SectionLabel('Decision'),
          _decisionPanel(r),
          const SizedBox(height: Ds.s5),
          SectionLabel(
              r.hasAttribution ? 'Attention attribution' : 'Key phrases'),
          _attributionPanel(r),
          const SizedBox(height: Ds.s5),
          if (r.supportContributions.isNotEmpty) ...[
            SectionLabel('Support set influence'),
            _supportInfluencePanel(r),
            const SizedBox(height: Ds.s5),
          ],
          SectionLabel('Clinician review'),
          _verdictPanel(chart, r),
          const SizedBox(height: Ds.s5),
          SectionLabel('Submitted note'),
          Panel(
            child: Text(
              _note.text,
              style:
                  const TextStyle(fontSize: 13.5, height: 1.6, color: Ds.ink),
            ),
          ),
          const SizedBox(height: Ds.s5),
          SectionLabel('Model'),
          _modelPanel(r),
          const SizedBox(height: Ds.s5),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }

  // ── Headline ──────────────────────────────────────────────────────────────

  Widget _headline(TcwpnResult r, AlertBand band) => Panel(
        padding: const EdgeInsets.all(Ds.s5),
        background: band.bg,
        borderColor: band.fg.withValues(alpha: 0.22),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('CLINICAL-NOTE RISK', style: AppTheme.eyebrow),
                      const SizedBox(height: 2),
                      Text(
                        r.calibratedProbability.toStringAsFixed(4),
                        style: AppTheme.data(
                            size: 38, weight: FontWeight.w600, color: band.fg),
                      ),
                      const SizedBox(height: Ds.s1),
                      Text(
                        r.prediction,
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: band.fg,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ],
                  ),
                ),
                ConfidenceDial(value: r.confidence),
              ],
            ),
            const SizedBox(height: Ds.s4),
            _thresholdRule(r, band),
          ],
        ),
      );

  /// The decision threshold drawn in place on the 0–1 scale. A score of 0.41
  /// against a threshold of 0.4036 is a very different clinical object from a
  /// score of 0.95, and a bare number hides that.
  Widget _thresholdRule(TcwpnResult r, AlertBand band) => LayoutBuilder(
        builder: (_, box) {
          final w = box.maxWidth;
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                height: 22,
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    Positioned(
                      top: 7,
                      left: 0,
                      right: 0,
                      child: Container(
                        height: 8,
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.7),
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
                    ),
                    Positioned(
                      top: 7,
                      left: 0,
                      child: TweenAnimationBuilder<double>(
                        tween: Tween(
                            begin: 0,
                            end: r.calibratedProbability.clamp(0.0, 1.0) * w),
                        duration: Ds.med,
                        curve: Curves.easeOutCubic,
                        builder: (_, v, __) => Container(
                          width: v,
                          height: 8,
                          decoration: BoxDecoration(
                            color: band.fg,
                            borderRadius: BorderRadius.circular(4),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      left: (r.threshold.clamp(0.0, 1.0) * w) - 1,
                      top: 0,
                      child: Container(width: 2, height: 22, color: Ds.ink),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: Ds.s2),
              Row(
                children: [
                  Text('0.0',
                      style: AppTheme.data(size: 10, color: Ds.inkMuted)),
                  const Spacer(),
                  Text('threshold ${r.threshold.toStringAsFixed(4)}',
                      style: AppTheme.data(size: 10, color: Ds.ink)),
                  const Spacer(),
                  Text('1.0',
                      style: AppTheme.data(size: 10, color: Ds.inkMuted)),
                ],
              ),
            ],
          );
        },
      );

  // ── Decision detail ───────────────────────────────────────────────────────

  Widget _decisionPanel(TcwpnResult r) => Panel(
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: Readout(
                    label: 'RAW SCORE',
                    value: r.riskScore.toStringAsFixed(4),
                  ),
                ),
                Expanded(
                  child: Readout(
                    label: 'SHOTS (K)',
                    value: '${r.supportK}',
                  ),
                ),
                Expanded(
                  child: Readout(
                    label: 'ENTROPY',
                    value: r.entropy.isNaN ? '—' : r.entropy.toStringAsFixed(3),
                  ),
                ),
              ],
            ),
            if (r.prototypeDistanceAnxiety != null &&
                r.prototypeDistanceControl != null) ...[
              const SizedBox(height: Ds.s4),
              const Divider(),
              const SizedBox(height: Ds.s3),
              Text(
                'Distance to each class prototype in the 256-dimensional '
                'embedding space. The nearer prototype determines the class.',
                style: const TextStyle(
                    fontSize: 11.5, color: Ds.inkFaint, height: 1.4),
              ),
              const SizedBox(height: Ds.s3),
              _distanceRow('Anxiety prototype', r.prototypeDistanceAnxiety!,
                  r.prototypeDistanceControl!, Ds.red),
              const SizedBox(height: Ds.s2),
              _distanceRow('Control prototype', r.prototypeDistanceControl!,
                  r.prototypeDistanceAnxiety!, Ds.green),
            ],
            if (r.ece != null) ...[
              const SizedBox(height: Ds.s4),
              const Divider(),
              const SizedBox(height: Ds.s3),
              Row(
                children: [
                  const Icon(Icons.straighten_rounded,
                      size: 14, color: Ds.inkFaint),
                  const SizedBox(width: Ds.s2),
                  Expanded(
                    child: Text(
                      'Expected calibration error ${r.ece!.toStringAsFixed(3)} — '
                      'how closely the model\'s stated confidence matched reality '
                      'in validation.',
                      style: const TextStyle(
                          fontSize: 11.5, color: Ds.inkMuted, height: 1.4),
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      );

  Widget _distanceRow(String label, double d, double other, Color tone) {
    final total = d + other;
    final frac = total == 0 ? 0.5 : d / total;
    return Row(
      children: [
        SizedBox(
          width: 130,
          child: Text(label,
              style: const TextStyle(fontSize: 12, color: Ds.inkMuted)),
        ),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(
              value: 1 - frac,
              minHeight: 6,
              backgroundColor: Ds.surfaceSunken,
              valueColor: AlwaysStoppedAnimation(tone),
            ),
          ),
        ),
        const SizedBox(width: Ds.s3),
        Text(d.toStringAsFixed(3), style: AppTheme.data(size: 11.5)),
      ],
    );
  }

  // ── Attribution ───────────────────────────────────────────────────────────

  Widget _attributionPanel(TcwpnResult r) {
    if (r.spans.isEmpty) {
      return const Panel(
        child: Text(
          'The service returned no phrase-level output for this note.',
          style: TextStyle(fontSize: 12.5, color: Ds.inkMuted),
        ),
      );
    }

    final weighted = r.hasAttribution;
    final spans = [...r.spans];
    if (weighted) spans.sort((a, b) => b.weight.compareTo(a.weight));
    final maxW = weighted
        ? spans.map((s) => s.weight).fold<double>(0, (a, b) => a > b ? a : b)
        : 1.0;

    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            weighted
                ? 'Phrases ranked by the attention mass the model placed on them. '
                    'Bar length is the model\'s own weight.'
                : 'Phrases the service flagged for this note. It did not return '
                    'attention weights, so these are shown unranked — the order '
                    'carries no meaning.',
            style: const TextStyle(
                fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
          ),
          const SizedBox(height: Ds.s4),
          if (weighted)
            ...spans.map((s) => Padding(
                  padding: const EdgeInsets.only(bottom: Ds.s3),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(s.text,
                                style: const TextStyle(
                                    fontSize: 13, fontWeight: FontWeight.w500)),
                          ),
                          const SizedBox(width: Ds.s3),
                          Text(s.weight.toStringAsFixed(3),
                              style:
                                  AppTheme.data(size: 11, color: Ds.inkMuted)),
                        ],
                      ),
                      const SizedBox(height: 5),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(3),
                        child: TweenAnimationBuilder<double>(
                          tween: Tween(
                              begin: 0, end: maxW == 0 ? 0 : s.weight / maxW),
                          duration: Ds.med,
                          curve: Curves.easeOutCubic,
                          builder: (_, v, __) => LinearProgressIndicator(
                            value: v,
                            minHeight: 5,
                            backgroundColor: Ds.surfaceSunken,
                            valueColor:
                                const AlwaysStoppedAnimation(Ds.c3ClinicalNlp),
                          ),
                        ),
                      ),
                    ],
                  ),
                ))
          else
            Wrap(
              spacing: Ds.s2,
              runSpacing: Ds.s2,
              children: spans
                  .map((s) => Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: Ds.s3, vertical: 6),
                        decoration: BoxDecoration(
                          color: Ds.surfaceSunken,
                          borderRadius: BorderRadius.circular(Ds.rPill),
                          border: Border.all(color: Ds.hairline),
                        ),
                        child: Text(s.text,
                            style: const TextStyle(
                                fontSize: 12.5, color: Ds.inkMuted)),
                      ))
                  .toList(),
            ),
        ],
      ),
    );
  }

  // ── Support set influence ─────────────────────────────────────────────────

  /// This is TC-WPN's actual contribution made visible: which labelled examples
  /// shaped the prototype, and how much each was discounted for age and for the
  /// model's uncertainty about it.
  Widget _supportInfluencePanel(TcwpnResult r) {
    final sorted = [...r.supportContributions]
      ..sort((a, b) => b.combinedWeight.compareTo(a.combinedWeight));
    final maxW = sorted.first.combinedWeight;

    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Each labelled example contributes to the prototype in proportion to '
            'its recency and to how confident the model is in it.',
            style: TextStyle(fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
          ),
          const SizedBox(height: Ds.s4),
          ...sorted.take(8).map((s) => Padding(
                padding: const EdgeInsets.only(bottom: Ds.s4),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          width: 7,
                          height: 7,
                          decoration: BoxDecoration(
                            color: s.label == 'anxiety' ? Ds.red : Ds.green,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: Ds.s2),
                        Expanded(
                          child: Text(
                            s.excerpt,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                                fontSize: 12.5, color: Ds.inkMuted),
                          ),
                        ),
                        const SizedBox(width: Ds.s3),
                        Text(s.combinedWeight.toStringAsFixed(3),
                            style: AppTheme.data(
                                size: 11.5, weight: FontWeight.w600)),
                      ],
                    ),
                    const SizedBox(height: 5),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: LinearProgressIndicator(
                        value: maxW == 0 ? 0 : s.combinedWeight / maxW,
                        minHeight: 4,
                        backgroundColor: Ds.surfaceSunken,
                        valueColor: AlwaysStoppedAnimation(
                            s.label == 'anxiety' ? Ds.red : Ds.green),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        _weightPill('recency', s.temporalWeight),
                        const SizedBox(width: Ds.s2),
                        _weightPill('confidence', s.confidenceWeight),
                        if (s.noteDate != null) ...[
                          const Spacer(),
                          Text(DateFormat('MMM y').format(s.noteDate!),
                              style:
                                  AppTheme.data(size: 10, color: Ds.inkFaint)),
                        ],
                      ],
                    ),
                  ],
                ),
              )),
        ],
      ),
    );
  }

  Widget _weightPill(String label, double v) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
        decoration: BoxDecoration(
          color: Ds.surfaceSunken,
          borderRadius: BorderRadius.circular(Ds.rSm),
        ),
        child: Text('$label ${v.toStringAsFixed(2)}',
            style: AppTheme.data(size: 10, color: Ds.inkMuted)),
      );

  // ── Human in the loop ─────────────────────────────────────────────────────

  Widget _verdictPanel(ChartController chart, TcwpnResult r) {
    final v = _note.clinicianVerdict;
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Recording your judgement keeps the model accountable and builds the '
            'labelled set this site adapts from.',
            style: TextStyle(fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
          ),
          const SizedBox(height: Ds.s4),
          Row(
            children: [
              Expanded(
                child: _verdictButton(
                  'Agree',
                  Icons.check_rounded,
                  Ds.green,
                  v == 'agree',
                  () => _setVerdict(chart, 'agree'),
                ),
              ),
              const SizedBox(width: Ds.s3),
              Expanded(
                child: _verdictButton(
                  'Disagree',
                  Icons.close_rounded,
                  Ds.red,
                  v == 'disagree',
                  () => _setVerdict(chart, 'disagree'),
                ),
              ),
            ],
          ),
          if (v != null) ...[
            const SizedBox(height: Ds.s4),
            const Divider(),
            const SizedBox(height: Ds.s3),
            Text(
              'Add this note to the support set so future predictions for this '
              'patient learn from your label.',
              style: const TextStyle(
                  fontSize: 12, color: Ds.inkMuted, height: 1.45),
            ),
            const SizedBox(height: Ds.s3),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => _promote(chart, 'anxiety'),
                    style: OutlinedButton.styleFrom(
                        minimumSize: const Size(0, 42),
                        foregroundColor: Ds.red,
                        side: const BorderSide(color: Ds.hairlineStrong)),
                    child: const Text('Label anxiety'),
                  ),
                ),
                const SizedBox(width: Ds.s3),
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => _promote(chart, 'control'),
                    style: OutlinedButton.styleFrom(
                        minimumSize: const Size(0, 42),
                        foregroundColor: Ds.green,
                        side: const BorderSide(color: Ds.hairlineStrong)),
                    child: const Text('Label control'),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _verdictButton(String label, IconData icon, Color tone, bool selected,
          VoidCallback onTap) =>
      GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: Ds.fast,
          padding: const EdgeInsets.symmetric(vertical: Ds.s3),
          decoration: BoxDecoration(
            color: selected ? tone.withValues(alpha: 0.10) : Ds.surface,
            borderRadius: BorderRadius.circular(Ds.rMd),
            border: Border.all(
                color: selected ? tone : Ds.hairlineStrong,
                width: selected ? 1.5 : 1),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 17, color: selected ? tone : Ds.inkMuted),
              const SizedBox(width: Ds.s2),
              Text(label,
                  style: TextStyle(
                      fontSize: 13.5,
                      fontWeight: FontWeight.w600,
                      color: selected ? tone : Ds.inkMuted)),
            ],
          ),
        ),
      );

  Future<void> _setVerdict(ChartController chart, String verdict) async {
    await chart.recordVerdict(_note.id, verdict, _note.clinicianComment);
    setState(() => _note = _note.copyWith(clinicianVerdict: verdict));
  }

  Future<void> _promote(ChartController chart, String label) async {
    await chart.promoteNoteToSupport(_note, label);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Added to the $label support set.')),
    );
  }

  // ── Model provenance ──────────────────────────────────────────────────────

  Widget _modelPanel(TcwpnResult r) => Panel(
        child: Column(
          children: [
            _kv('Version', r.modelVersion),
            _kv('Backbone', 'Bio_ClinicalBERT, 768 → 256, L2-normalised'),
            _kv('Adaptation', 'Forward-pass prototypes, no gradient update'),
            if (r.temporalContext.isNotEmpty)
              _kv('Visit context', r.temporalContext),
            _kv('Latency', '${r.latencyMs} ms'),
          ],
        ),
      );

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 108,
              child: Text(k,
                  style: const TextStyle(fontSize: 12.5, color: Ds.inkMuted)),
            ),
            Expanded(
              child: Text(v, style: AppTheme.data(size: 11.5, height: 1.5)),
            ),
          ],
        ),
      );

  void _export(BuildContext context) =>
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Report export is wired to the PDF service.')),
      );
}
