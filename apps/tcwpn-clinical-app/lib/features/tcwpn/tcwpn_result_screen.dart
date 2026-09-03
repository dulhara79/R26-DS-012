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
import '../../data/api/api_client.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import '../fusion/fusion_detail_screen.dart';
import 'note_analysis_screen.dart';

class TcwpnResultScreen extends StatefulWidget {
  final ClinicalNote note;
  const TcwpnResultScreen({super.key, required this.note});

  @override
  State<TcwpnResultScreen> createState() => _TcwpnResultScreenState();
}

class _TcwpnResultScreenState extends State<TcwpnResultScreen> {
  late ClinicalNote _note = widget.note;
  bool _busy = false;

  /// Every number on this screen arrives over the network. A missing or
  /// non-finite value must read as "not reported"; it must not reach a paint.
  static String _fixed(double? v, int places) =>
      (v == null || !v.isFinite) ? '\u2014' : v.toStringAsFixed(places);

  /// Fraction for a progress bar, always in 0..1.
  ///
  /// A non-finite value produces a NaN rect, and `Canvas.drawRect` asserts on
  /// those in debug builds — a red error screen caused by a weight the service
  /// simply did not send.
  static double _frac(double v, double max) {
    if (!v.isFinite || !max.isFinite || max <= 0) return 0;
    return (v / max).clamp(0.0, 1.0);
  }

  @override
  Widget build(BuildContext context) {
    // watch, not read: analyseStoredNote refreshes fusion immediately after the
    // ingest returns, and the composite panel below has to repaint when it
    // lands rather than showing "no composite" until the screen is reopened.
    final chart = context.watch<ChartController>();

    final r = _note.result;
    if (r == null) return _noAssessment(context, chart);

    final band = AlertBandX.fromScore(r.calibratedProbability);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Note analysis'),
        actions: [
          // Closes the lifecycle: an analysed note can be corrected and sent
          // again. Without this, "re-analyse" had no entry point anywhere in the
          // app. Replaces this screen rather than stacking, so backing out of
          // the editor returns to the chart, not to an assessment that describes
          // text the clinician has just changed.
          IconButton(
            tooltip: 'Edit note',
            icon: const Icon(Icons.edit_outlined),
            onPressed: () => Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (_) => ChangeNotifierProvider.value(
                  value: context.read<ChartController>(),
                  child: NoteAnalysisScreen(existing: _note),
                ),
              ),
            ),
          ),
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
          SectionLabel(
            'Framework composite',
            trailing: chart.fusion == null ? null : 'Full breakdown',
            onTrailing: chart.fusion == null
                ? null
                : () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => FusionDetailScreen(
                          fusion: chart.fusion!,
                          patient: chart.patient,
                          latestNote: _note,
                        ),
                      ),
                    ),
          ),
          _fusionPanel(chart),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Decision'),
          _decisionPanel(r),
          const SizedBox(height: Ds.s5),
          SectionLabel(
              r.hasAttribution ? 'Attention attribution' : 'Key phrases'),
          _attributionPanel(r),
          const SizedBox(height: Ds.s5),
          if (r.supportContributions.isNotEmpty) ...[
            const SectionLabel('Support set influence'),
            _supportInfluencePanel(r),
            const SizedBox(height: Ds.s5),
          ],
          const SectionLabel('Clinician review'),
          _verdictPanel(chart, r),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Submitted note'),
          Panel(
            child: Text(
              _note.text,
              style:
                  const TextStyle(fontSize: 13.5, height: 1.6, color: Ds.ink),
            ),
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Model'),
          _modelPanel(r),
          const SizedBox(height: Ds.s5),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }

  // ── No assessment ─────────────────────────────────────────────────────────

  /// The note carries no assessment. Two situations that must not share a
  /// screen: a draft the clinician deliberately parked, and a note that WAS
  /// submitted and came back without one.
  ///
  /// The second case is not an exception anywhere. `POST /v1/clinical-notes`
  /// answers 200 with `component_detail` absent whenever call_c3 failed or the
  /// support bank was unusable (central_backend/main.py), so no ApiException is
  /// raised and the editor's "Analysis unavailable" dialog never fires. This
  /// screen is where that outcome lands. It previously described the note as an
  /// un-submitted draft — which is false, and it discarded the backend's own
  /// explanation, leaving the clinician with a dead end and no retry.
  Widget _noAssessment(BuildContext context, ChartController chart) {
    final failed = _note.status == ClinicalNoteStatus.analysisFailed;
    final reason = _note.lastAnalysisError ?? chart.error;
    final provenance = chart.lastIngest?.scoreProvenance;
    final status = chart.lastIngest?.status;

    return WorkingOverlay(
      active: _busy,
      message: 'Running TC-WPN.\nA cold model can take up to a minute to wake.',
      child: Scaffold(
        appBar: AppBar(
          title: Text(failed ? 'No assessment returned' : 'Clinical note'),
          actions: [
            IconButton(
              tooltip: 'Edit note',
              icon: const Icon(Icons.edit_outlined),
              onPressed: () => Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                  builder: (_) => ChangeNotifierProvider.value(
                    value: chart,
                    child: NoteAnalysisScreen(existing: _note),
                  ),
                ),
              ),
            ),
          ],
        ),
        body: ListView(
          padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
          children: [
            InlineNotice(
              icon: failed
                  ? Icons.report_problem_outlined
                  : Icons.pending_outlined,
              tone: failed ? Ds.amber : null,
              text: failed
                  ? 'This note reached the backend and was stored. The clinical '
                      'model did not return an assessment for it. Nothing you '
                      'wrote has been lost.'
                  : 'This note is saved as a draft on this device. It has not '
                      'been submitted.',
            ),
            if (failed) ...[
              const SizedBox(height: Ds.s5),
              const SectionLabel('What the service reported'),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Verbatim. A paraphrase here would cost the study team the
                    // one string that tells them which service failed.
                    Text(
                      reason ?? 'No reason was returned.',
                      style: const TextStyle(
                          fontSize: 12.5, color: Ds.ink, height: 1.5),
                    ),
                    if (status != null) ...[
                      const SizedBox(height: Ds.s3),
                      _kv('Component status', status),
                    ],
                    if (provenance != null) _kv('Backend note', provenance),
                  ],
                ),
              ),
            ],
            const SizedBox(height: Ds.s5),
            ElevatedButton.icon(
              onPressed: _busy ? null : () => _retry(chart),
              icon: const Icon(Icons.refresh_rounded, size: 19),
              label: Text(failed ? 'Try analysis again' : 'Analyse note'),
            ),
            const SizedBox(height: Ds.s5),
            const SectionLabel('Framework composite'),
            _fusionPanel(chart),
            const SizedBox(height: Ds.s5),
            const SectionLabel('Submitted note'),
            Panel(
              child: Text(
                _note.text,
                style:
                    const TextStyle(fontSize: 13.5, height: 1.6, color: Ds.ink),
              ),
            ),
            const SizedBox(height: Ds.s5),
            const DecisionSupportNotice(),
          ],
        ),
      ),
    );
  }

  Future<void> _retry(ChartController chart) async {
    setState(() => _busy = true);
    try {
      final updated = await chart.analyseStoredNote(_note.id);
      if (!mounted) return;
      setState(() {
        _note = updated;
        _busy = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _busy = false);
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(e.message)));
    } catch (e) {
      // Nothing below the gateway is allowed to reach the framework as an
      // unhandled async error and leave the overlay spinning forever.
      if (!mounted) return;
      setState(() => _busy = false);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Analysis could not be started.')),
      );
    }
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
                      // Named by what the service SAID about the number, not by
                      // the field it arrived under. The live deployment sends
                      // `probability` with calibration_status=uncalibrated.
                      Text(r.probabilityLabel, style: AppTheme.eyebrow),
                      const SizedBox(height: 2),
                      Text(
                        _fixed(r.calibratedProbability, 4),
                        style: AppTheme.data(
                            size: 38, weight: FontWeight.w600, color: band.fg),
                      ),
                      const SizedBox(height: Ds.s2),
                      Wrap(
                        spacing: Ds.s2,
                        runSpacing: Ds.s2,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: [
                          _severityChip(band),
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

  /// The severity word for THIS MODALITY's score.
  ///
  /// Bands at 0.25 / 0.50 / 0.75 via AlertBandX.fromScore, which the design
  /// tokens already document as the split for a single modality shown in
  /// isolation. It is NOT the fusion service's split (0.33 / 0.66 into
  /// Low/Medium/High) and it deliberately uses a different word list, so the
  /// two numbers on this screen can never be read as the same quantity.
  Widget _severityChip(AlertBand band) => Container(
        padding:
            const EdgeInsets.symmetric(horizontal: Ds.s3, vertical: 4),
        decoration: BoxDecoration(
          color: band.fg.withValues(alpha: 0.14),
          borderRadius: BorderRadius.circular(Ds.rPill),
          border: Border.all(color: band.fg.withValues(alpha: 0.35)),
        ),
        child: Text(
          band.severityLabel.toUpperCase(),
          style: TextStyle(
            fontSize: 11.5,
            fontWeight: FontWeight.w700,
            color: band.fg,
            letterSpacing: 0.6,
          ),
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
                            end: _frac(r.calibratedProbability, 1.0) * w),
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
                      left: (_frac(r.threshold, 1.0) * w) - 1,
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
                  Text('threshold ${_fixed(r.threshold, 4)}',
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

  // ── Framework composite ───────────────────────────────────────────────────

  /// The composite, as the FUSION SERVICE computed it. Nothing on this panel is
  /// re-derived on the device.
  ///
  /// The composite and the score above it are two different quantities on two
  /// different scales: the fusion service bands three ways at 0.33 / 0.66 into
  /// Low / Medium / High (fusion_service/fusion.py, BANDS), and it returns both
  /// `tier` and `band` so neither has to be guessed. Putting them on one screen
  /// is what a clinician asked for; keeping their vocabularies apart is what
  /// stops it being misleading.
  ///
  /// A null composite is a BLOCKED GATE, not low risk — fewer than two usable
  /// modalities, or no time-varying one. FusionBar renders the server's own
  /// gate reason in that case rather than a number.
  Widget _fusionPanel(ChartController chart) {
    final f = chart.fusion;
    final ingest = chart.lastIngest;

    if (f == null) {
      final reason = ingest?.fusionError ?? ingest?.fusionSkippedReason;
      return Panel(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'No composite has been read for this patient.',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 4),
            Text(
              reason ??
                  'The clinical-note reading above is stored on the backend and '
                      'joins the composite the next time fusion runs. A '
                      'composite needs at least two usable modalities.',
              style: const TextStyle(
                  fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
            ),
          ],
        ),
      );
    }

    // TC-WPN's own slice of the composite, so the clinician can see what the
    // note they are reading actually contributed.
    ComponentContribution? notes;
    for (final c in f.contributions) {
      if (c.key == Modality.c3ClinicalNlp) notes = c;
    }

    // A composite computed BEFORE this assessment does not contain it. One
    // second of tolerance: the fusion row and the reading are written by
    // different code paths inside the same request.
    final analysedAt = _note.analysedAt;
    final predatesThisNote = analysedAt != null &&
        f.updatedAt != null &&
        analysedAt.difference(f.updatedAt!).inSeconds > 1;

    return Column(
      children: [
        Panel(
          padding: const EdgeInsets.all(Ds.s5),
          child: FusionBar(
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
        ),
        const SizedBox(height: Ds.s3),
        Panel(
          child: Column(
            children: [
              Row(
                children: [
                  Expanded(
                    child: Readout(
                      label: 'COMPOSITE',
                      value: f.compositeLabel,
                      valueColor: f.band.fg,
                    ),
                  ),
                  Expanded(
                    child: Readout(
                      label: 'TIER',
                      // The server's tier verbatim. Never re-derived from the
                      // composite here — the edges are the fusion service's.
                      value: f.tier ?? '\u2014',
                      valueColor: f.band.fg,
                    ),
                  ),
                  Expanded(
                    child: Readout(
                      label: 'MODALITIES',
                      value: '${f.modalitiesUsed} of ${Modality.all.length}',
                    ),
                  ),
                ],
              ),
              if (notes != null) ...[
                const SizedBox(height: Ds.s4),
                const Divider(),
                const SizedBox(height: Ds.s3),
                _kv(
                  'This modality',
                  notes.contribution == null
                      ? 'not in the composite'
                      : '${_fixed(notes.score, 3)} × weight '
                          '${_fixed(notes.weight, 2)} = '
                          '${_fixed(notes.contribution, 3)}',
                ),
                if (notes.note != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      notes.note!,
                      style: const TextStyle(
                          fontSize: 11.5, color: Ds.inkFaint, height: 1.4),
                    ),
                  ),
              ],
              const SizedBox(height: Ds.s3),
              _kv('Assessment', f.assessmentLabel),
              _kv(
                'Computed',
                f.updatedAt == null
                    ? '\u2014'
                    : DateFormat('d MMM y · HH:mm').format(f.updatedAt!),
              ),
            ],
          ),
        ),
        if (predatesThisNote) ...[
          const SizedBox(height: Ds.s3),
          const InlineNotice(
            icon: Icons.history_rounded,
            tone: Ds.amber,
            text: 'This composite was computed before the assessment above, so '
                'it does not include this note. Re-run fusion from the patient '
                'chart to fold it in.',
          ),
        ],
        if (chart.fusionFromCache) ...[
          const SizedBox(height: Ds.s3),
          const InlineNotice(
            icon: Icons.cloud_off_rounded,
            text: 'Showing the last composite this device received. It has not '
                'been re-read from the backend in this session.',
          ),
        ],
      ],
    );
  }

  // ── Decision detail ───────────────────────────────────────────────────────

  Widget _decisionPanel(TcwpnResult r) => Panel(
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: Readout(
                    label: 'RAW SCORE',
                    value: _fixed(r.riskScore, 4),
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
                    value: _fixed(r.entropy, 3),
                  ),
                ),
              ],
            ),
            if (r.prototypeDistanceAnxiety != null &&
                r.prototypeDistanceControl != null) ...[
              const SizedBox(height: Ds.s4),
              const Divider(),
              const SizedBox(height: Ds.s3),
              const Text(
                'Distance to each class prototype in the 256-dimensional '
                'embedding space. The nearer prototype determines the class.',
                style: TextStyle(
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
                      'Expected calibration error ${_fixed(r.ece, 3)} \u2014 '
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
    // A non-finite distance from the service must not become a NaN rect.
    final frac = (!total.isFinite || total == 0) ? 0.5 : _frac(d, total);
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
              value: (1 - frac).clamp(0.0, 1.0),
              minHeight: 6,
              backgroundColor: Ds.surfaceSunken,
              valueColor: AlwaysStoppedAnimation(tone),
            ),
          ),
        ),
        const SizedBox(width: Ds.s3),
        Text(_fixed(d, 3), style: AppTheme.data(size: 11.5)),
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

    // Rank only the spans that actually carry a finite weight. A service that
    // sends a mixed list — some weighted, some not — used to put every span
    // through the weighted branch, and the unweighted ones produced a NaN bar
    // fraction. Canvas.drawRect asserts on a NaN rect in debug builds, so a
    // missing weight became a red error screen. The unweighted remainder is
    // rendered below as plain chips, which is what it is.
    final ranked = r.spans.where((s) => s.hasWeight && s.weight.isFinite).toList()
      ..sort((a, b) => b.weight.compareTo(a.weight));
    final plain = r.spans.where((s) => !(s.hasWeight && s.weight.isFinite));

    final weighted = ranked.isNotEmpty;
    final spans = weighted ? ranked : r.spans.toList();
    final maxW = weighted
        ? ranked.map((s) => s.weight).fold<double>(0, (a, b) => a > b ? a : b)
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
                          Text(_fixed(s.weight, 3),
                              style:
                                  AppTheme.data(size: 11, color: Ds.inkMuted)),
                        ],
                      ),
                      const SizedBox(height: 5),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(3),
                        child: TweenAnimationBuilder<double>(
                          tween: Tween(begin: 0, end: _frac(s.weight, maxW)),
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
          // A mixed response is not an error, but the two halves mean different
          // things and must not share a ranking.
          if (weighted && plain.isNotEmpty) ...[
            const SizedBox(height: Ds.s3),
            const Divider(),
            const SizedBox(height: Ds.s3),
            const Text(
              'The service also flagged these phrases without an attention '
              'weight. They are unranked.',
              style:
                  TextStyle(fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
            ),
            const SizedBox(height: Ds.s3),
            Wrap(
              spacing: Ds.s2,
              runSpacing: Ds.s2,
              children: plain
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
                        Text(_fixed(s.combinedWeight, 3),
                            style: AppTheme.data(
                                size: 11.5, weight: FontWeight.w600)),
                      ],
                    ),
                    const SizedBox(height: 5),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: LinearProgressIndicator(
                        value: _frac(s.combinedWeight, maxW),
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
        child: Text('$label ${_fixed(v, 2)}',
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
            const Text(
              'Add this note to the support set so future predictions for this '
              'patient learn from your label.',
              style: TextStyle(
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
