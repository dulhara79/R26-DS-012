// lib/features/evidence/ask_care_screen.dart
//
// CARE-AnxRAG — Evidence-Aware Anxiety Assistant.
//
// Decision support, not diagnosis. The screen's contract with the clinician is
// that it shows what the evidence layer actually returned and nothing more: a
// grounded answer with its sources, or an explicit account of why no answer is
// being given. There is no path through this widget that renders text the RAG
// service did not produce.

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../core/design/tokens.dart';
import '../../data/api/gateways.dart';
import '../../domain/evidence.dart';

class AskCareScreen extends StatefulWidget {
  const AskCareScreen({super.key});

  @override
  State<AskCareScreen> createState() => _AskCareScreenState();
}

class _AskCareScreenState extends State<AskCareScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scroll = ScrollController();

  // The gateway is stateless over HTTP and takes no constructor arguments, so
  // this screen owns one rather than threading it down from the shell.
  final CentralBackendGateway _gateway = CentralBackendGateway();

  EvidenceResult? _result;
  bool _loading = false;

  /// The question that produced [_result]. Kept so the answer stays labelled
  /// with what was actually asked, even after the clinician edits the field.
  String? _askedQuestion;

  @override
  void dispose() {
    _controller.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _ask() async {
    final question = _controller.text.trim();
    if (question.isEmpty || _loading) return;

    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _result = null;
      _askedQuestion = question;
    });

    final result = await _gateway.askEvidence(question);
    if (!mounted) return;
    setState(() {
      _result = result;
      _loading = false;
    });
  }

  Future<void> _openUrl(String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null) return;
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Could not open the source link.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Ds.canvas,
      appBar: AppBar(
        backgroundColor: Ds.surface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        title: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'CARE-AnxRAG',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: Ds.ink,
              ),
            ),
            Text(
              'Evidence-Aware Anxiety Assistant',
              style: TextStyle(fontSize: 11, color: Ds.inkFaint),
            ),
          ],
        ),
      ),
      body: ListView(
        controller: _scroll,
        padding: const EdgeInsets.all(Ds.s4),
        children: [
          _QuestionBox(
            controller: _controller,
            loading: _loading,
            onAsk: _ask,
          ),
          if (_loading) ...[
            const SizedBox(height: Ds.s6),
            const _LoadingState(),
          ],
          if (_result != null) ...[
            const SizedBox(height: Ds.s5),
            _ResultView(
              result: _result!,
              question: _askedQuestion,
              onOpenUrl: _openUrl,
            ),
          ],
          if (_result == null && !_loading) ...[
            const SizedBox(height: Ds.s6),
            const _EmptyState(),
          ],
          const SizedBox(height: Ds.s8),
        ],
      ),
    );
  }
}

// ── Question input ───────────────────────────────────────────────────────────

class _QuestionBox extends StatelessWidget {
  final TextEditingController controller;
  final bool loading;
  final VoidCallback onAsk;

  const _QuestionBox({
    required this.controller,
    required this.loading,
    required this.onAsk,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        TextField(
          controller: controller,
          maxLines: 4,
          minLines: 2,
          enabled: !loading,
          textInputAction: TextInputAction.newline,
          style: const TextStyle(fontSize: 14, color: Ds.ink, height: 1.5),
          decoration: InputDecoration(
            hintText: 'Ask an evidence question…',
            hintStyle: const TextStyle(color: Ds.inkFaint, fontSize: 14),
            filled: true,
            fillColor: Ds.surface,
            contentPadding: const EdgeInsets.all(Ds.s3),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(Ds.rMd),
              borderSide: const BorderSide(color: Ds.hairlineStrong, width: 0.5),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(Ds.rMd),
              borderSide: const BorderSide(color: Ds.hairlineStrong, width: 0.5),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(Ds.rMd),
              borderSide: const BorderSide(color: Ds.brand, width: 1.5),
            ),
          ),
        ),
        const SizedBox(height: Ds.s3),
        Align(
          alignment: Alignment.centerRight,
          child: FilledButton.icon(
            onPressed: loading ? null : onAsk,
            style: FilledButton.styleFrom(
              backgroundColor: Ds.brand,
              padding: const EdgeInsets.symmetric(
                horizontal: Ds.s5,
                vertical: Ds.s3,
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(Ds.rSm),
              ),
            ),
            icon: const Icon(Icons.search_rounded, size: 18),
            label: const Text('Ask CARE'),
          ),
        ),
      ],
    );
  }
}

// ── Loading ──────────────────────────────────────────────────────────────────

class _LoadingState extends StatelessWidget {
  const _LoadingState();

  @override
  Widget build(BuildContext context) {
    return const Column(
      children: [
        SizedBox(
          width: 28,
          height: 28,
          child: CircularProgressIndicator(strokeWidth: 2.5, color: Ds.brand),
        ),
        SizedBox(height: Ds.s4),
        Text(
          'Retrieving and appraising evidence…',
          style: TextStyle(fontSize: 13, color: Ds.inkMuted),
        ),
        SizedBox(height: Ds.s1),
        // Sets expectations honestly: generation runs locally and a normal
        // answer has been measured near a minute. Without this the clinician
        // reasonably assumes the app has hung.
        Text(
          'Local generation can take up to a minute.',
          style: TextStyle(fontSize: 11, color: Ds.inkFaint),
        ),
      ],
    );
  }
}

// ── Empty ────────────────────────────────────────────────────────────────────

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Ds.s4),
      decoration: BoxDecoration(
        color: Ds.brandSoft,
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: Ds.brandEdge, width: 0.5),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Evidence retrieval, not diagnosis',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Ds.brandDeep,
            ),
          ),
          SizedBox(height: Ds.s2),
          Text(
            'Answers are grounded in retrieved sources and every claim is '
            'cited. Where the evidence base is insufficient, CARE-AnxRAG '
            'declines to answer rather than generating an unsupported one.',
            style: TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
          ),
        ],
      ),
    );
  }
}

// ── Result router ────────────────────────────────────────────────────────────

class _ResultView extends StatelessWidget {
  final EvidenceResult result;
  final String? question;
  final Future<void> Function(String) onOpenUrl;

  const _ResultView({
    required this.result,
    required this.question,
    required this.onOpenUrl,
  });

  @override
  Widget build(BuildContext context) {
    // One state, one branch. Every branch is terminal, so there is no way for
    // an abstention or a failure to fall through into the answer layout.
    switch (result.state) {
      case EvidenceState.crisisBypass:
        return _CrisisCard(result: result);
      case EvidenceState.unavailable:
        return _UnavailableCard(result: result);
      case EvidenceState.abstained:
        return _AbstainedCard(result: result);
      case EvidenceState.answered:
        return _AnsweredView(
          result: result,
          question: question,
          onOpenUrl: onOpenUrl,
        );
    }
  }
}

// ── State 1: crisis bypass ───────────────────────────────────────────────────

class _CrisisCard extends StatelessWidget {
  final EvidenceResult result;

  const _CrisisCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Ds.s4),
      decoration: BoxDecoration(
        color: Ds.darkRedSoft,
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: Ds.darkRed, width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Ds.darkRed, size: 20),
              SizedBox(width: Ds.s2),
              Expanded(
                child: Text(
                  'Crisis pre-screen matched',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Ds.darkRed,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: Ds.s3),
          Text(
            result.safetyMessage ??
                'Possible self-harm or suicide-related content detected. '
                    'Follow the ward\'s crisis protocol immediately.',
            style: const TextStyle(
              fontSize: 13,
              color: Ds.darkRed,
              height: 1.5,
            ),
          ),
          const SizedBox(height: Ds.s3),
          Container(
            padding: const EdgeInsets.all(Ds.s3),
            decoration: BoxDecoration(
              color: Ds.surface,
              borderRadius: BorderRadius.circular(Ds.rSm),
            ),
            child: const Text(
              'The evidence layer was not consulted for this question. '
              'Decision support does not respond in crisis situations.',
              style: TextStyle(fontSize: 12, color: Ds.inkMuted, height: 1.5),
            ),
          ),
        ],
      ),
    );
  }
}

// ── State 2: abstained ───────────────────────────────────────────────────────

class _AbstainedCard extends StatelessWidget {
  final EvidenceResult result;

  const _AbstainedCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Ds.s4),
      decoration: BoxDecoration(
        color: Ds.amberSoft,
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: Ds.amber, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.help_outline_rounded, color: Ds.amber, size: 20),
              SizedBox(width: Ds.s2),
              Expanded(
                child: Text(
                  'Insufficient evidence to answer',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Ds.amber,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: Ds.s3),
          const Text(
            'CARE-AnxRAG could not find sufficiently reliable evidence to '
            'provide a grounded answer, so it has not generated one.',
            style: TextStyle(fontSize: 13, color: Ds.ink, height: 1.5),
          ),
          if ((result.abstentionReason ?? '').isNotEmpty) ...[
            const SizedBox(height: Ds.s3),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(Ds.s3),
              decoration: BoxDecoration(
                color: Ds.surface,
                borderRadius: BorderRadius.circular(Ds.rSm),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'REASON',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      color: Ds.inkFaint,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: Ds.s1),
                  Text(
                    result.abstentionReason!,
                    style: const TextStyle(
                      fontSize: 12,
                      color: Ds.inkMuted,
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ── State 3: unavailable ─────────────────────────────────────────────────────

class _UnavailableCard extends StatelessWidget {
  final EvidenceResult result;

  const _UnavailableCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Ds.s4),
      decoration: BoxDecoration(
        color: Ds.greySoft,
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: Ds.hairlineStrong, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.cloud_off_rounded, color: Ds.grey, size: 20),
              SizedBox(width: Ds.s2),
              Expanded(
                child: Text(
                  'Decision support unavailable',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Ds.inkMuted,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: Ds.s3),
          const Text(
            'CARE-AnxRAG could not be reached. No answer is shown because '
            'none was produced.',
            style: TextStyle(fontSize: 13, color: Ds.ink, height: 1.5),
          ),
          if ((result.error ?? '').isNotEmpty) ...[
            const SizedBox(height: Ds.s3),
            Text(
              result.error!,
              style: const TextStyle(
                fontSize: 11,
                color: Ds.inkFaint,
                fontFamily: 'monospace',
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ── State 4: answered ────────────────────────────────────────────────────────

class _AnsweredView extends StatelessWidget {
  final EvidenceResult result;
  final String? question;
  final Future<void> Function(String) onOpenUrl;

  const _AnsweredView({
    required this.result,
    required this.question,
    required this.onOpenUrl,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if ((question ?? '').isNotEmpty) ...[
          Text(
            question!,
            style: const TextStyle(
              fontSize: 12,
              color: Ds.inkFaint,
              fontStyle: FontStyle.italic,
            ),
          ),
          const SizedBox(height: Ds.s3),
        ],
        const _SectionLabel('ANSWER'),
        const SizedBox(height: Ds.s2),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(Ds.s4),
          decoration: BoxDecoration(
            color: Ds.surface,
            borderRadius: BorderRadius.circular(Ds.rMd),
            border: Border.all(color: Ds.hairline, width: 0.5),
          ),
          child: Text(
            result.answer ?? '',
            style: const TextStyle(fontSize: 14, color: Ds.ink, height: 1.65),
          ),
        ),
        const SizedBox(height: Ds.s5),
        _MetricsRow(result: result),
        if (result.citations.isNotEmpty) ...[
          const SizedBox(height: Ds.s5),
          _SectionLabel('SOURCES  ·  ${result.citations.length}'),
          const SizedBox(height: Ds.s2),
          ...result.citations.map(
            (c) => Padding(
              padding: const EdgeInsets.only(bottom: Ds.s2),
              child: _CitationCard(citation: c, onOpenUrl: onOpenUrl),
            ),
          ),
        ],
        const SizedBox(height: Ds.s4),
        _ProvenanceFooter(result: result),
      ],
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;

  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: const TextStyle(
        fontSize: 10,
        fontWeight: FontWeight.w700,
        color: Ds.inkFaint,
        letterSpacing: 0.8,
      ),
    );
  }
}

// ── Confidence + conflict ────────────────────────────────────────────────────

class _MetricsRow extends StatelessWidget {
  final EvidenceResult result;

  const _MetricsRow({required this.result});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(child: _ConfidenceTile(result: result)),
            const SizedBox(width: Ds.s3),
            Expanded(child: _ConflictTile(result: result)),
          ],
        ),
        const SizedBox(height: Ds.s2),
        // Required disclaimer. The confidence value describes how good the
        // retrieved evidence was, and a clinician who reads it as "82% likely
        // to be true of this patient" has been actively misled.
        const Text(
          'Confidence reflects retrieval and evidence quality. It is not a '
          'diagnostic probability.',
          style: TextStyle(fontSize: 11, color: Ds.inkFaint, height: 1.4),
        ),
      ],
    );
  }
}

class _ConfidenceTile extends StatelessWidget {
  final EvidenceResult result;

  const _ConfidenceTile({required this.result});

  @override
  Widget build(BuildContext context) {
    final percent = result.confidencePercent;
    final fraction = (result.confidence ?? 0).clamp(0.0, 1.0);

    return Container(
      padding: const EdgeInsets.all(Ds.s3),
      decoration: BoxDecoration(
        color: Ds.greenSoft,
        borderRadius: BorderRadius.circular(Ds.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Evidence confidence',
            style: TextStyle(fontSize: 11, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s2),
          ClipRRect(
            borderRadius: BorderRadius.circular(Ds.rPill),
            child: LinearProgressIndicator(
              value: fraction,
              minHeight: 6,
              backgroundColor: Ds.hairline,
              valueColor: const AlwaysStoppedAnimation(Ds.green),
            ),
          ),
          const SizedBox(height: Ds.s2),
          Text(
            percent == null
                ? result.confidenceLabel
                : '${result.confidenceLabel}  ·  $percent%',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Ds.green,
            ),
          ),
        ],
      ),
    );
  }
}

class _ConflictTile extends StatelessWidget {
  final EvidenceResult result;

  const _ConflictTile({required this.result});

  @override
  Widget build(BuildContext context) {
    final conflicted = result.hasConflict;
    final tone = conflicted ? Ds.amber : Ds.green;
    final background = conflicted ? Ds.amberSoft : Ds.greenSoft;

    return Container(
      padding: const EdgeInsets.all(Ds.s3),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(Ds.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Evidence conflict',
            style: TextStyle(fontSize: 11, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s2),
          Row(
            children: [
              Icon(
                conflicted
                    ? Icons.warning_amber_rounded
                    : Icons.check_circle_outline_rounded,
                size: 16,
                color: tone,
              ),
              const SizedBox(width: Ds.s1),
              Expanded(
                child: Text(
                  conflicted ? result.conflictLabel : 'None detected',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: tone,
                  ),
                ),
              ),
            ],
          ),
          if (conflicted) ...[
            const SizedBox(height: Ds.s1),
            const Text(
              'Sources disagree — review both below.',
              style: TextStyle(fontSize: 11, color: Ds.inkMuted),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Citation card ────────────────────────────────────────────────────────────

class _CitationCard extends StatelessWidget {
  final EvidenceCitation citation;
  final Future<void> Function(String) onOpenUrl;

  const _CitationCard({required this.citation, required this.onOpenUrl});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Ds.surface,
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: Ds.hairline, width: 0.5),
      ),
      child: Theme(
        // The default divider on ExpansionTile fights the card border.
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(
            horizontal: Ds.s3,
            vertical: Ds.s1,
          ),
          childrenPadding: const EdgeInsets.fromLTRB(Ds.s3, 0, Ds.s3, Ds.s3),
          expandedCrossAxisAlignment: CrossAxisAlignment.start,
          leading: Container(
            width: 34,
            height: 34,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: Ds.brandSoft,
              borderRadius: BorderRadius.circular(Ds.rSm),
            ),
            child: Text(
              citation.citationId,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: Ds.brandDeep,
              ),
            ),
          ),
          title: Text(
            citation.sourceName ?? 'Unknown source',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Ds.ink,
            ),
          ),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Text(
              citation.evidenceLevelLabel,
              style: const TextStyle(fontSize: 11, color: Ds.inkFaint),
            ),
          ),
          children: [
            if ((citation.title ?? '').isNotEmpty) ...[
              Text(
                citation.title!,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Ds.ink,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: Ds.s2),
            ],
            if ((citation.excerpt ?? '').isNotEmpty) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(Ds.s3),
                decoration: BoxDecoration(
                  color: Ds.surfaceSunken,
                  borderRadius: BorderRadius.circular(Ds.rSm),
                ),
                child: Text(
                  citation.excerpt!,
                  style: const TextStyle(
                    fontSize: 12,
                    color: Ds.inkMuted,
                    height: 1.55,
                  ),
                ),
              ),
              const SizedBox(height: Ds.s2),
            ],
            if (citation.hasUrl)
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: () => onOpenUrl(citation.url!),
                  style: TextButton.styleFrom(
                    foregroundColor: Ds.brand,
                    padding: const EdgeInsets.symmetric(horizontal: Ds.s2),
                  ),
                  icon: const Icon(Icons.open_in_new_rounded, size: 15),
                  label: const Text(
                    'View original source',
                    style: TextStyle(fontSize: 12),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ── Provenance footer ────────────────────────────────────────────────────────

class _ProvenanceFooter extends StatelessWidget {
  final EvidenceResult result;

  const _ProvenanceFooter({required this.result});

  @override
  Widget build(BuildContext context) {
    final parts = <String>[];
    final sync = result.knowledgeBaseLastSyncAt;
    if (sync != null && sync.isNotEmpty) {
      // Trim the ISO timestamp to a date; the time-of-day of a knowledge-base
      // sync is noise to a clinician.
      parts.add('Knowledge base synced ${sync.split('T').first}');
    }
    if (result.latencyMs != null) {
      parts.add('${(result.latencyMs! / 1000).toStringAsFixed(1)}s');
    }
    if (parts.isEmpty) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.only(top: Ds.s2),
      child: Text(
        parts.join('  ·  '),
        style: const TextStyle(fontSize: 10, color: Ds.inkFaint),
      ),
    );
  }
}
