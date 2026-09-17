import 'package:flutter/material.dart';

import '../../core/design/components.dart';
import '../../core/design/tokens.dart';
import '../../data/repositories/p5c_repositories.dart';
import '../../domain/evidence.dart';
import '../../state/supporting_evidence_controller.dart';

class SupportingEvidenceScreen extends StatefulWidget {
  final SupportingEvidenceController? controller;
  final String? subjectId;
  final String? displayId;

  const SupportingEvidenceScreen({
    super.key,
    required this.controller,
    this.displayId,
  }) : subjectId = null;

  const SupportingEvidenceScreen.production({
    super.key,
    required this.subjectId,
    this.displayId,
  }) : controller = null;

  @override
  State<SupportingEvidenceScreen> createState() =>
      _SupportingEvidenceScreenState();
}

class _SupportingEvidenceScreenState extends State<SupportingEvidenceScreen> {
  late final SupportingEvidenceController _controller;
  CentralBackendEvidenceRepository? _ownedRepository;
  final _question = TextEditingController();

  @override
  void initState() {
    super.initState();
    if (widget.controller != null) {
      _controller = widget.controller!;
    } else {
      final repository = CentralBackendEvidenceRepository();
      _ownedRepository = repository;
      _controller = SupportingEvidenceController(
        subjectId: widget.subjectId!,
        repository: repository,
      );
    }
  }

  @override
  void dispose() {
    _question.dispose();
    if (widget.controller == null) {
      _controller.dispose();
      _ownedRepository?.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) => Scaffold(
        appBar: AppBar(title: const Text('Supporting Evidence')),
        body: ListView(
          padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
          children: [
            if ((widget.displayId ?? '').isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: Ds.s3),
                child: Text(
                  widget.displayId!,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                  ),
                ),
              ),
            const Panel(
              child: Text(
                'General evidence support is accessed in this patient review context. '
                'The patient fusion score and clinical notes are not sent to CARE-AnxRAG. '
                'This evidence is decision support, not a patient-specific assessment.',
                style: TextStyle(fontSize: 12.5, height: 1.45, color: Ds.inkMuted),
              ),
            ),
            const SizedBox(height: Ds.s5),
            const SectionLabel('Evidence question'),
            TextField(
              controller: _question,
              minLines: 2,
              maxLines: 5,
              decoration: const InputDecoration(
                hintText: 'Ask an evidence question…',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: Ds.s3),
            FilledButton.icon(
              onPressed: _controller.loading
                  ? null
                  : () => _controller.ask(_question.text),
              icon: _controller.loading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.search_rounded),
              label: Text(
                _controller.loading ? 'Checking evidence…' : 'Ask CARE-AnxRAG',
              ),
            ),
            if (_controller.error != null) ...[
              const SizedBox(height: Ds.s5),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Supporting evidence unavailable',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: Ds.s2),
                    Text(
                      _controller.error!,
                      style: const TextStyle(color: Ds.inkMuted),
                    ),
                  ],
                ),
              ),
            ],
            if (_controller.result != null) ...[
              const SizedBox(height: Ds.s5),
              _EvidenceResultView(result: _controller.result!),
            ],
          ],
        ),
      ),
    );
  }
}

class _EvidenceResultView extends StatelessWidget {
  final EvidenceResult result;

  const _EvidenceResultView({required this.result});

  @override
  Widget build(BuildContext context) {
    return switch (result.state) {
      EvidenceState.crisisBypass => _statePanel(
          'Central Backend safety pre-screen',
          result.safetyMessage ??
              'The Central Backend safety pre-screen bypassed evidence retrieval.',
        ),
      EvidenceState.abstained => _statePanel(
          'CARE-AnxRAG abstained',
          result.abstentionReason ??
              'The evidence service did not return an answer.',
        ),
      EvidenceState.unavailable => _statePanel(
          'Supporting evidence unavailable',
          result.error ?? 'The evidence service is currently unavailable.',
        ),
      EvidenceState.answered => _answered(),
    };
  }

  Widget _statePanel(String title, String message) => Panel(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: Ds.s2),
            Text(
              message,
              style: const TextStyle(color: Ds.inkMuted, height: 1.4),
            ),
          ],
        ),
      );

  Widget _answered() => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SectionLabel('Evidence response'),
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.hasAnswer
                      ? result.answer!
                      : 'Supporting evidence unavailable',
                  style: const TextStyle(height: 1.5),
                ),
                if (result.confidence != null || result.conflictScore != null) ...[
                  const SizedBox(height: Ds.s3),
                  if (result.confidence != null)
                    Text(
                      'Evidence retrieval confidence: ${result.confidence!.toStringAsFixed(2)}',
                      style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
                    ),
                  if (result.conflictScore != null)
                    Text(
                      'Retrieved-source conflict: ${result.conflictScore!.toStringAsFixed(2)}',
                      style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
                    ),
                ],
                if ((result.knowledgeBaseLastSyncAt ?? '').isNotEmpty) ...[
                  const SizedBox(height: Ds.s2),
                  Text(
                    'Knowledge base last synced: ${result.knowledgeBaseLastSyncAt}',
                    style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                  ),
                ],
              ],
            ),
          ),
          if (result.citations.isNotEmpty) ...[
            const SizedBox(height: Ds.s5),
            const SectionLabel('Citations'),
            for (final citation in result.citations)
              Padding(
                padding: const EdgeInsets.only(bottom: Ds.s3),
                child: _CitationCard(citation: citation),
              ),
          ],
        ],
      );
}

class _CitationCard extends StatelessWidget {
  final EvidenceCitation citation;

  const _CitationCard({required this.citation});

  @override
  Widget build(BuildContext context) => Panel(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              (citation.title ?? '').isEmpty
                  ? 'Source ${citation.citationId}'
                  : citation.title!,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
            if ((citation.sourceName ?? '').isNotEmpty) ...[
              const SizedBox(height: Ds.s1),
              Text(
                citation.sourceName!,
                style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
              ),
            ],
            if ((citation.excerpt ?? '').isNotEmpty) ...[
              const SizedBox(height: Ds.s2),
              Text(citation.excerpt!, style: const TextStyle(height: 1.4)),
            ],
            if ((citation.evidenceLevel ?? '').isNotEmpty) ...[
              const SizedBox(height: Ds.s2),
              Text(
                'Evidence level: ${citation.evidenceLevel}',
                style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
              ),
            ],
          ],
        ),
      );
}
