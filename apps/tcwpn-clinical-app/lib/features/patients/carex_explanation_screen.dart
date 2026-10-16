import 'package:flutter/material.dart';

import '../../core/design/components.dart';
import '../../core/design/tokens.dart';
import '../../data/api/api_client.dart';
import '../../data/repositories/carex_explanation_repository.dart';

/// Explains exactly one clinician-visible fusion result via the Central
/// Backend. This never derives risk, requests a new prediction, or calls RAG.
class CarexExplanationScreen extends StatefulWidget {
  final String subjectId;
  final int? fusionResultId;
  final String? displayId;
  final CarexExplanationRepository? repository;

  const CarexExplanationScreen({
    super.key,
    required this.subjectId,
    required this.fusionResultId,
    this.displayId,
    this.repository,
  });

  @override
  State<CarexExplanationScreen> createState() => _CarexExplanationScreenState();
}

class _CarexExplanationScreenState extends State<CarexExplanationScreen> {
  late final CarexExplanationRepository _repository;
  late Future<Map<String, dynamic>> _request;

  @override
  void initState() {
    super.initState();
    _repository =
        widget.repository ?? CentralBackendCarexExplanationRepository();
    _request = _fetch();
  }

  Future<Map<String, dynamic>> _fetch() async {
    if (widget.fusionResultId == null) {
      throw StateError('No fusion result is recorded for this assessment.');
    }
    final result = await _repository.explanation(widget.subjectId);
    // This endpoint returns the LATEST result. A new ingestion can occur while
    // the doctor is reviewing an older assessment. Never misattribute the
    // newer explanation to the older score: require an exact identity match.
    if (result['subject_id'] != widget.subjectId ||
        result['fusion_result_id'] is! num ||
        (result['fusion_result_id'] as num).toInt() != widget.fusionResultId) {
      throw StateError(
        'The assessment changed on the server. Return to the patient and '
        'refresh their assessment before viewing CARE-X.',
      );
    }
    return result;
  }

  void _retry() => setState(() => _request = _fetch());

  @override
  void dispose() {
    if (widget.repository == null &&
        _repository is CentralBackendCarexExplanationRepository) {
      _repository.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('CARE-X · Why this assessment?')),
        body: FutureBuilder<Map<String, dynamic>>(
          future: _request,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              final error = snapshot.error;
              return _Notice(
                title: 'CARE-X explanation unavailable',
                message: error is ApiException
                    ? error.message
                    : error.toString().replaceFirst('Bad state: ', ''),
                onRetry: _retry,
              );
            }
            final result = snapshot.data;
            if (result == null) {
              return _Notice(
                title: 'CARE-X explanation unavailable',
                message: 'The Central Backend returned no explanation.',
                onRetry: _retry,
              );
            }
            return _details(result);
          },
        ),
      );

  Widget _details(Map<String, dynamic> result) {
    final attribution = _map(result['attribution']);
    final provenance = _map(result['weight_provenance']);
    final counterfactuals = _map(result['counterfactuals']);
    final reliability = _map(result['reliability_audit']);
    final inflation = _map(result['weight_inflation']);
    final sufficiency = _map(result['sufficiency']);
    final ledger = _map(result['honesty_ledger']);

    return ListView(
      padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
      children: [
        Text(
          widget.displayId ?? widget.subjectId,
          style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: Ds.s1),
        Text(
          'Fusion result #${widget.fusionResultId} · '
          '${_value(result['explainer_version'])}',
          style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
        ),
        const SizedBox(height: Ds.s4),
        const InlineNotice(
          icon: Icons.info_outline,
          text: 'CARE-X explains the recorded fusion calculation. It is not '
              'CARE-AnxRAG evidence retrieval and does not establish a '
              'clinically calibrated probability or treatment decision.',
        ),
        const SizedBox(height: Ds.s4),
        if (result['assessed'] != true) ...[
          _Section(
            title: 'No scored assessment',
            body: _value(result['refusal_reason'] ?? result['narrative']),
          ),
        ] else ...[
          _Section(
            title: 'Explanation summary',
            body: _value(result['narrative'] ?? result['headline']),
          ),
          const SizedBox(height: Ds.s4),
          const SectionLabel('L1 · Attribution'),
          ..._rows(attribution['ranked']).map(
            (row) => _Section(
              title: _value(row['label'] ?? row['modality']),
              body: 'Contribution: ${_value(row['contribution'])}; '
                  'share of composite: ${_percentage(row['share_pct'])}.',
            ),
          ),
          if (_rows(attribution['ranked']).isEmpty)
            const _Section(title: 'Not reported', body: 'No attribution rows.'),
          const SizedBox(height: Ds.s4),
          const SectionLabel('L2 · Weight provenance'),
          ..._rows(provenance['per_modality']).map(
            (row) => _Section(
              title: _value(row['label'] ?? row['modality']),
              body: 'Stored weight: ${_value(row['final_weight'])}; '
                  'reported confidence: ${_value(row['reading_confidence'])}; '
                  'coverage: ${_value(row['coverage'])}.'
                  '${_rowsAsLines(row['drivers'])}',
            ),
          ),
          if (provenance['renormalisation_note'] != null)
            _Section(
              title: 'Missing-modality adjustment',
              body: _value(provenance['renormalisation_note']),
            ),
          const SizedBox(height: Ds.s4),
          const SectionLabel('L3 · Counterfactuals'),
          const Text(
            'These are mathematical score scenarios, not predicted patient '
            'responses or recommended interventions.',
            style: TextStyle(fontSize: 12, color: Ds.inkMuted),
          ),
          const SizedBox(height: Ds.s2),
          ..._rows(counterfactuals['drop_one']).map(
            (row) => _Section(
              title: 'Without ${_value(row['label'] ?? row['modality'])}',
              body: row['composite_without'] == null
                  ? _value(row['note'])
                  : 'Reweighted score: ${_value(row['composite_without'])}; '
                      'change: ${_value(row['delta'])}; '
                      'tier: ${_value(row['tier_without'])}.',
            ),
          ),
          ..._rows(counterfactuals['tier_flip_points']).map(
            (row) => _Section(
              title: '${_value(row['label'])} → ${_value(row['target_tier'])}',
              body: row['reachable'] == true
                  ? 'Threshold reachable at harmonised score '
                      '${_value(row['harmonised_value_required'])}, '
                      'holding other scores and the usable set fixed.'
                  : 'Not reachable by changing this harmonised score alone '
                      'with other scores and eligibility held fixed.',
            ),
          ),
          if (counterfactuals['thresholds_authoritative'] == false)
            const _Section(
              title: 'Threshold warning',
              body: 'The explainer could not confirm deployed tier boundaries.',
            ),
          const SizedBox(height: Ds.s4),
          const SectionLabel('L4 · Reliability & sufficiency'),
          if (inflation['quantified'] != true)
            _Section(
              title: 'Weight inflation not quantified',
              body: _value(inflation['unavailable_reason']),
            ),
          ..._rows(inflation['per_modality']).map(
            (row) => _Section(
              title: 'Weight audit: ${_value(row['label'])}',
              body: _value(row['interpretation']),
            ),
          ),
          ..._rows(reliability['findings']).map(
            (row) => _Section(
              title: _value(row['label'] ?? row['modality']),
              body: '${_value(row['verdict'])}: '
                  '${_value(row['interpretation'])}',
            ),
          ),
          ..._rows(sufficiency['per_modality']).map(
            (row) => _Section(
              title: 'Decision relevance: ${_value(row['label'])}',
              body: 'Necessary: ${_value(row['necessary'])}; '
                  'alone sufficient: ${_value(row['alone_sufficient'])}; '
                  'decision relevant: ${_value(row['decision_relevant'])}.',
            ),
          ),
          _Section(
            title: 'Minimum sufficient modalities',
            body: sufficiency['minimal_sufficient_size'] == null
                ? 'Not determined'
                : '${_value(sufficiency['minimal_sufficient_size'])} '
                    'for the current fixed usable set. '
                    '${_value(sufficiency['scope_caveat'])}',
          ),
        ],
        const SizedBox(height: Ds.s4),
        const SectionLabel('L5 · Limitations and uncertainty'),
        if (_rows(ledger['items']).isEmpty)
          const _Section(
            title: 'No limitations reported',
            body: 'This does not establish clinical validation.',
          ),
        ..._rows(ledger['items']).map(
          (row) => _Section(
            title: '${_value(row['severity'])} · ${_value(row['issue'])}',
            body: '${_value(row['detail'])}\n'
                'Affects: ${_value(row['affects'])}\n'
                'Resolution: ${_value(row['resolved_by'])}',
          ),
        ),
        const SizedBox(height: Ds.s4),
        const DecisionSupportNotice(),
      ],
    );
  }
}

Map<String, dynamic> _map(Object? value) =>
    value is Map ? Map<String, dynamic>.from(value) : <String, dynamic>{};

List<Map<String, dynamic>> _rows(Object? value) => value is List
    ? value.whereType<Map>().map((row) => Map<String, dynamic>.from(row)).toList()
    : const [];

String _value(Object? value) =>
    value == null || value.toString().trim().isEmpty
        ? 'Not reported'
        : value.toString();

String _percentage(Object? value) =>
    value is num ? '${value.toStringAsFixed(1)}%' : 'Not reported';

String _rowsAsLines(Object? value) =>
    value is List && value.isNotEmpty ? '\n${value.join('; ')}' : '';

class _Section extends StatelessWidget {
  final String title;
  final String body;

  const _Section({required this.title, required this.body});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: Ds.s2),
              Text(body, style: const TextStyle(fontSize: 12.5, height: 1.45)),
            ],
          ),
        ),
      );
}

class _Notice extends StatelessWidget {
  final String title;
  final String message;
  final VoidCallback onRetry;

  const _Notice({
    required this.title,
    required this.message,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(Ds.s5),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: Ds.s3),
              Text(message, textAlign: TextAlign.center),
              const SizedBox(height: Ds.s3),
              FilledButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ),
        ),
      );
}
