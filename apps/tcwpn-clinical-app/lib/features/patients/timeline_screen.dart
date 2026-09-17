import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../data/repositories/p5b_repositories.dart';
import '../../domain/contracts/timeline_entry.dart';
import '../../state/async_data_state.dart';
import '../../state/timeline_controller.dart';

class TimelineScreen extends StatefulWidget {
  final TimelineController? controller;
  final String? subjectId;
  final String? displayId;

  const TimelineScreen({
    super.key,
    required this.controller,
    this.displayId,
  }) : subjectId = null;

  const TimelineScreen.production({
    super.key,
    required this.subjectId,
    this.displayId,
  }) : controller = null;

  @override
  State<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  late final TimelineController _controller;
  late final bool _ownsController;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ??
        TimelineController(
          subjectId: widget.subjectId!,
          repository: CentralBackendTimelineRepository(),
        );
    _controller.addListener(_changed);
    if (_ownsController) _controller.load();
  }

  void _changed() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _controller.removeListener(_changed);
    if (_ownsController) _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = _controller.state;
    return Scaffold(
      appBar: AppBar(title: Text(widget.displayId == null ? 'Timeline' : '${widget.displayId} · Timeline')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SegmentedButton<TimelineWindow>(
              segments: const [
                ButtonSegment(value: TimelineWindow.hours24, label: Text('24 hours')),
                ButtonSegment(value: TimelineWindow.days7, label: Text('7 days')),
              ],
              selected: {_controller.window},
              onSelectionChanged: (value) => _controller.setWindow(value.first),
            ),
            const SizedBox(height: 12),
            const Text(
              'Gaps mean no assessment record was returned. They are not low-risk values and are not interpolated.',
              style: TextStyle(fontSize: 12),
            ),
            const SizedBox(height: 12),
            Expanded(child: _body(state)),
          ],
        ),
      ),
    );
  }

  Widget _body(AsyncDataState<List<TimelineEntry>> state) {
    if (state.status == AsyncDataStatus.loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (state.status == AsyncDataStatus.empty) {
      return const Center(child: Text('No assessment records were returned for this window.'));
    }
    if (state.status != AsyncDataStatus.data) {
      return Center(child: Text(state.message ?? 'Timeline history unavailable.'));
    }
    final rows = _controller.visibleEntries;
    if (rows.isEmpty) {
      return const Center(child: Text('No assessment records were returned for this window.'));
    }
    return ListView.separated(
      itemCount: rows.length,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (_, index) => _TimelineRow(entry: rows[index]),
    );
  }
}

class _TimelineRow extends StatelessWidget {
  final TimelineEntry entry;
  const _TimelineRow({required this.entry});

  @override
  Widget build(BuildContext context) {
    final score = entry.composite == null ? '—' : entry.composite!.toStringAsFixed(2);
    final tier = entry.tier ?? '—';
    final band = entry.band ?? '—';
    final status = _human(entry.assessmentStatus) ?? 'Status not reported';
    final time = entry.computedAt == null
        ? 'Time not reported'
        : DateFormat('d MMM yyyy, HH:mm').format(entry.computedAt!);
    final trigger = entry.trigger == null ? 'Trigger not reported' : 'Trigger: ${entry.trigger}';
    final missing = entry.missingModalities.isEmpty
        ? null
        : 'Missing: ${entry.missingModalities.join(', ')}';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('$tier · $score · $band', style: const TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text(status),
            Text(time),
            Text(trigger),
            if (missing != null) Text(missing),
          ],
        ),
      ),
    );
  }

  static String? _human(String? value) {
    final trimmed = value?.trim();
    if (trimmed == null || trimmed.isEmpty) return null;
    final spaced = trimmed.replaceAll('_', ' ');
    return '${spaced[0].toUpperCase()}${spaced.substring(1)}';
  }
}
