import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/repositories/central_backend_repositories.dart';
import '../../domain/contracts/attention_event.dart';
import '../../domain/contracts/contract_enums.dart';
import '../../state/async_data_state.dart';
import '../../state/attention_event_detail_controller.dart';
import '../patients/patient_overview_screen.dart';

class AttentionEventDetailScreen extends StatefulWidget {
  final AttentionEventDetailController? controller;
  final String? eventId;
  final ValueChanged<String>? onOpenPatient;

  const AttentionEventDetailScreen({
    super.key,
    required this.controller,
    this.onOpenPatient,
  }) : eventId = null;

  const AttentionEventDetailScreen.production({
    super.key,
    required this.eventId,
    this.onOpenPatient,
  }) : controller = null;

  @override
  State<AttentionEventDetailScreen> createState() =>
      _AttentionEventDetailScreenState();
}

class _AttentionEventDetailScreenState
    extends State<AttentionEventDetailScreen> {
  late final AttentionEventDetailController _controller;
  late final bool _ownsController;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ??
        AttentionEventDetailController(
          eventId: widget.eventId!,
          repository: CentralBackendAttentionEventRepository(),
          authRepository: CentralBackendAuthRepository(),
        );
    if (_ownsController) {
      _controller.load();
    }
  }

  @override
  void dispose() {
    if (_ownsController) {
      _controller.dispose();
    }
    super.dispose();
  }

  void _openPatient(BuildContext context, String subjectId) {
    final callback = widget.onOpenPatient;
    if (callback != null) {
      callback(subjectId);
      return;
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => PatientOverviewScreen.production(subjectId: subjectId),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        final state = _controller.state;
        return Scaffold(
          appBar: AppBar(title: const Text('Attention event')),
          body: _body(context, state),
        );
      },
    );
  }

  Widget _body(
    BuildContext context,
    AsyncDataState<AttentionEvent> state,
  ) {
    if (state.status == AsyncDataStatus.loading) {
      return const Center(child: CircularProgressIndicator());
    }

    final event = state.data;
    if (event == null) {
      final title = switch (state.status) {
        AsyncDataStatus.sessionExpired => 'Session expired',
        AsyncDataStatus.forbidden => 'Event access unavailable',
        AsyncDataStatus.offline => 'Event unavailable offline',
        AsyncDataStatus.conflict => 'Event changed on the server',
        _ => 'Attention event unavailable',
      };
      return EmptyState(
        icon: state.status == AsyncDataStatus.offline
            ? Icons.cloud_off_rounded
            : Icons.notification_important_outlined,
        title: title,
        body: state.message ??
            'The authoritative server attention event could not be loaded.',
        actionLabel: state.status == AsyncDataStatus.sessionExpired ? null : 'Retry',
        onAction: state.status == AsyncDataStatus.sessionExpired
            ? null
            : () => _controller.load(),
      );
    }

    final mutationAllowed = state.status == AsyncDataStatus.data ||
        state.status == AsyncDataStatus.partial;

    return RefreshIndicator(
      onRefresh: () => _controller.load(showLoading: false),
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          if (state.status == AsyncDataStatus.offline ||
              state.status == AsyncDataStatus.forbidden ||
              state.status == AsyncDataStatus.partial) ...[
            InlineNotice(
              icon: state.status == AsyncDataStatus.offline
                  ? Icons.cloud_off_rounded
                  : Icons.info_outline_rounded,
              text: state.message ??
                  'The last canonical server event state is shown. No local lifecycle transition was created.',
            ),
            const SizedBox(height: Ds.s4),
          ],
          if (_controller.mutationMessage != null) ...[
            InlineNotice(
              icon: Icons.sync_problem_rounded,
              text: _controller.mutationMessage!,
            ),
            const SizedBox(height: Ds.s4),
          ],
          _StatusPanel(event: event),
          const SizedBox(height: Ds.s4),
          _ProvenancePanel(event: event),
          const SizedBox(height: Ds.s4),
          OutlinedButton.icon(
            onPressed: () => _openPatient(context, event.subjectId),
            icon: const Icon(Icons.person_search_outlined),
            label: const Text('View patient'),
          ),
          const SizedBox(height: Ds.s3),
          if (mutationAllowed && event.status == AttentionEventStatus.open)
            OutlinedButton(
              onPressed: _controller.isMutating ? null : _controller.acknowledge,
              child: const Text('Acknowledge'),
            ),
          if (mutationAllowed &&
              event.status == AttentionEventStatus.acknowledged)
            OutlinedButton(
              onPressed: _controller.isMutating ? null : _controller.resolve,
              child: const Text('Resolve'),
            ),
          const SizedBox(height: Ds.s5),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }
}

class _StatusPanel extends StatelessWidget {
  final AttentionEvent event;

  const _StatusPanel({required this.event});

  @override
  Widget build(BuildContext context) {
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  event.id,
                  style: AppTheme.display(size: 18),
                ),
              ),
              Text(
                _statusLabel(event.status),
                style: AppTheme.data(
                  size: 11.5,
                  weight: FontWeight.w700,
                  color: Ds.inkMuted,
                ),
              ),
            ],
          ),
          const SizedBox(height: Ds.s3),
          Text(event.subjectId, style: const TextStyle(color: Ds.inkMuted)),
          if ((event.reason ?? '').trim().isNotEmpty) ...[
            const SizedBox(height: Ds.s3),
            Text(event.reason!, style: const TextStyle(height: 1.4)),
          ],
          if (event.forecastHorizon != null) ...[
            const SizedBox(height: Ds.s3),
            Text(
              '${event.forecastHorizon}-minute forecast horizon',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: Ds.s1),
            const Text(
              'Potential escalation was flagged within the server-provided forecast horizon.',
              style: TextStyle(fontSize: 12.5, color: Ds.inkMuted, height: 1.4),
            ),
          ],
          if (event.acknowledgedBy != null) ...[
            const SizedBox(height: Ds.s3),
            Text(
              'Acknowledged by ${event.acknowledgedBy}'
              '${event.acknowledgedAt == null ? '' : ' · ${_time(event.acknowledgedAt!)}'}',
              style: const TextStyle(color: Ds.inkMuted),
            ),
          ],
          if (event.resolvedBy != null) ...[
            const SizedBox(height: Ds.s2),
            Text(
              'Resolved by ${event.resolvedBy}'
              '${event.resolvedAt == null ? '' : ' · ${_time(event.resolvedAt!)}'}',
              style: const TextStyle(color: Ds.inkMuted),
            ),
          ],
        ],
      ),
    );
  }
}

class _ProvenancePanel extends StatelessWidget {
  final AttentionEvent event;

  const _ProvenancePanel({required this.event});

  @override
  Widget build(BuildContext context) {
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SectionLabel('Server provenance'),
          _Field('Event id', event.id),
          _Field('Subject id', event.subjectId),
          _Field('Event type', event.eventType ?? 'Unavailable'),
          _Field('Severity', event.severity.name),
          _Field(
            'Fusion result',
            event.fusionResultId == null
                ? 'Unavailable'
                : 'Fusion result #${event.fusionResultId}',
          ),
          _Field('Forecast result', event.forecastResultId ?? 'Unavailable'),
          _Field('Policy version', event.policyVersion ?? 'Unavailable'),
          _Field(
            'Created',
            event.createdAt == null ? 'Unavailable' : _time(event.createdAt!),
          ),
        ],
      ),
    );
  }
}

class _Field extends StatelessWidget {
  final String label;
  final String value;

  const _Field(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: Ds.s2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 110,
            child: Text(
              label,
              style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontSize: 12.5, color: Ds.ink),
            ),
          ),
        ],
      ),
    );
  }
}

String _statusLabel(AttentionEventStatus status) => switch (status) {
      AttentionEventStatus.open => 'OPEN',
      AttentionEventStatus.acknowledged => 'ACKNOWLEDGED',
      AttentionEventStatus.resolved => 'RESOLVED',
      AttentionEventStatus.unknown => 'UNKNOWN',
    };

String _time(DateTime value) =>
    DateFormat('d MMM y, HH:mm').format(value.toLocal());
