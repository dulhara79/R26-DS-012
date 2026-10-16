import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/repositories/central_backend_repositories.dart';
import '../../domain/contracts/attention_event.dart';
import '../../domain/contracts/contract_enums.dart';
import '../../state/async_data_state.dart';
import '../../state/attention_events_controller.dart';
import 'attention_event_detail_screen.dart';

class ActivityScreen extends StatefulWidget {
  final AttentionEventsController? controller;
  final ValueChanged<AttentionEvent>? onOpenEvent;

  const ActivityScreen({
    super.key,
    required this.controller,
    this.onOpenEvent,
  });

  const ActivityScreen.production({
    super.key,
    this.onOpenEvent,
  }) : controller = null;

  @override
  State<ActivityScreen> createState() => _ActivityScreenState();
}

class _ActivityScreenState extends State<ActivityScreen> {
  late final AttentionEventsController _controller;
  late final bool _ownsController;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ??
        AttentionEventsController(
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

  void _openEvent(BuildContext context, AttentionEvent event) {
    final callback = widget.onOpenEvent;
    if (callback != null) {
      callback(event);
      return;
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => AttentionEventDetailScreen.production(eventId: event.id),
      ),
    ).then((_) => _controller.load(showLoading: false));
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        final state = _controller.state;
        return Scaffold(
          appBar: AppBar(title: const Text('Activity')),
          body: _body(context, state),
        );
      },
    );
  }

  Widget _body(
    BuildContext context,
    AsyncDataState<List<AttentionEvent>> state,
  ) {
    if (state.status == AsyncDataStatus.loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (state.status == AsyncDataStatus.empty) {
      return const EmptyState(
        icon: Icons.history_rounded,
        title: 'No server attention-event activity',
        body:
            'No persistent attention events were returned for the current clinician scope.',
      );
    }

    final events = state.data;
    if (events == null) {
      final title = switch (state.status) {
        AsyncDataStatus.sessionExpired => 'Session expired',
        AsyncDataStatus.forbidden => 'Activity access unavailable',
        AsyncDataStatus.offline => 'Activity unavailable offline',
        AsyncDataStatus.conflict => 'Activity changed on the server',
        _ => 'Activity unavailable',
      };
      return EmptyState(
        icon: state.status == AsyncDataStatus.offline
            ? Icons.cloud_off_rounded
            : Icons.notifications_active_outlined,
        title: title,
        body: state.message ??
            'The authoritative server attention-event activity is not available.',
        actionLabel: state.status == AsyncDataStatus.sessionExpired ? null : 'Retry',
        onAction: state.status == AsyncDataStatus.sessionExpired
            ? null
            : () => _controller.load(),
      );
    }

    final open = <AttentionEvent>[];
    final acknowledged = <AttentionEvent>[];
    final resolved = <AttentionEvent>[];
    final unknown = <AttentionEvent>[];
    for (final event in events) {
      switch (event.status) {
        case AttentionEventStatus.open:
          open.add(event);
        case AttentionEventStatus.acknowledged:
          acknowledged.add(event);
        case AttentionEventStatus.resolved:
          resolved.add(event);
        case AttentionEventStatus.unknown:
          unknown.add(event);
      }
    }

    return RefreshIndicator(
      onRefresh: () => _controller.load(showLoading: false),
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          if (state.status == AsyncDataStatus.offline) ...[
            InlineNotice(
              icon: Icons.cloud_off_rounded,
              text: state.message ?? 'Offline. No event state was changed locally.',
            ),
            const SizedBox(height: Ds.s4),
          ],
          _EventSection(
            label: 'OPEN',
            events: open,
            onOpen: (event) => _openEvent(context, event),
          ),
          _EventSection(
            label: 'ACKNOWLEDGED',
            events: acknowledged,
            onOpen: (event) => _openEvent(context, event),
          ),
          _EventSection(
            label: 'RESOLVED',
            events: resolved,
            onOpen: (event) => _openEvent(context, event),
          ),
          if (unknown.isNotEmpty)
            _EventSection(
              label: 'UNKNOWN',
              events: unknown,
              onOpen: (event) => _openEvent(context, event),
            ),
          const SizedBox(height: Ds.s4),
          const DecisionSupportNotice(),
        ],
      ),
    );
  }
}

class _EventSection extends StatelessWidget {
  final String label;
  final List<AttentionEvent> events;
  final ValueChanged<AttentionEvent> onOpen;

  const _EventSection({
    required this.label,
    required this.events,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    if (events.isEmpty) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.only(bottom: Ds.s5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SectionLabel(label),
          for (final event in events)
            Padding(
              padding: const EdgeInsets.only(bottom: Ds.s3),
              child: _EventRow(
                event: event,
                onTap: () => onOpen(event),
              ),
            ),
        ],
      ),
    );
  }
}

class _EventRow extends StatelessWidget {
  final AttentionEvent event;
  final VoidCallback onTap;

  const _EventRow({required this.event, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Panel(
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  event.id,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    color: Ds.ink,
                  ),
                ),
              ),
              Text(
                _statusLabel(event.status),
                style: AppTheme.data(size: 11, color: Ds.inkMuted),
              ),
            ],
          ),
          const SizedBox(height: Ds.s2),
          Text(
            event.subjectId,
            style: const TextStyle(fontSize: 12.5, color: Ds.inkMuted),
          ),
          if ((event.reason ?? '').trim().isNotEmpty) ...[
            const SizedBox(height: Ds.s2),
            Text(event.reason!, style: const TextStyle(fontSize: 13)),
          ],
          if (event.forecastHorizon != null) ...[
            const SizedBox(height: Ds.s2),
            Text(
              '${event.forecastHorizon}-minute forecast horizon',
              style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
            ),
          ],
          if (event.createdAt != null) ...[
            const SizedBox(height: Ds.s2),
            Text(
              'Created ${_time(event.createdAt!)}',
              style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
            ),
          ],
          if (event.acknowledgedBy != null) ...[
            const SizedBox(height: Ds.s2),
            Text(
              'Acknowledged by ${event.acknowledgedBy}'
              '${event.acknowledgedAt == null ? '' : ' · ${_time(event.acknowledgedAt!)}'}',
              style: const TextStyle(fontSize: 11.5, color: Ds.inkMuted),
            ),
          ],
          if (event.resolvedBy != null) ...[
            const SizedBox(height: Ds.s1),
            Text(
              'Resolved by ${event.resolvedBy}'
              '${event.resolvedAt == null ? '' : ' · ${_time(event.resolvedAt!)}'}',
              style: const TextStyle(fontSize: 11.5, color: Ds.inkMuted),
            ),
          ],
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
