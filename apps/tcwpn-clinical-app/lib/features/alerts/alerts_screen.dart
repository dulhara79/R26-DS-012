// lib/features/alerts/alerts_screen.dart
//
// The escalation queue. Every RED and DARK RED composite raises an alert here,
// and an alert is only cleared when a named clinician acknowledges it — the app
// never claims someone has been notified, only that a notice was raised and
// whether anyone has picked it up.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/local/stores.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import '../shell.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();
    final open = roster.alerts.where((a) => !a.acknowledged).toList();
    final closed = roster.alerts.where((a) => a.acknowledged).toList();

    return Scaffold(
      appBar: AppBar(title: const Text('Alerts')),
      body: roster.alerts.isEmpty
          ? const EmptyState(
              icon: Icons.notifications_none_rounded,
              title: 'No alerts',
              body:
                  'Alerts appear here when a patient\'s composite risk reaches the '
                  'RED or DARK RED band.',
            )
          : ListView(
              padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
              children: [
                if (open.isNotEmpty) ...[
                  const SectionLabel('Awaiting acknowledgement'),
                  ...open.map((a) => _AlertRow(alert: a)),
                  const SizedBox(height: Ds.s5),
                ],
                if (closed.isNotEmpty) ...[
                  const SectionLabel('Acknowledged'),
                  ...closed.take(30).map((a) => _AlertRow(alert: a)),
                ],
              ],
            ),
    );
  }
}

class _AlertRow extends StatelessWidget {
  final ClinicalAlert alert;
  const _AlertRow({required this.alert});

  @override
  Widget build(BuildContext context) {
    final roster = context.read<RosterController>();
    final band = alert.band;

    return Padding(
      padding: const EdgeInsets.only(bottom: Ds.s3),
      child: Panel(
        padding: const EdgeInsets.all(Ds.s4),
        borderColor: alert.acknowledged
            ? Ds.hairline
            : (band?.fg.withValues(alpha: 0.32) ?? Ds.hairlineStrong),
        onTap: alert.patientMrn == null
            ? null
            : () {
                final p = roster.patients
                    .where((x) => x.mrn == alert.patientMrn)
                    .firstOrNull;
                if (p != null) openChart(context, p);
              },
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (band != null) BandChip(band: band, showProtocolName: true),
                const Spacer(),
                Text(_relative(alert.raisedAt),
                    style: const TextStyle(fontSize: 11, color: Ds.inkFaint)),
              ],
            ),
            const SizedBox(height: Ds.s3),
            Text(alert.title,
                style: const TextStyle(
                    fontSize: 14, fontWeight: FontWeight.w600)),
            const SizedBox(height: 3),
            Text(alert.body,
                style: const TextStyle(
                    fontSize: 12.5, color: Ds.inkMuted, height: 1.45)),
            const SizedBox(height: Ds.s3),
            if (alert.acknowledged)
              Row(
                children: [
                  const Icon(Icons.check_circle_outline_rounded,
                      size: 14, color: Ds.green),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      'Acknowledged by ${alert.acknowledgedBy ?? "unknown"}'
                      '${alert.acknowledgedAt == null ? "" : " · ${DateFormat('d MMM, HH:mm').format(alert.acknowledgedAt!)}"}',
                      style: AppTheme.data(size: 10.5, color: Ds.inkFaint),
                    ),
                  ),
                ],
              )
            else
              SizedBox(
                height: 38,
                child: OutlinedButton(
                  onPressed: () async {
                    final who = await SecureStore.clinicianId() ?? 'unknown';
                    await roster.acknowledge(alert.id, who);
                  },
                  style: OutlinedButton.styleFrom(minimumSize: const Size(0, 38)),
                  child: const Text('Acknowledge'),
                ),
              ),
          ],
        ),
      ),
    );
  }

  String _relative(DateTime t) {
    final d = DateTime.now().difference(t);
    if (d.inMinutes < 1) return 'just now';
    if (d.inMinutes < 60) return '${d.inMinutes}m ago';
    if (d.inHours < 24) return '${d.inHours}h ago';
    if (d.inDays < 7) return '${d.inDays}d ago';
    return DateFormat('d MMM').format(t);
  }
}
