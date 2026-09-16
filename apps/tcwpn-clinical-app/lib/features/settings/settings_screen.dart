// lib/features/settings/settings_screen.dart
//
// Settings reports configuration and security state without implying that a
// service, certificate pin, or model value exists when it has not been verified.

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/config/env.dart';
import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../core/security/pinned_certificates.dart';
import '../../core/security/secure_http.dart';
import '../../data/api/session.dart';
import '../../data/local/stores.dart';
import '../../state/controllers.dart';
import '../auth/login_screen.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();
    final info = roster.modelInfo;

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          FutureBuilder<String?>(
            future: SecureStore.clinicianName(),
            builder: (_, snap) => Panel(
              padding: const EdgeInsets.all(Ds.s5),
              child: Row(
                children: [
                  Container(
                    width: 46,
                    height: 46,
                    decoration: BoxDecoration(
                      color: Ds.brandSoft,
                      borderRadius: BorderRadius.circular(Ds.rMd),
                    ),
                    alignment: Alignment.center,
                    child: const Icon(Icons.person_rounded,
                        color: Ds.brand, size: 22),
                  ),
                  const SizedBox(width: Ds.s4),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(snap.data ?? 'Signed in',
                            style: const TextStyle(
                                fontSize: 15, fontWeight: FontWeight.w600)),
                        FutureBuilder<String?>(
                          future: SecureStore.clinicianId(),
                          builder: (_, s) => Text(s.data ?? '',
                              style:
                                  AppTheme.data(size: 11, color: Ds.inkFaint)),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Services ClinAnx uses'),
          Panel(
            child: Column(
              children: [
                _ServiceRow(
                  name: 'Central Backend',
                  url: Env.backendBase,
                  reachable: roster.backendReachable,
                  accent: Ds.brand,
                  role: 'Enrolment, note ingestion, gate, fusion, timeline',
                ),
                const Divider(height: Ds.s5),
                _ServiceRow(
                  name: 'Clinical NLP (TC-WPN)',
                  url: Env.tcwpnBase,
                  reachable: info != null,
                  accent: Ds.c3ClinicalNlp,
                  role: 'Orchestrated by the Central Backend · Clinical NLP signal',
                ),
              ],
            ),
          ),
          const SizedBox(height: Ds.s3),
          const InlineNotice(
            icon: Icons.devices_rounded,
            text:
                'The wearable, behavioural and intake components are collected '
                'by the patient-facing app and sent to the Central Backend. '
                'ClinAnx never contacts them directly. Their values reach this '
                'app only through backend clinician views keyed by subject_id.',
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Model information'),
          Panel(
            child: info == null
                ? const Text(
                    'Waiting for the Clinical NLP service to report. Values shown '
                    'here come from the service health response; unavailable '
                    'metadata is not filled with local defaults.',
                    style: TextStyle(
                        fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
                  )
                : Column(
                    children: info.entries
                        .where((e) => e.value is! Map && e.value is! List)
                        .map((e) => Padding(
                              padding: const EdgeInsets.symmetric(vertical: 5),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  SizedBox(
                                    width: 130,
                                    child: Text(
                                      e.key.replaceAll('_', ' '),
                                      style: const TextStyle(
                                          fontSize: 12.5, color: Ds.inkMuted),
                                    ),
                                  ),
                                  Expanded(
                                    child: Text('${e.value}',
                                        style: AppTheme.data(size: 11.5)),
                                  ),
                                ],
                              ),
                            ))
                        .toList(),
                  ),
          ),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Connection security'),
          const _PinningPanel(),
          const SizedBox(height: Ds.s5),
          const SectionLabel('Data governance'),
          const Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _GovRow(Icons.lock_outline_rounded, 'Credentials',
                    'Stored in the device keychain, encrypted at rest.'),
                Divider(height: Ds.s5),
                _GovRow(Icons.folder_outlined, 'Clinical records',
                    'Local caches are isolated by clinician and patient. Signing out prevents another clinician account from reading the previous clinician\'s cache.'),
                Divider(height: Ds.s5),
                _GovRow(Icons.cloud_upload_outlined, 'Note text',
                    'Sent to the Central Backend, which orchestrates Clinical NLP / TC-WPN analysis. De-identify before submitting.'),
              ],
            ),
          ),
          const SizedBox(height: Ds.s5),
          OutlinedButton.icon(
            onPressed: () => _signOut(context),
            icon: const Icon(Icons.logout_rounded, size: 18),
            label: const Text('Sign out'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Ds.red,
              side: const BorderSide(color: Ds.hairlineStrong),
            ),
          ),
          const SizedBox(height: Ds.s4),
          Center(
            child: Text('R26-DS-012 · SLIIT Faculty of Computing',
                style: AppTheme.data(size: 10.5, color: Ds.inkFaint)),
          ),
        ],
      ),
    );
  }

  Future<void> _signOut(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Sign out?'),
        content: const Text(
          'Your session ends. This clinician\'s local cache remains isolated and '
          'will be available only when this clinician signs in again.',
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Sign out', style: TextStyle(color: Ds.red)),
          ),
        ],
      ),
    );
    if (ok != true || !context.mounted) return;
    await Session.signOut();
    if (!context.mounted) return;
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (_) => false,
    );
  }
}

class _ServiceRow extends StatelessWidget {
  final String name;
  final String url;
  final bool? reachable;
  final Color accent;
  final String role;

  const _ServiceRow({
    required this.name,
    required this.url,
    required this.reachable,
    required this.accent,
    required this.role,
  });

  @override
  Widget build(BuildContext context) {
    final configured = url.isNotEmpty;
    final (dot, status) = switch ((configured, reachable)) {
      (false, _) => (Ds.hairlineStrong, 'Not configured'),
      (true, true) => (Ds.green, 'Reachable'),
      (true, false) => (Ds.red, 'Not responding'),
      _ => (Ds.inkFaint, 'Configured'),
    };

    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(color: dot, shape: BoxShape.circle),
        ),
        const SizedBox(width: Ds.s3),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(name,
                  style: const TextStyle(
                      fontSize: 13, fontWeight: FontWeight.w600)),
              Text(role,
                  style: const TextStyle(fontSize: 11.5, color: Ds.inkMuted)),
              Text(
                configured ? Uri.tryParse(url)?.host ?? url : status,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: AppTheme.data(size: 10.5, color: Ds.inkFaint),
              ),
            ],
          ),
        ),
        Text(status,
            style: TextStyle(
                fontSize: 11, color: dot == Ds.green ? Ds.green : Ds.inkFaint)),
      ],
    );
  }
}

class _GovRow extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;
  const _GovRow(this.icon, this.title, this.body);

  @override
  Widget build(BuildContext context) => Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: Ds.inkMuted),
          const SizedBox(width: Ds.s3),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w600)),
                const SizedBox(height: 2),
                Text(body,
                    style: const TextStyle(
                        fontSize: 12, color: Ds.inkMuted, height: 1.45)),
              ],
            ),
          ),
        ],
      );
}

class _PinningPanel extends StatefulWidget {
  const _PinningPanel();
  @override
  State<_PinningPanel> createState() => _PinningPanelState();
}

class _PinningPanelState extends State<_PinningPanel> {
  bool _checking = false;
  List<PinReport> _reports = SecureHttp.reports;

  Future<void> _recheck() async {
    setState(() => _checking = true);
    final r = await SecureHttp.verifyAll();
    if (!mounted) return;
    setState(() {
      _reports = r;
      _checking = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final configured = SecureHttp.isPinnedHost0Configured;
    final allOk = _reports.isNotEmpty && _reports.every((r) => r.isHealthy);

    return Column(
      children: [
        if (kIsWeb)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.gpp_maybe_outlined,
              tone: Ds.amber,
              text: 'This is a web build. Certificate pinning is a '
                  'native-only capability, so the browser performs TLS '
                  'validation instead. Use the Android or iOS build with '
                  'patient data.',
            ),
          )
        else if (kPinningDisabled)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.gpp_bad_outlined,
              tone: Ds.red,
              text: 'Certificate pinning is DISABLED by a build flag. Explicit '
                  'pinned hosts fall back to platform TLS in this development '
                  'configuration. Do not use it with patient data.',
            ),
          )
        else if (!configured)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.gpp_maybe_outlined,
              tone: Ds.red,
              text: 'No certificate pin set is configured. HTTPS hosts use '
                  'platform TLS validation; generate reviewed pins before '
                  'claiming that a host is pinned.',
            ),
          )
        else if (SecureHttp.needsReview)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.update_outlined,
              tone: Ds.amber,
              text: 'The configured pin set is past its review date '
                  '($kPinsReviewBy). Regenerate it before relying on the pins.',
            ),
          ),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    allOk ? Icons.verified_user_rounded : Icons.shield_outlined,
                    size: 17,
                    color: allOk ? Ds.green : Ds.inkMuted,
                  ),
                  const SizedBox(width: Ds.s3),
                  Expanded(
                    child: Text(
                      configured && !kPinningDisabled && !kIsWeb
                          ? 'Pinned TLS is active only for hosts listed below'
                          : 'Connections use platform TLS unless a host is explicitly pinned',
                      style: const TextStyle(
                          fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: Ds.s2),
              const Text(
                'Hosts in the generated pin list use the restricted trust store. '
                'Other HTTPS hosts, including any Central Backend host not shown '
                'below, use platform TLS validation. A healthy pinned TC-WPN '
                'host is not proof that the Central Backend is pinned.',
                style:
                    TextStyle(fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
              ),
              const SizedBox(height: Ds.s4),
              const Divider(),
              const SizedBox(height: Ds.s3),
              if (_reports.isEmpty)
                Text('No pinned host has been checked yet.',
                    style: AppTheme.data(size: 11.5, color: Ds.inkFaint))
              else
                ..._reports.map(_row),
              const SizedBox(height: Ds.s3),
              Row(
                children: [
                  Expanded(
                    child: Text('Pin metadata: generated $kPinsGeneratedOn',
                        style: AppTheme.data(size: 10.5, color: Ds.inkFaint)),
                  ),
                  TextButton(
                    onPressed: _checking ? null : _recheck,
                    style: TextButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: Ds.s2),
                      minimumSize: Size.zero,
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                    child: Text(_checking ? 'Checking…' : 'Check pinned hosts'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _row(PinReport r) {
    final (color, label) = switch (r.status) {
      PinStatus.ok => (Ds.green, 'verified'),
      PinStatus.stale => (Ds.amber, 'review due'),
      PinStatus.disabled => (Ds.red, 'disabled'),
      PinStatus.notChecked => (Ds.inkFaint, 'not configured'),
      PinStatus.hostUnreachable => (Ds.inkFaint, 'unreachable'),
      PinStatus.pinMismatch => (Ds.red, 'REFUSED'),
    };

    return Padding(
      padding: const EdgeInsets.only(bottom: Ds.s3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 7,
            height: 7,
            margin: const EdgeInsets.only(top: 5),
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: Ds.s3),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(r.host,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: AppTheme.data(
                              size: 11.5, weight: FontWeight.w600)),
                    ),
                    const SizedBox(width: Ds.s2),
                    Text(label,
                        style: TextStyle(
                            fontSize: 10.5,
                            fontWeight: FontWeight.w700,
                            color: color)),
                  ],
                ),
                const SizedBox(height: 2),
                Text(r.detail,
                    style: const TextStyle(
                        fontSize: 11, color: Ds.inkFaint, height: 1.4)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
