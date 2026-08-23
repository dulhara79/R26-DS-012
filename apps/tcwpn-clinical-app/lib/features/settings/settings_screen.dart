// lib/features/settings/settings_screen.dart
//
// Settings shows what is actually true. The previous build advertised a
// "Biometric login: Enabled" switch wired to an empty callback, and printed
// three different AUROC figures across three screens. Every value here is read
// from configuration or from the model service's own /health response; nothing
// is a literal typed into the UI.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/config/env.dart';
import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../core/security/secure_http.dart';
import '../../data/api/session.dart';
import '../../data/local/stores.dart';
import '../../state/controllers.dart';
import '../auth/login_screen.dart';
import '../../core/security/pinned_certificates.dart';

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
          SectionLabel('Services ClinAnx uses'),
          Panel(
            child: Column(
              children: [
                _ServiceRow(
                  name: 'Clinical NLP (TC-WPN)',
                  url: Env.tcwpnBase,
                  reachable: info != null,
                  accent: Ds.c4ClinicalNlp,
                  role: 'Analyses clinical notes · Component 4',
                ),
                const Divider(height: Ds.s5),
                _ServiceRow(
                  name: 'Fusion layer',
                  url: Env.fusionBase,
                  reachable: null,
                  accent: Ds.brand,
                  role: 'Combines all four components · §5.1',
                ),
                const Divider(height: Ds.s5),
                _ServiceRow(
                  name: 'Intervention engine',
                  url: Env.c3Base,
                  reachable: null,
                  accent: Ds.c3Intervention,
                  role: 'Risk tiering and coping plans · Component 3',
                ),
              ],
            ),
          ),
          const SizedBox(height: Ds.s3),
          const InlineNotice(
            icon: Icons.devices_rounded,
            text:
                'The wearable and behavioural components are collected by the '
                'patient-facing app and sent to the fusion service directly. '
                'ClinAnx never contacts them, and holds no credentials for '
                'them. Their values reach this app only inside the fused '
                'composite, keyed by MRN.',
          ),
          const SizedBox(height: Ds.s5),
          SectionLabel('Model information'),
          Panel(
            child: info == null
                ? const Text(
                    'Waiting for the clinical NLP service to report. Figures shown '
                    'here come from the service itself, never from values typed '
                    'into the app.',
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
          SectionLabel('Connection security'),
          const _PinningPanel(),
          const SizedBox(height: Ds.s5),
          SectionLabel('Data governance'),
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                _GovRow(Icons.lock_outline_rounded, 'Credentials',
                    'Stored in the device keychain, encrypted at rest.'),
                Divider(height: Ds.s5),
                _GovRow(Icons.folder_outlined, 'Clinical records',
                    'Held on this device and scoped per patient. Removing a patient purges every namespace.'),
                Divider(height: Ds.s5),
                _GovRow(Icons.cloud_upload_outlined, 'Note text',
                    'Sent to the model service for analysis. De-identify before submitting.'),
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
          'Your session ends. Patient records stay on this device and will be '
          'available at the next sign-in.',
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
    await SecureStore.signOut();
    Session.clear();
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

// ─────────────────────────────────────────────────────────────────────────────
// TLS pinning status
//
// Reports what is actually true rather than a reassuring constant. A clinician
// on a hospital network that inspects traffic needs to see WHY submissions are
// failing, and the study team needs to see when the pin set is going stale
// before it becomes an outage someone else discovers.
// ─────────────────────────────────────────────────────────────────────────────
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
        if (kPinningDisabled)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.gpp_bad_outlined,
              tone: Ds.red,
              text: 'Certificate pinning is DISABLED by a build flag. This '
                  'build must not be used with patient data.',
            ),
          )
        else if (!configured)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.gpp_maybe_outlined,
              tone: Ds.red,
              text: 'No certificates are pinned. Run tool/pin_certs.py and '
                  'rebuild before using this with patient data.',
            ),
          )
        else if (SecureHttp.needsReview)
          Padding(
            padding: const EdgeInsets.only(bottom: Ds.s3),
            child: InlineNotice(
              icon: Icons.update_outlined,
              tone: Ds.amber,
              text: 'The pinned certificate set is past its review date '
                  '($kPinsReviewBy). Regenerate it before it drifts out of '
                  'date and blocks connections.',
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
                      configured && !kPinningDisabled
                          ? 'Traffic is restricted to pinned certificate '
                              'authorities'
                          : 'Pinning is not active',
                      style: const TextStyle(
                          fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: Ds.s2),
              const Text(
                'Clinical note text crosses the public internet. Pinning means '
                'the app trusts only the certificate authority that actually '
                'serves these hosts — an intercepting proxy is refused before '
                'any text is sent.',
                style:
                    TextStyle(fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
              ),
              const SizedBox(height: Ds.s4),
              const Divider(),
              const SizedBox(height: Ds.s3),
              if (_reports.isEmpty)
                Text('Not checked yet.',
                    style: AppTheme.data(size: 11.5, color: Ds.inkFaint))
              else
                ..._reports.map(_row),
              const SizedBox(height: Ds.s3),
              Row(
                children: [
                  Text('Pins generated $kPinsGeneratedOn',
                      style: AppTheme.data(size: 10.5, color: Ds.inkFaint)),
                  const Spacer(),
                  TextButton(
                    onPressed: _checking ? null : _recheck,
                    style: TextButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: Ds.s2),
                      minimumSize: Size.zero,
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                    child: Text(_checking ? 'Checking…' : 'Check now'),
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
