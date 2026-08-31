// lib/features/consent/agreement_viewer_screen.dart
//
// Reached from Settings after acceptance. Read-only: the documents can be read
// and copied, and the acceptance receipt can be inspected, but nothing here can
// alter the recorded acceptance.
//
// The withdrawal control lives at the bottom. It exists because s.16 of the
// PDPA makes withdrawal a right that cannot be waived by agreement. It is
// deliberately unglamorous, guarded by a typed confirmation, and its effect is
// forward-only — see clause 10 of the Privacy Notice.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/session.dart';
import '../../data/local/consent_store.dart';
import '../../domain/consent.dart';
import '../auth/login_screen.dart';
import 'agreement_text.dart';

class AgreementViewerScreen extends StatelessWidget {
  const AgreementViewerScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Terms and privacy'),
          bottom: const TabBar(
            labelColor: Ds.brand,
            unselectedLabelColor: Ds.inkMuted,
            indicatorColor: Ds.brand,
            dividerColor: Ds.hairline,
            labelStyle: TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600),
            tabs: [
              Tab(text: 'Terms'),
              Tab(text: 'Privacy'),
              Tab(text: 'Your record'),
            ],
          ),
        ),
        body: const TabBarView(
          children: [
            _Doc(kTermsOfUse),
            _Doc(kPrivacyNotice),
            _ReceiptTab(),
          ],
        ),
      ),
    );
  }
}

class _Doc extends StatelessWidget {
  final String body;
  const _Doc(this.body);

  @override
  Widget build(BuildContext context) => SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(Ds.s5, Ds.s5, Ds.s5, Ds.s10),
        child: SelectionArea(
          child: Text(body,
              style:
                  const TextStyle(fontSize: 13, height: 1.62, color: Ds.ink)),
        ),
      );
}

class _ReceiptTab extends StatefulWidget {
  const _ReceiptTab();
  @override
  State<_ReceiptTab> createState() => _ReceiptTabState();
}

class _ReceiptTabState extends State<_ReceiptTab> {
  ConsentRecord? _rec;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    ConsentStore.current().then((r) {
      if (mounted) {
        setState(() {
          _rec = r;
          _loading = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator(color: Ds.brand));
    }
    final r = _rec;
    if (r == null) {
      return const EmptyState(
        icon: Icons.description_outlined,
        title: 'No acceptance on record',
        body:
            'This should not happen. Report it to the Principal Investigator.',
      );
    }

    final fmt = DateFormat('d MMMM y · HH:mm');

    return ListView(
      padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s5, Ds.s4, Ds.s10),
      children: [
        Panel(
          padding: const EdgeInsets.all(Ds.s5),
          background: r.isWithdrawn ? Ds.amberSoft : Ds.greenSoft,
          borderColor:
              (r.isWithdrawn ? Ds.amber : Ds.green).withValues(alpha: 0.24),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                r.isWithdrawn
                    ? Icons.pause_circle_outline_rounded
                    : Icons.verified_rounded,
                size: 20,
                color: r.isWithdrawn ? Ds.amber : Ds.green,
              ),
              const SizedBox(width: Ds.s3),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      r.isWithdrawn ? 'Consent withdrawn' : 'Accepted',
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: r.isWithdrawn ? Ds.amber : Ds.green),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      r.isWithdrawn
                          ? 'Withdrawn ${fmt.format(r.withdrawnAtUtc!.toLocal())}. '
                              'The original acceptance below is retained as a record.'
                          : 'This record cannot be edited. It is superseded only '
                              'if a new version of the agreement is published.',
                      style: const TextStyle(
                          fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: Ds.s5),
        const SectionLabel('Acceptance receipt'),
        Panel(
          child: Column(
            children: [
              _kv('Version', r.agreementVersion),
              _kv('Accepted', fmt.format(r.acceptedAtUtc.toLocal())),
              _kv('Recorded (UTC)', r.acceptedAtUtc.toIso8601String()),
              _kv('Clinician', r.clinicianId ?? 'not yet linked'),
              _kv('Platform', r.platform),
              _kv('App build', r.appVersion),
            ],
          ),
        ),
        const SizedBox(height: Ds.s3),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('TEXT ANCHOR (SHA-256)', style: AppTheme.eyebrow),
              const SizedBox(height: Ds.s2),
              SelectableText(
                r.textSha256,
                style: AppTheme.data(size: 11, height: 1.6),
              ),
              const SizedBox(height: Ds.s3),
              const Text(
                'This hash is computed from the exact text you accepted. It '
                'proves what was on screen at the time, and would change if a '
                'single character of that text were altered.',
                style:
                    TextStyle(fontSize: 11.5, color: Ds.inkFaint, height: 1.45),
              ),
            ],
          ),
        ),
        if (!r.isWithdrawn) ...[
          const SizedBox(height: Ds.s6),
          const SectionLabel('Withdrawing'),
          const InlineNotice(
            icon: Icons.info_outline_rounded,
            text:
                'Withdrawal ends your access and stops further collection. It '
                'does not undo processing that has already lawfully taken place, '
                'and it does not delete this acceptance record. Read clause 10 of '
                'the Privacy Notice before proceeding.',
          ),
          const SizedBox(height: Ds.s4),
          OutlinedButton.icon(
            onPressed: () => _confirmWithdraw(context),
            icon: const Icon(Icons.logout_rounded, size: 18),
            label: const Text('Withdraw consent'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Ds.red,
              side: const BorderSide(color: Ds.hairlineStrong),
            ),
          ),
        ],
      ],
    );
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 118,
              child: Text(k,
                  style: const TextStyle(fontSize: 12.5, color: Ds.inkMuted)),
            ),
            Expanded(
                child: Text(v, style: AppTheme.data(size: 11.5, height: 1.5))),
          ],
        ),
      );

  /// Typed confirmation. A single tap is too easy for something irreversible.
  Future<void> _confirmWithdraw(BuildContext context) async {
    final ctrl = TextEditingController();
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialog) => AlertDialog(
          title: const Text('Withdraw consent?'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'You will be signed out and will not be able to use ClinAnx '
                'again unless consent is re-established with the study team.\n\n'
                'Type WITHDRAW to confirm.',
                style: TextStyle(fontSize: 13, height: 1.5),
              ),
              const SizedBox(height: Ds.s4),
              TextField(
                controller: ctrl,
                autofocus: true,
                textCapitalization: TextCapitalization.characters,
                decoration: const InputDecoration(hintText: 'WITHDRAW'),
                onChanged: (_) => setDialog(() {}),
              ),
            ],
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Cancel')),
            TextButton(
              onPressed: ctrl.text.trim().toUpperCase() == 'WITHDRAW'
                  ? () => Navigator.pop(ctx, true)
                  : null,
              child: const Text('Withdraw', style: TextStyle(color: Ds.red)),
            ),
          ],
        ),
      ),
    );
    ctrl.dispose();
    if (ok != true || !context.mounted) return;

    await ConsentStore.withdraw();
    await Session.signOut();
    if (!context.mounted) return;
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (_) => false,
    );
  }
}
