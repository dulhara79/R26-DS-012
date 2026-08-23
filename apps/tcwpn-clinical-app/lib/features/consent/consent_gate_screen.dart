// lib/features/consent/consent_gate_screen.dart
//
// The gate. Nothing in the application is reachable until this is cleared.
//
// Enforcement:
//   • PopScope blocks the system back gesture and the Android back button.
//   • The agree button stays disabled until the user has scrolled to the end of
//     BOTH documents and ticked BOTH boxes. "I didn't read it" is a real defence
//     against a click-through agreement; scroll-tracking is what defeats it.
//   • Declining does not proceed to a reduced-function mode. It closes the app.
//   • Acceptance is written before navigation, so a crash mid-transition cannot
//     leave a user inside the app without a record.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/local/consent_store.dart';
import '../../domain/consent.dart';
import 'agreement_text.dart';

class ConsentGateScreen extends StatefulWidget {
  final String appVersion;
  final VoidCallback onAccepted;

  const ConsentGateScreen({
    super.key,
    required this.appVersion,
    required this.onAccepted,
  });

  @override
  State<ConsentGateScreen> createState() => _ConsentGateScreenState();
}

class _ConsentGateScreenState extends State<ConsentGateScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs = TabController(length: 2, vsync: this);
  final _termsCtrl = ScrollController();
  final _privacyCtrl = ScrollController();

  bool _termsRead = false;
  bool _privacyRead = false;
  bool _termsTicked = false;
  bool _privacyTicked = false;
  bool _busy = false;

  bool get _canAccept =>
      _termsRead && _privacyRead && _termsTicked && _privacyTicked && !_busy;

  @override
  void initState() {
    super.initState();
    _termsCtrl.addListener(() => _watch(_termsCtrl, () => _termsRead = true));
    _privacyCtrl
        .addListener(() => _watch(_privacyCtrl, () => _privacyRead = true));
  }

  void _watch(ScrollController c, VoidCallback mark) {
    if (!c.hasClients) return;
    // 40px tolerance: overscroll physics can stop just short of the extent.
    if (c.offset >= c.position.maxScrollExtent - 40) {
      setState(mark);
    }
  }

  @override
  void dispose() {
    _tabs.dispose();
    _termsCtrl.dispose();
    _privacyCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false, // no back out of the gate
      child: Scaffold(
        backgroundColor: Ds.canvas,
        body: SafeArea(
          child: Column(
            children: [
              _header(),
              _TabStrip(
                controller: _tabs,
                termsRead: _termsRead,
                privacyRead: _privacyRead,
              ),
              const Divider(height: 1),
              Expanded(
                child: TabBarView(
                  controller: _tabs,
                  children: [
                    _document(kTermsOfUse, _termsCtrl, _termsRead),
                    _document(kPrivacyNotice, _privacyCtrl, _privacyRead),
                  ],
                ),
              ),
              _footer(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header() => Padding(
        padding: const EdgeInsets.fromLTRB(Ds.s5, Ds.s5, Ds.s5, Ds.s4),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: Ds.brand,
                borderRadius: BorderRadius.circular(Ds.rMd),
              ),
              alignment: Alignment.center,
              child:
                  const Icon(Icons.hub_rounded, color: Colors.white, size: 20),
            ),
            const SizedBox(width: Ds.s3),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Before you begin', style: AppTheme.display(size: 19)),
                  Text(
                    'Version $kAgreementVersion · Effective $kAgreementEffectiveDate',
                    style: AppTheme.data(size: 10.5, color: Ds.inkFaint),
                  ),
                ],
              ),
            ),
          ],
        ),
      );

  Widget _document(String body, ScrollController ctrl, bool read) => Stack(
        children: [
          Scrollbar(
            controller: ctrl,
            child: SingleChildScrollView(
              controller: ctrl,
              padding: const EdgeInsets.fromLTRB(Ds.s5, Ds.s5, Ds.s5, Ds.s10),
              child: SelectionArea(
                child: Text(
                  body,
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.62,
                    color: Ds.ink,
                  ),
                ),
              ),
            ),
          ),
          if (!read)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: IgnorePointer(
                child: Container(
                  height: 72,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Ds.canvas.withValues(alpha: 0),
                        Ds.canvas,
                      ],
                    ),
                  ),
                  alignment: Alignment.bottomCenter,
                  padding: const EdgeInsets.only(bottom: Ds.s2),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.keyboard_arrow_down_rounded,
                          size: 16, color: Ds.inkFaint),
                      const SizedBox(width: 4),
                      Text('Scroll to the end to continue',
                          style: AppTheme.eyebrow),
                    ],
                  ),
                ),
              ),
            ),
        ],
      );

  Widget _footer() => Container(
        decoration: const BoxDecoration(
          color: Ds.surface,
          border: Border(top: BorderSide(color: Ds.hairline)),
        ),
        padding: const EdgeInsets.fromLTRB(Ds.s5, Ds.s4, Ds.s5, Ds.s5),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _tick(
              value: _termsTicked,
              enabled: _termsRead,
              label:
                  'I have read the Terms of Use and I accept them. I understand I '
                  'must de-identify every clinical note before submitting it.',
              onChanged: (v) => setState(() => _termsTicked = v),
            ),
            const SizedBox(height: Ds.s2),
            _tick(
              value: _privacyTicked,
              enabled: _privacyRead,
              label:
                  'I have read the Privacy Notice. I understand that note text is '
                  'processed outside Sri Lanka.',
              onChanged: (v) => setState(() => _privacyTicked = v),
            ),
            const SizedBox(height: Ds.s4),
            ElevatedButton(
              onPressed: _canAccept ? _accept : null,
              child: _busy
                  ? const SizedBox(
                      width: 19,
                      height: 19,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white),
                    )
                  : const Text('I agree and continue'),
            ),
            const SizedBox(height: Ds.s2),
            TextButton(
              onPressed: _busy ? null : _decline,
              style: TextButton.styleFrom(foregroundColor: Ds.inkMuted),
              child: const Text('I do not agree'),
            ),
            const SizedBox(height: Ds.s1),
            Text(
              'Your acceptance is recorded with a timestamp and a cryptographic '
              'hash of this exact text. It cannot be edited afterwards.',
              textAlign: TextAlign.center,
              style: const TextStyle(
                  fontSize: 11, color: Ds.inkFaint, height: 1.4),
            ),
          ],
        ),
      );

  Widget _tick({
    required bool value,
    required bool enabled,
    required String label,
    required ValueChanged<bool> onChanged,
  }) =>
      Opacity(
        opacity: enabled ? 1 : 0.42,
        child: InkWell(
          onTap: enabled ? () => onChanged(!value) : null,
          borderRadius: BorderRadius.circular(Ds.rSm),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 2),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(
                  width: 26,
                  height: 26,
                  child: Checkbox(
                    value: value,
                    onChanged: enabled ? (v) => onChanged(v ?? false) : null,
                    activeColor: Ds.brand,
                    visualDensity: VisualDensity.compact,
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ),
                const SizedBox(width: Ds.s3),
                Expanded(
                  child: Text(
                    label,
                    style: const TextStyle(
                        fontSize: 12.5, color: Ds.ink, height: 1.45),
                  ),
                ),
              ],
            ),
          ),
        ),
      );

  Future<void> _accept() async {
    setState(() => _busy = true);
    // The hash covers the exact bytes on screen, so the record proves content.
    final hash =
        ConsentRecord.hashOf(kTermsOfUse, kPrivacyNotice, kAgreementVersion);
    await ConsentStore.record(
      textSha256: hash,
      appVersion: widget.appVersion,
    );
    if (!mounted) return;
    widget.onAccepted();
  }

  Future<void> _decline() async {
    final leave = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('ClinAnx cannot be used'),
        content: const Text(
          'Acceptance of the Terms of Use and the Privacy Notice is required to '
          'use this application. Without it, the application will close.\n\n'
          'If you have questions about the terms, contact the Principal '
          'Investigator before accepting.',
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Back to the terms')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Close the app', style: TextStyle(color: Ds.red)),
          ),
        ],
      ),
    );
    if (leave == true) await SystemNavigator.pop();
  }
}

class _TabStrip extends StatelessWidget {
  final TabController controller;
  final bool termsRead;
  final bool privacyRead;

  const _TabStrip({
    required this.controller,
    required this.termsRead,
    required this.privacyRead,
  });

  @override
  Widget build(BuildContext context) => Container(
        color: Ds.surface,
        child: TabBar(
          controller: controller,
          labelColor: Ds.brand,
          unselectedLabelColor: Ds.inkMuted,
          indicatorColor: Ds.brand,
          indicatorWeight: 2.5,
          dividerColor: Colors.transparent,
          labelStyle:
              const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
          tabs: [
            _tab('Terms of Use', termsRead),
            _tab('Privacy Notice', privacyRead),
          ],
        ),
      );

  Widget _tab(String label, bool read) => Tab(
        height: 46,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(label),
            const SizedBox(width: 6),
            Icon(
              read
                  ? Icons.check_circle_rounded
                  : Icons.radio_button_unchecked_rounded,
              size: 14,
              color: read ? Ds.green : Ds.hairlineStrong,
            ),
          ],
        ),
      );
}
