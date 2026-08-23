// lib/features/auth/register_screen.dart
//
// Self-service registration in two steps: details, then the emailed code.
//
// The screen is honest about the three gates in front of an account, because a
// clinician who registers and then cannot sign in — with no explanation — will
// assume the app is broken. Stating "approval is required" up front costs one
// line and saves a support call.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/auth_service.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});
  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _form = GlobalKey<FormState>();
  final _id = TextEditingController();
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _confirm = TextEditingController();
  final _invite = TextEditingController();
  final _code = TextEditingController();

  bool _obscure = true;
  bool _busy = false;
  bool _sent = false; // step 2 once the code is out
  String? _error;
  String? _sentTo;

  @override
  void dispose() {
    for (final c in [_id, _name, _email, _password, _confirm, _invite, _code]) {
      c.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: Ds.surface,
        appBar: AppBar(
          title: Text(_sent ? 'Verify your email' : 'Create an account'),
        ),
        body: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(Ds.s6),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 420),
                child: _sent ? _verifyStep() : _detailsStep(),
              ),
            ),
          ),
        ),
      );

  // ── Step 1 ───────────────────────────────────────────────────────────────

  Widget _detailsStep() => Form(
        key: _form,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const InlineNotice(
              icon: Icons.how_to_reg_outlined,
              text: 'Accounts need an invite code from the study team, an '
                  'institutional email address, and approval before first '
                  'sign-in. You will be emailed when your account is active.',
            ),
            const SizedBox(height: Ds.s5),
            TextFormField(
              controller: _invite,
              decoration: const InputDecoration(
                labelText: 'Invite code',
                prefixIcon: Icon(Icons.vpn_key_outlined, size: 19),
              ),
              validator: (v) => (v ?? '').trim().isEmpty
                  ? 'Ask the study team for an invite code'
                  : null,
            ),
            const SizedBox(height: Ds.s3),
            TextFormField(
              controller: _id,
              textCapitalization: TextCapitalization.characters,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'Clinician ID',
                hintText: 'e.g. DR004',
                prefixIcon: Icon(Icons.badge_outlined, size: 19),
              ),
              validator: (v) => (v ?? '').trim().length < 2
                  ? 'Enter your clinician ID'
                  : null,
            ),
            const SizedBox(height: Ds.s3),
            TextFormField(
              controller: _name,
              textCapitalization: TextCapitalization.words,
              decoration: const InputDecoration(
                labelText: 'Full name',
                hintText: 'e.g. Dr A. Perera',
                prefixIcon: Icon(Icons.person_outline, size: 19),
              ),
              validator: (v) =>
                  (v ?? '').trim().length < 2 ? 'Enter your full name' : null,
            ),
            const SizedBox(height: Ds.s3),
            TextFormField(
              controller: _email,
              keyboardType: TextInputType.emailAddress,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'Institutional email',
                prefixIcon: Icon(Icons.mail_outline, size: 19),
              ),
              validator: (v) {
                final s = (v ?? '').trim();
                if (!s.contains('@') || !s.contains('.')) {
                  return 'Enter a valid email address';
                }
                return null;
              },
            ),
            const SizedBox(height: Ds.s3),
            TextFormField(
              controller: _password,
              obscureText: _obscure,
              decoration: InputDecoration(
                labelText: 'Password',
                helperText: 'At least 10 characters, with letters and numbers',
                prefixIcon: const Icon(Icons.lock_outline_rounded, size: 19),
                suffixIcon: IconButton(
                  icon: Icon(
                      _obscure
                          ? Icons.visibility_outlined
                          : Icons.visibility_off_outlined,
                      size: 19),
                  onPressed: () => setState(() => _obscure = !_obscure),
                ),
              ),
              validator: (v) {
                final s = v ?? '';
                if (s.length < 10) return 'At least 10 characters';
                if (!RegExp(r'[A-Za-z]').hasMatch(s) ||
                    !RegExp(r'\d').hasMatch(s)) {
                  return 'Include both letters and numbers';
                }
                return null;
              },
            ),
            const SizedBox(height: Ds.s3),
            TextFormField(
              controller: _confirm,
              obscureText: _obscure,
              decoration: const InputDecoration(
                labelText: 'Repeat password',
                prefixIcon: Icon(Icons.lock_outline_rounded, size: 19),
              ),
              validator: (v) =>
                  v != _password.text ? 'Passwords do not match' : null,
            ),
            if (_error != null) ...[
              const SizedBox(height: Ds.s4),
              InlineNotice(
                  icon: Icons.error_outline_rounded,
                  tone: Ds.red,
                  text: _error!),
            ],
            const SizedBox(height: Ds.s5),
            ElevatedButton(
              onPressed: _busy ? null : _register,
              child: _busy
                  ? const SizedBox(
                      width: 19,
                      height: 19,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Text('Create account'),
            ),
          ],
        ),
      );

  // ── Step 2 ───────────────────────────────────────────────────────────────

  Widget _verifyStep() => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Icon(Icons.mark_email_unread_outlined,
              size: 42, color: Ds.brand),
          const SizedBox(height: Ds.s4),
          Text('Check your email', style: AppTheme.display(size: 20)),
          const SizedBox(height: Ds.s2),
          Text(
            'A 6-digit code was sent to ${_sentTo ?? "your email"}. '
            'It expires in 10 minutes.',
            style: const TextStyle(
                fontSize: 13.5, color: Ds.inkMuted, height: 1.5),
          ),
          const SizedBox(height: Ds.s6),
          TextField(
            controller: _code,
            keyboardType: TextInputType.number,
            inputFormatters: [
              FilteringTextInputFormatter.digitsOnly,
              LengthLimitingTextInputFormatter(6),
            ],
            textAlign: TextAlign.center,
            style: AppTheme.data(
                size: 26, weight: FontWeight.w600, letterSpacing: 10),
            decoration: const InputDecoration(hintText: '••••••'),
            onChanged: (v) {
              if (v.length == 6) _verify();
            },
          ),
          if (_error != null) ...[
            const SizedBox(height: Ds.s4),
            InlineNotice(
                icon: Icons.error_outline_rounded, tone: Ds.red, text: _error!),
          ],
          const SizedBox(height: Ds.s5),
          ElevatedButton(
            onPressed: _busy ? null : _verify,
            child: _busy
                ? const SizedBox(
                    width: 19,
                    height: 19,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : const Text('Verify'),
          ),
          const SizedBox(height: Ds.s2),
          TextButton(
            onPressed: _busy ? null : _resend,
            child: const Text('Send a new code'),
          ),
          const SizedBox(height: Ds.s4),
          const InlineNotice(
            icon: Icons.schedule_rounded,
            text: 'After verifying, the study team must approve your account '
                'before you can sign in. You will be emailed when that happens.',
          ),
        ],
      );

  // ── Actions ──────────────────────────────────────────────────────────────

  Future<void> _register() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final to = await AuthService.register(
        clinicianId: _id.text.trim().toUpperCase(),
        displayName: _name.text.trim(),
        email: _email.text.trim(),
        password: _password.text,
        inviteCode: _invite.text.trim(),
      );
      if (!mounted) return;
      setState(() {
        _sent = true;
        _sentTo = to;
        _busy = false;
      });
    } on AuthMessage catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = e.message;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = 'Could not reach the study server. Check your connection.';
      });
    }
  }

  Future<void> _verify() async {
    if (_code.text.trim().length != 6) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final next = await AuthService.verifyEmail(
        clinicianId: _id.text.trim().toUpperCase(),
        code: _code.text.trim(),
      );
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Email verified'),
          content: Text(next),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(ctx), child: const Text('Done')),
          ],
        ),
      );
      if (mounted) Navigator.pop(context);
    } on AuthMessage catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = e.message;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = 'Could not reach the study server. Check your connection.';
      });
    }
  }

  Future<void> _resend() async {
    setState(() => _busy = true);
    try {
      await AuthService.resendVerification(_id.text.trim().toUpperCase());
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('A new code is on its way.')),
      );
    } catch (_) {
      // The endpoint answers identically whether or not the account is
      // pending, so there is nothing specific to report.
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }
}
