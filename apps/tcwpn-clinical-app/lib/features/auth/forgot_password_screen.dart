// lib/features/auth/forgot_password_screen.dart
//
// Three steps: email → code → new password.
//
// One deliberate restraint: after the email is submitted, the screen says "if
// that address is registered, a code has been sent" — never "code sent" or "no
// such account". The server answers identically either way so the endpoint
// cannot be used to discover who is on the study, and the UI must not undo
// that by being more helpful than the server.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/auth_service.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});
  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _email = TextEditingController();
  final _code = TextEditingController();
  final _password = TextEditingController();
  final _confirm = TextEditingController();

  int _step = 0; // 0 email · 1 code + new password
  bool _obscure = true;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    for (final c in [_email, _code, _password, _confirm]) {
      c.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: Ds.surface,
        appBar: AppBar(title: const Text('Reset your password')),
        body: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(Ds.s6),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 420),
                child: _step == 0 ? _emailStep() : _resetStep(),
              ),
            ),
          ),
        ),
      );

  Widget _emailStep() => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Icon(Icons.lock_reset_rounded, size: 42, color: Ds.brand),
          const SizedBox(height: Ds.s4),
          Text('Forgot your password?', style: AppTheme.display(size: 20)),
          const SizedBox(height: Ds.s2),
          const Text(
            'Enter the institutional email address on your ClinAnx account. '
            'We will send a 6-digit reset code.',
            style: TextStyle(fontSize: 13.5, color: Ds.inkMuted, height: 1.5),
          ),
          const SizedBox(height: Ds.s6),
          TextField(
            controller: _email,
            keyboardType: TextInputType.emailAddress,
            autocorrect: false,
            decoration: const InputDecoration(
              labelText: 'Email address',
              prefixIcon: Icon(Icons.mail_outline, size: 19),
            ),
            onSubmitted: (_) => _request(),
          ),
          if (_error != null) ...[
            const SizedBox(height: Ds.s4),
            InlineNotice(
                icon: Icons.error_outline_rounded, tone: Ds.red, text: _error!),
          ],
          const SizedBox(height: Ds.s5),
          ElevatedButton(
            onPressed: _busy ? null : _request,
            child: _busy
                ? const SizedBox(
                    width: 19,
                    height: 19,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : const Text('Send reset code'),
          ),
        ],
      );

  Widget _resetStep() => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const InlineNotice(
            icon: Icons.mark_email_unread_outlined,
            text:
                'If that address is registered, a 6-digit code has been sent. '
                'It expires in 10 minutes.',
          ),
          const SizedBox(height: Ds.s5),
          TextField(
            controller: _code,
            keyboardType: TextInputType.number,
            inputFormatters: [
              FilteringTextInputFormatter.digitsOnly,
              LengthLimitingTextInputFormatter(6),
            ],
            textAlign: TextAlign.center,
            style: AppTheme.data(
                size: 24, weight: FontWeight.w600, letterSpacing: 9),
            decoration: const InputDecoration(hintText: '••••••'),
          ),
          const SizedBox(height: Ds.s4),
          TextField(
            controller: _password,
            obscureText: _obscure,
            decoration: InputDecoration(
              labelText: 'New password',
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
          ),
          const SizedBox(height: Ds.s3),
          TextField(
            controller: _confirm,
            obscureText: _obscure,
            decoration: const InputDecoration(
              labelText: 'Repeat new password',
              prefixIcon: Icon(Icons.lock_outline_rounded, size: 19),
            ),
          ),
          if (_error != null) ...[
            const SizedBox(height: Ds.s4),
            InlineNotice(
                icon: Icons.error_outline_rounded, tone: Ds.red, text: _error!),
          ],
          const SizedBox(height: Ds.s5),
          ElevatedButton(
            onPressed: _busy ? null : _reset,
            child: _busy
                ? const SizedBox(
                    width: 19,
                    height: 19,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : const Text('Set new password'),
          ),
          const SizedBox(height: Ds.s2),
          TextButton(
            onPressed: _busy ? null : _request,
            child: const Text('Send a new code'),
          ),
        ],
      );

  Future<void> _request() async {
    final email = _email.text.trim();
    if (!email.contains('@')) {
      setState(() => _error = 'Enter a valid email address.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await AuthService.requestReset(email);
      if (!mounted) return;
      setState(() {
        _step = 1;
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

  Future<void> _reset() async {
    if (_code.text.trim().length != 6) {
      setState(() => _error = 'Enter the 6-digit code from your email.');
      return;
    }
    if (_password.text.length < 10 ||
        !RegExp(r'[A-Za-z]').hasMatch(_password.text) ||
        !RegExp(r'\d').hasMatch(_password.text)) {
      setState(() => _error =
          'Password must be at least 10 characters, with letters and numbers.');
      return;
    }
    if (_password.text != _confirm.text) {
      setState(() => _error = 'Passwords do not match.');
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await AuthService.resetPassword(
        email: _email.text.trim(),
        code: _code.text.trim(),
        newPassword: _password.text,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Password changed. Sign in with your new password.')),
      );
      Navigator.pop(context);
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
}
