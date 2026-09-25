// lib/features/auth/login_screen.dart
//
// Authentication is real: it delegates to AuthService, which uses either local
// build-time credentials (development, demos) or your auth endpoint (required
// before real patient data).
//
// The screen makes the mode visible. A debug build shows the development
// credentials openly — hiding them helps nobody and encourages hardcoding them
// somewhere worse. A release build still on local auth shows a red warning
// banner, because that build must not be handed to a clinician.

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/auth_service.dart';
import '../../data/api/session.dart';
import '../../data/local/consent_store.dart';
import '../../data/local/stores.dart';
import '../shell.dart';
import 'forgot_password_screen.dart';
import 'register_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _form = GlobalKey<FormState>();
  final _id = TextEditingController();
  final _pw = TextEditingController();
  bool _obscure = true;
  bool _busy = false;
  String? _failure;

  @override
  void dispose() {
    _id.dispose();
    _pw.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: Ds.surface,
        body: SafeArea(
          child: Column(
            children: [
              if (AuthService.shouldWarnInsecure) const _InsecureBanner(),
              Expanded(
                child: Center(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(Ds.s6),
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 420),
                      child: Form(
                        key: _form,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Container(
                              width: 46,
                              height: 46,
                              decoration: BoxDecoration(
                                color: Ds.brand,
                                borderRadius: BorderRadius.circular(Ds.rMd),
                              ),
                              alignment: Alignment.center,
                              child: const Icon(Icons.hub_rounded,
                                  color: Colors.white, size: 23),
                            ),
                            const SizedBox(height: Ds.s6),
                            Text('ClinAnx',
                                style: AppTheme.display(size: 30, height: 1.1)),
                            const SizedBox(height: Ds.s1),
                            const Text(
                              'Multimodal anxiety risk console for psychiatry '
                              'teams. National Hospital of Sri Lanka.',
                              style: TextStyle(
                                  fontSize: 13.5,
                                  color: Ds.inkMuted,
                                  height: 1.5),
                            ),
                            const SizedBox(height: Ds.s8),
                            TextFormField(
                              controller: _id,
                              textCapitalization: TextCapitalization.characters,
                              autocorrect: false,
                              decoration: const InputDecoration(
                                labelText: 'Clinician ID',
                                prefixIcon:
                                    Icon(Icons.badge_outlined, size: 19),
                              ),
                              validator: (v) => (v ?? '').trim().isEmpty
                                  ? 'Enter your clinician ID'
                                  : null,
                            ),
                            const SizedBox(height: Ds.s3),
                            TextFormField(
                              controller: _pw,
                              obscureText: _obscure,
                              decoration: InputDecoration(
                                labelText: 'Password',
                                prefixIcon: const Icon(
                                    Icons.lock_outline_rounded,
                                    size: 19),
                                suffixIcon: IconButton(
                                  icon: Icon(
                                      _obscure
                                          ? Icons.visibility_outlined
                                          : Icons.visibility_off_outlined,
                                      size: 19),
                                  onPressed: () =>
                                      setState(() => _obscure = !_obscure),
                                ),
                              ),
                              validator: (v) => (v ?? '').isEmpty
                                  ? 'Enter your password'
                                  : null,
                              onFieldSubmitted: (_) => _signIn(),
                            ),
                            if (_failure != null) ...[
                              const SizedBox(height: Ds.s4),
                              InlineNotice(
                                icon: Icons.error_outline_rounded,
                                tone: Ds.red,
                                text: _failure!,
                              ),
                            ],
                            const SizedBox(height: Ds.s5),
                            ElevatedButton(
                              onPressed: _busy ? null : _signIn,
                              child: _busy
                                  ? const SizedBox(
                                      width: 19,
                                      height: 19,
                                      child: CircularProgressIndicator(
                                          strokeWidth: 2, color: Colors.white),
                                    )
                                  : const Text('Sign in'),
                            ),
                            if (AuthService.supportsSelfService) ...[
                              const SizedBox(height: Ds.s2),
                              TextButton(
                                onPressed: _busy
                                    ? null
                                    : () => Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                            builder: (_) =>
                                                const ForgotPasswordScreen())),
                                child: const Text('Forgot your password?'),
                              ),
                              const Divider(height: Ds.s6),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Text('No account yet?',
                                      style: TextStyle(
                                          fontSize: 13, color: Ds.inkMuted)),
                                  TextButton(
                                    onPressed: _busy
                                        ? null
                                        : () => Navigator.push(
                                            context,
                                            MaterialPageRoute(
                                                builder: (_) =>
                                                    const RegisterScreen())),
                                    child: const Text('Register'),
                                  ),
                                ],
                              ),
                            ],
                            if (kDebugMode && AuthService.isLocalMode) ...[
                              const SizedBox(height: Ds.s4),
                              const _DevHint(),
                            ],
                            const SizedBox(height: Ds.s5),
                            const DecisionSupportNotice(),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      );

  Future<void> _signIn() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _failure = null;
    });

    try {
      final session = await AuthService.signIn(
        clinicianId: _id.text.trim(),
        password: _pw.text,
      );

      if (session == null) {
        setState(() {
          _busy = false;
          _failure = 'That clinician ID and password were not recognised.';
        });
        return;
      }

      await SecureStore.saveSession(
        clinicianId: session.clinicianId,
        clinicianName: session.displayName,
        token: session.token,
      );
      Session.set(
        token: session.token,
        clinicianId: session.clinicianId,
      );
      // Links the acceptance record to the clinician. One-way: fills a null
      // field only, never overwrites.
      await ConsentStore.attachClinician(session.clinicianId);

      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const AppShell()),
      );
    } on AuthMessage catch (e) {
      // "Verify your email", "awaiting approval", "deactivated" — the clinician
      // needs the actual reason, not a generic failure.
      if (!mounted) return;
      setState(() {
        _busy = false;
        _failure = e.message;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _failure = 'Could not reach the sign-in service. Check your connection '
            'and try again.';
      });
    }
  }
}

class _InsecureBanner extends StatelessWidget {
  const _InsecureBanner();

  @override
  Widget build(BuildContext context) => Container(
        width: double.infinity,
        color: Ds.red,
        padding: const EdgeInsets.symmetric(horizontal: Ds.s4, vertical: Ds.s2),
        child: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.white, size: 15),
            SizedBox(width: Ds.s2),
            Expanded(
              child: Text(
                'Development build — credentials are stored in the app, not on a '
                'server. Do not use with real patient data.',
                style:
                    TextStyle(color: Colors.white, fontSize: 11, height: 1.35),
              ),
            ),
          ],
        ),
      );
}

class _DevHint extends StatelessWidget {
  const _DevHint();

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(Ds.s3),
        decoration: BoxDecoration(
          color: Ds.surfaceSunken,
          borderRadius: BorderRadius.circular(Ds.rMd),
          border: Border.all(color: Ds.hairline),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('DEVELOPMENT CREDENTIALS', style: AppTheme.eyebrow),
            const SizedBox(height: Ds.s2),
            Text('DR001  ·  clinanx-dev',
                style: AppTheme.data(size: 12, weight: FontWeight.w600)),
            Text('DR002  ·  clinanx-dev',
                style: AppTheme.data(size: 12, weight: FontWeight.w600)),
            const SizedBox(height: Ds.s2),
            const Text(
              'Demo/local credentials. Never use this branch with real patient data.',
              style: TextStyle(fontSize: 11, color: Ds.inkFaint),
            ),
          ],
        ),
      );
}
