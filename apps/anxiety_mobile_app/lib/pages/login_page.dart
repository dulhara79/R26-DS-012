import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../background_service.dart';
import '../profile_page.dart';
import '../services/auth_validators.dart';
import '../services/api_service.dart';
import '../services/demo_auth_service.dart';
import '../services/participant_identity_service.dart';
import '../services/user_manager.dart';
import '../theme/app_theme.dart';
import 'baseline_calibration_page.dart';
import 'informed_consent_page.dart';
import 'main_navigation_page.dart';

enum _AuthMode { login, signup }

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _loginFormKey = GlobalKey<FormState>();
  final _signupFormKey = GlobalKey<FormState>();

  final _loginEmailController = TextEditingController();
  final _loginPasswordController = TextEditingController();

  final _nameController = TextEditingController();
  final _signupEmailController = TextEditingController();
  final _ageController = TextEditingController();
  final _signupPasswordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();

  _AuthMode _mode = _AuthMode.login;
  bool _loginPasswordVisible = false;
  bool _signupPasswordVisible = false;
  bool _confirmPasswordVisible = false;
  bool _submitting = false;

  @override
  void dispose() {
    _loginEmailController.dispose();
    _loginPasswordController.dispose();
    _nameController.dispose();
    _signupEmailController.dispose();
    _ageController.dispose();
    _signupPasswordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  Future<void> _submitLogin() async {
    if (_submitting || !_loginFormKey.currentState!.validate()) return;

    setState(() => _submitting = true);
    try {
      final result = await DemoAuthService.login(
        email: _loginEmailController.text,
        password: _loginPasswordController.text,
      );

      if (!mounted) return;
      if (!result.isSuccess) {
        _showError(result.error ?? 'Could not log in.');
        return;
      }

      final account = result.account!;
      await _selfEnrolParticipant(account.participantId);
      UserManager().login(account.participantId);
      await _startCollectionIfPossible();

      final prefs = await SharedPreferences.getInstance();
      final profileComplete = prefs.getBool('profile_complete') ?? false;
      final calibrationComplete =
          prefs.getBool('calibration_complete') ?? false;

      if (!mounted) return;
      final Widget destination;
      if (!profileComplete) {
        destination = const ProfilePage();
      } else if (!calibrationComplete) {
        destination = BaselineCalibrationPage(userId: account.participantId);
      } else {
        destination = MainNavigationPage(userId: account.participantId);
      }

      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => destination),
      );
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _submitSignup() async {
    if (_submitting || !_signupFormKey.currentState!.validate()) return;

    setState(() => _submitting = true);
    try {
      final result = await DemoAuthService.signUp(
        displayName: _nameController.text,
        email: _signupEmailController.text,
        age: int.parse(_ageController.text.trim()),
        password: _signupPasswordController.text,
      );

      if (!mounted) return;
      if (!result.isSuccess) {
        _showError(result.error ?? 'Could not create the account.');
        return;
      }

      final account = result.account!;
      await _selfEnrolParticipant(account.participantId);
      UserManager().login(account.participantId);
      await _startCollectionIfPossible();

      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const ProfilePage()),
      );
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _selfEnrolParticipant(String participantId) async {
    // Register this participant on the central backend so fusion can
    // accumulate scores before a clinician scans their QR. Idempotent
    // server-side, so a retry after a dropped response is safe.
    final subjectId = await ApiService.selfEnrol(participantId);
    if (subjectId != null && subjectId.isNotEmpty) {
      await ParticipantIdentityService.saveCentralSubjectId(subjectId);
    }
  }

  Future<void> _startCollectionIfPossible() async {
    if (kIsWeb) return;
    try {
      await startBackgroundServiceIfPermitted();
    } catch (e) {
      debugPrint('Background service start deferred after demo auth: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: GoogleFonts.poppins(fontSize: 13)),
        backgroundColor: Colors.red.shade700,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _switchMode(_AuthMode mode) {
    if (_mode == mode) return;
    FocusScope.of(context).unfocus();
    setState(() => _mode = mode);
  }

  @override
  Widget build(BuildContext context) {
    final dark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: dark
                ? const [Color(0xFF111218), Color(0xFF1A1B24)]
                : const [AppTheme.kBgTop, AppTheme.kBgBottom],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 28),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 520),
                child: Column(
                  children: [
                    _buildLogo(),
                    const SizedBox(height: 18),
                    Text(
                      'Welcome to Aura',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.poppins(
                        fontSize: 28,
                        fontWeight: FontWeight.w700,
                        color: Theme.of(context).colorScheme.onSurface,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Sign in to continue, or create a simple demo account.',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.poppins(
                        fontSize: 13.5,
                        height: 1.45,
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                    const SizedBox(height: 22),
                    _modeSelector(),
                    const SizedBox(height: 18),
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 220),
                      child: _mode == _AuthMode.login
                          ? _loginForm()
                          : _signupForm(),
                    ),
                    const SizedBox(height: 18),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Theme.of(context)
                            .colorScheme
                            .surface
                            .withValues(alpha: 0.72),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(
                            Icons.info_outline_rounded,
                            size: 17,
                            color: Theme.of(context).colorScheme.primary,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'Demo authentication only: no JWT, refresh token, or server session is used. '
                              'The account exists only on this device. Research data still uses a separate random Participant ID.',
                              style: GoogleFonts.poppins(
                                fontSize: 10.5,
                                height: 1.45,
                                color: Theme.of(
                                  context,
                                ).colorScheme.onSurfaceVariant,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextButton(
                      onPressed: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) =>
                                const InformedConsentPage(readOnly: true),
                          ),
                        );
                      },
                      child: const Text('Review Informed Consent & Privacy'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _modeSelector() {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Expanded(
            child: _modeButton(
              label: 'Log in',
              mode: _AuthMode.login,
              icon: Icons.login_rounded,
            ),
          ),
          Expanded(
            child: _modeButton(
              label: 'Sign up',
              mode: _AuthMode.signup,
              icon: Icons.person_add_alt_1_rounded,
            ),
          ),
        ],
      ),
    );
  }

  Widget _modeButton({
    required String label,
    required _AuthMode mode,
    required IconData icon,
  }) {
    final selected = _mode == mode;
    return InkWell(
      onTap: _submitting ? null : () => _switchMode(mode),
      borderRadius: BorderRadius.circular(12),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(vertical: 11),
        decoration: BoxDecoration(
          color: selected
              ? AppTheme.kPrimaryDeep.withValues(alpha: 0.13)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 18,
              color: selected
                  ? AppTheme.kPrimaryDeep
                  : Theme.of(context).colorScheme.onSurfaceVariant,
            ),
            const SizedBox(width: 7),
            Text(
              label,
              style: GoogleFonts.poppins(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: selected
                    ? AppTheme.kPrimaryDeep
                    : Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _loginForm() {
    return _formCard(
      key: const ValueKey('login'),
      child: Form(
        key: _loginFormKey,
        child: Column(
          children: [
            _field(
              controller: _loginEmailController,
              label: 'Email address',
              hint: 'name@example.com',
              icon: Icons.email_outlined,
              keyboardType: TextInputType.emailAddress,
              validator: AuthValidators.email,
              autofillHints: const [AutofillHints.email],
            ),
            const SizedBox(height: 14),
            _passwordField(
              controller: _loginPasswordController,
              label: 'Password',
              visible: _loginPasswordVisible,
              onVisibilityChanged: () => setState(
                () => _loginPasswordVisible = !_loginPasswordVisible,
              ),
              validator: (value) =>
                  (value ?? '').isEmpty ? 'Password is required.' : null,
              onSubmitted: (_) => _submitLogin(),
            ),
            const SizedBox(height: 18),
            _primaryButton(
              text: 'Log in',
              icon: Icons.arrow_forward_rounded,
              onPressed: _submitLogin,
            ),
            const SizedBox(height: 10),
            Text(
              'For the demo, log in with an account previously created on this device.',
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: 10.5,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _signupForm() {
    return _formCard(
      key: const ValueKey('signup'),
      child: Form(
        key: _signupFormKey,
        child: Column(
          children: [
            _field(
              controller: _nameController,
              label: 'Display name',
              hint: 'What should Aura call you?',
              icon: Icons.person_outline_rounded,
              textCapitalization: TextCapitalization.words,
              validator: AuthValidators.displayName,
              autofillHints: const [AutofillHints.name],
            ),
            const SizedBox(height: 14),
            _field(
              controller: _signupEmailController,
              label: 'Email address',
              hint: 'name@example.com',
              icon: Icons.email_outlined,
              keyboardType: TextInputType.emailAddress,
              validator: AuthValidators.email,
              autofillHints: const [AutofillHints.email],
            ),
            const SizedBox(height: 14),
            _field(
              controller: _ageController,
              label: 'Age',
              hint: '18–30',
              icon: Icons.cake_outlined,
              keyboardType: TextInputType.number,
              validator: AuthValidators.age,
            ),
            const SizedBox(height: 14),
            _passwordField(
              controller: _signupPasswordController,
              label: 'Password',
              visible: _signupPasswordVisible,
              onVisibilityChanged: () => setState(
                () => _signupPasswordVisible = !_signupPasswordVisible,
              ),
              validator: AuthValidators.password,
            ),
            const SizedBox(height: 14),
            _passwordField(
              controller: _confirmPasswordController,
              label: 'Confirm password',
              visible: _confirmPasswordVisible,
              onVisibilityChanged: () => setState(
                () => _confirmPasswordVisible = !_confirmPasswordVisible,
              ),
              validator: (value) => AuthValidators.confirmPassword(
                value,
                _signupPasswordController.text,
              ),
              onSubmitted: (_) => _submitSignup(),
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Password: 8+ characters with uppercase, lowercase and a number.',
                style: GoogleFonts.poppins(
                  fontSize: 10.5,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ),
            const SizedBox(height: 18),
            _primaryButton(
              text: 'Create account',
              icon: Icons.person_add_alt_1_rounded,
              onPressed: _submitSignup,
            ),
          ],
        ),
      ),
    );
  }

  Widget _formCard({required Key key, required Widget child}) {
    return Container(
      key: key,
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 18,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _field({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    required String? Function(String?) validator,
    TextInputType keyboardType = TextInputType.text,
    TextCapitalization textCapitalization = TextCapitalization.none,
    List<String>? autofillHints,
  }) {
    return TextFormField(
      controller: controller,
      keyboardType: keyboardType,
      textCapitalization: textCapitalization,
      autofillHints: autofillHints,
      enabled: !_submitting,
      validator: validator,
      autovalidateMode: AutovalidateMode.onUserInteraction,
      style: GoogleFonts.poppins(fontSize: 13.5),
      decoration: _inputDecoration(label, hint, icon),
    );
  }

  Widget _passwordField({
    required TextEditingController controller,
    required String label,
    required bool visible,
    required VoidCallback onVisibilityChanged,
    required String? Function(String?) validator,
    ValueChanged<String>? onSubmitted,
  }) {
    return TextFormField(
      controller: controller,
      obscureText: !visible,
      enabled: !_submitting,
      autofillHints: const [AutofillHints.password],
      validator: validator,
      autovalidateMode: AutovalidateMode.onUserInteraction,
      onFieldSubmitted: onSubmitted,
      style: GoogleFonts.poppins(fontSize: 13.5),
      decoration: _inputDecoration(
        label,
        'Enter your password',
        Icons.lock_outline_rounded,
      ).copyWith(
        suffixIcon: IconButton(
          onPressed: onVisibilityChanged,
          icon: Icon(
            visible
                ? Icons.visibility_off_outlined
                : Icons.visibility_outlined,
          ),
        ),
      ),
    );
  }

  InputDecoration _inputDecoration(
    String label,
    String hint,
    IconData icon,
  ) {
    return InputDecoration(
      labelText: label,
      hintText: hint,
      prefixIcon: Icon(icon),
      filled: true,
      fillColor: Theme.of(context)
          .colorScheme
          .surfaceContainerHighest
          .withValues(alpha: 0.52),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(14),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(14),
        borderSide: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(14),
        borderSide: const BorderSide(
          color: AppTheme.kPrimaryDeep,
          width: 1.5,
        ),
      ),
    );
  }

  Widget _primaryButton({
    required String text,
    required IconData icon,
    required VoidCallback onPressed,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: ElevatedButton.icon(
        onPressed: _submitting ? null : onPressed,
        icon: _submitting
            ? const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : Icon(icon, size: 18),
        label: Text(
          _submitting ? 'Please wait…' : text,
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.kPrimaryDeep,
          foregroundColor: Colors.white,
          disabledBackgroundColor: AppTheme.kPrimaryDeep.withValues(alpha: 0.55),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
    );
  }

  Widget _buildLogo() {
    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: AppTheme.kPrimaryDeep.withValues(alpha: 0.12),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: const Icon(
        Icons.spa_rounded,
        size: 38,
        color: AppTheme.kPrimaryDeep,
      ),
    );
  }
}
