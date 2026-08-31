import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:usage_stats/usage_stats.dart';

import '../theme/app_theme.dart';
import '../background_service.dart';
import '../profile_page.dart';
import '../services/user_manager.dart';
import '../services/participant_identity_service.dart';
import 'informed_consent_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});
  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _idController = TextEditingController();
  bool _permissionsGranted = false;
  bool _isLoading = false;

  @override
  void dispose() {
    _idController.dispose();
    super.dispose();
  }

  Future<void> _requestPermissions() async {
    setState(() => _isLoading = true);
    await Future.delayed(const Duration(milliseconds: 800)); // UX Delay

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      await [
        Permission.location,
        Permission.locationAlways,
        Permission.phone,
        Permission.sms,
        Permission.notification,
        Permission.ignoreBatteryOptimizations,
        Permission.bluetoothScan,
        Permission.bluetoothConnect,
      ].request();

      bool isUsageGranted = await UsageStats.checkUsagePermission() ?? false;
      if (!isUsageGranted) {
        await UsageStats.grantUsagePermission();
      }
    }

    if (mounted) {
      setState(() {
        _permissionsGranted = true;
        _isLoading = false;
      });
    }
  }

  Future<void> _login() async {
    final displayName = _idController.text.trim();
    if (displayName.isEmpty) return;

    if (displayName.length > 80) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Please keep the display name under 80 characters.',
            style: GoogleFonts.poppins(fontSize: 13),
          ),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    // The entered name stays on this device. Research systems receive only a
    // cryptographically random participant code such as P_7F3A9C2E4B10D6C1.
    final participantId = await ParticipantIdentityService.createForDisplayName(
      displayName,
    );

    // A new login happens after main() has already run, so initialise the
    // physiological upload session here as well as in the cold-start path.
    UserManager().login(participantId);

    try {
      await startBackgroundServiceIfPermitted();
    } catch (e) {
      debugPrint('Background service init error: $e');
    }

    if (mounted) {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (_, _, _) => ProfilePage(),
          transitionsBuilder: (_, a, _, c) =>
              FadeTransition(opacity: a, child: c),
          transitionDuration: const Duration(milliseconds: 800),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isChoosingName = _permissionsGranted;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: Theme.of(context).brightness == Brightness.dark
                ? const [Color(0xFF111218), Color(0xFF1A1B24)]
                : const [AppTheme.kBgTop, AppTheme.kBgBottom],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildLogo(),
                const SizedBox(height: 30),
                Text(
                  isChoosingName ? "What should Aura call you?" : "Welcome to Aura",
                  textAlign: TextAlign.center,
                  style: GoogleFonts.poppins(
                    fontSize: 28,
                    fontWeight: FontWeight.w600,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  isChoosingName
                      ? "Choose the name you'd like to see in the app."
                      : "A quiet place to check in, understand your patterns, and take things one moment at a time.",
                  textAlign: TextAlign.center,
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 50),
                if (!_permissionsGranted)
                  _buildGlassButton(
                    text: "Make this space yours",
                    icon: Icons.spa_rounded,
                    isLoading: _isLoading,
                    onTap: _requestPermissions,
                  ),
                if (_permissionsGranted) ...[
                  _buildInputField(),
                  const SizedBox(height: 10),
                  Text(
                    'You can change this name later. Aura creates a separate random Participant ID for research records.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 12,
                      height: 1.4,
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 20),
                  _buildGlassButton(
                    text: "Continue",
                    icon: Icons.check_rounded,
                    isPrimary: true,
                    onTap: _login,
                  ),
                ],
                const SizedBox(height: 40),
                TextButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) =>
                            const InformedConsentPage(readOnly: true),
                      ),
                    );
                  },
                  child: Text(
                    "Review Informed Consent & Privacy",
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                      fontSize: 12,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLogo() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: AppTheme.kPrimaryDeep.withValues(alpha: 0.1),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: const Icon(
        Icons.spa_rounded,
        size: 40,
        color: AppTheme.kPrimaryDeep,
      ),
    );
  }

  Widget _buildInputField() {
    return AnimatedOpacity(
      opacity: _permissionsGranted ? 1.0 : 0.0,
      duration: const Duration(milliseconds: 500),
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.05),
              blurRadius: 15,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: TextField(
          controller: _idController,
          textInputAction: TextInputAction.done,
          onSubmitted: (_) => _login(),
          style: TextStyle(
            fontSize: 16,
            color: Theme.of(context).colorScheme.onSurface,
          ),
          decoration: InputDecoration(
            labelText: "Display name",
            hintText: "Enter a name",
            hintStyle: TextStyle(color: Colors.grey.shade400),
            prefixIcon: const Icon(Icons.person_outline_rounded),
            border: InputBorder.none,
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 20,
              vertical: 18,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildGlassButton({
    required String text,
    required IconData icon,
    required VoidCallback onTap,
    bool isLoading = false,
    bool isPrimary = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 60,
        decoration: BoxDecoration(
          gradient: isPrimary
              ? const LinearGradient(
                  colors: [AppTheme.kAccentBlue, AppTheme.kPrimaryDeep],
                )
              : null,
          color: isPrimary ? null : Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: isPrimary
                  ? AppTheme.kPrimaryDeep.withValues(alpha: 0.3)
                  : Colors.black.withValues(alpha: 0.05),
              blurRadius: 20,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isLoading)
              const SizedBox(
                height: 20,
                width: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: AppTheme.kPrimaryDeep,
                ),
              )
            else ...[
              Icon(
                icon,
                color: isPrimary
                    ? Colors.white
                    : Theme.of(context).colorScheme.onSurface,
              ),
              const SizedBox(width: 12),
              Text(
                text,
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: isPrimary
                      ? Colors.white
                      : Theme.of(context).colorScheme.onSurface,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
