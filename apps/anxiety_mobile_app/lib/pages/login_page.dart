import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:usage_stats/usage_stats.dart';

import '../theme/app_theme.dart';
import '../background_service.dart';
import '../profile_page.dart';
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
    if (_idController.text.isEmpty) return;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user_id', _idController.text);

    // Initialize the background service
    await initializeService();

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
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppTheme.kBgTop, AppTheme.kBgBottom],
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
                  "Welcome",
                  style: GoogleFonts.poppins(
                    fontSize: 28,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.kTextDark,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  "Let's set up your safe space.",
                  textAlign: TextAlign.center,
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    color: AppTheme.kTextLight,
                  ),
                ),
                const SizedBox(height: 50),
                if (!_permissionsGranted)
                  _buildGlassButton(
                    text: "Grant Access",
                    icon: Icons.fingerprint,
                    isLoading: _isLoading,
                    onTap: _requestPermissions,
                  ),
                if (_permissionsGranted) ...[
                  _buildInputField(),
                  const SizedBox(height: 20),
                  _buildGlassButton(
                    text: "Begin Session",
                    icon: Icons.arrow_forward_rounded,
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
                        builder: (_) => const InformedConsentPage(),
                      ),
                    );
                  },
                  child: Text(
                    "Review Informed Consent & Privacy",
                    style: TextStyle(
                      color: AppTheme.kTextLight,
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
        color: Colors.white,
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
          color: Colors.white,
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
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 18, letterSpacing: 2),
          decoration: InputDecoration(
            hintText: "Enter Participant ID",
            hintStyle: TextStyle(color: Colors.grey.shade400),
            border: InputBorder.none,
            contentPadding: const EdgeInsets.all(20),
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
          color: isPrimary ? null : Colors.white,
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
              Icon(icon, color: isPrimary ? Colors.white : AppTheme.kTextDark),
              const SizedBox(width: 12),
              Text(
                text,
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: isPrimary ? Colors.white : AppTheme.kTextDark,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
