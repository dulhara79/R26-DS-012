import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../theme/app_theme.dart';
import 'home_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _idCtrl   = TextEditingController(text: 'DR001');
  final _pwCtrl   = TextEditingController(text: '••••••••');
  bool _obscure   = true;
  bool _loading   = false;

  Future<void> _login() async {
    setState(() => _loading = true);
    await Future.delayed(const Duration(milliseconds: 1200));
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const HomeScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.primary,
      body: SafeArea(
        child: Column(
          children: [
            // Top branding area
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 56,
                      height: 56,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      alignment: Alignment.center,
                      child: const Icon(
                        Icons.monitor_heart_rounded,
                        color: Colors.white,
                        size: 28,
                      ),
                    )
                        .animate()
                        .fadeIn(duration: 400.ms)
                        .slideY(begin: -0.3, curve: Curves.easeOut),
                    const SizedBox(height: 20),
                    Text(
                      'TC-WPN',
                      style: Theme.of(context).textTheme.displayLarge?.copyWith(
                        color: Colors.white,
                        letterSpacing: -1,
                      ),
                    ).animate().fadeIn(delay: 100.ms, duration: 400.ms),
                    const SizedBox(height: 8),
                    Text(
                      'Clinical anxiety detection\nfor psychiatry teams',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Colors.white.withOpacity(0.75),
                        height: 1.4,
                      ),
                    ).animate().fadeIn(delay: 200.ms, duration: 400.ms),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        'AUROC 0.963 · 3-shot adaptation',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withOpacity(0.9),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ).animate().fadeIn(delay: 300.ms, duration: 400.ms),
                  ],
                ),
              ),
            ),

            // Login card
            Container(
              decoration: const BoxDecoration(
                color: AppColors.surfaceSecond,
                borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
              ),
              padding: const EdgeInsets.fromLTRB(24, 32, 24, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    'Sign in',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'National Hospital of Sri Lanka',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  const SizedBox(height: 24),

                  // Clinician ID
                  TextField(
                    controller: _idCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Clinician ID',
                      prefixIcon: Icon(Icons.badge_outlined, size: 20),
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Password
                  TextField(
                    controller: _pwCtrl,
                    obscureText: _obscure,
                    decoration: InputDecoration(
                      labelText: 'Password',
                      prefixIcon: const Icon(Icons.lock_outline_rounded, size: 20),
                      suffixIcon: IconButton(
                        icon: Icon(
                          _obscure ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                          size: 20,
                        ),
                        onPressed: () => setState(() => _obscure = !_obscure),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Login button
                  ElevatedButton(
                    onPressed: _loading ? null : _login,
                    child: _loading
                        ? const SizedBox(
                            height: 20, width: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white,
                            ),
                          )
                        : const Text('Sign in'),
                  ),
                  const SizedBox(height: 16),

                  // Disclaimer
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppColors.infoBg,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: AppColors.info.withOpacity(0.2)),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.info_outline_rounded,
                            size: 16, color: AppColors.info),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Clinical decision support only. All clinical decisions '
                            'remain with the responsible psychiatrist.',
                            style: TextStyle(
                              fontSize: 12,
                              color: AppColors.info,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ).animate().slideY(begin: 0.2, duration: 500.ms, curve: Curves.easeOut),
          ],
        ),
      ),
    );
  }
}
