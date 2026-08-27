import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// A beautiful welcome / splash screen that shows the Aura branding
/// with the meditation illustration, then automatically transitions
/// to [nextPage] after a short delay.
class WelcomeSplashPage extends StatefulWidget {
  final Widget nextPage;
  const WelcomeSplashPage({super.key, required this.nextPage});

  @override
  State<WelcomeSplashPage> createState() => _WelcomeSplashPageState();
}

class _WelcomeSplashPageState extends State<WelcomeSplashPage>
    with TickerProviderStateMixin {
  late final AnimationController _fadeController;
  late final AnimationController _slideController;
  late final Animation<Offset> _slideAnim;
  late final Animation<double> _imageFadeAnim;
  late final Animation<double> _textFadeAnim;

  @override
  void initState() {
    super.initState();

    // Main fade-in for the whole page
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    // Slide-up animation for content
    _slideController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );
    _slideAnim = Tween<Offset>(begin: const Offset(0, 0.15), end: Offset.zero)
        .animate(
          CurvedAnimation(parent: _slideController, curve: Curves.easeOutCubic),
        );

    // Staggered fades for image and text
    _imageFadeAnim = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _fadeController,
        curve: const Interval(0.0, 0.7, curve: Curves.easeOut),
      ),
    );
    _textFadeAnim = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _fadeController,
        curve: const Interval(0.3, 1.0, curve: Curves.easeOut),
      ),
    );

    _fadeController.forward();
    _slideController.forward();

    // Auto-navigate after 3.5 seconds
    Timer(const Duration(milliseconds: 3500), _navigateNext);
  }

  void _navigateNext() {
    if (!mounted) return;
    final oldRoute = ModalRoute.of(context);
    final newRoute = PageRouteBuilder(
      pageBuilder: (_, _, _) => widget.nextPage,
      transitionsBuilder: (_, a, _, c) => FadeTransition(opacity: a, child: c),
      transitionDuration: const Duration(milliseconds: 600),
    );

    if (oldRoute != null && !oldRoute.isCurrent) {
      // A modal bottom sheet or dialog is open on top of the splash page.
      // Replace the splash page silently underneath the modal route.
      Navigator.of(context).replace(oldRoute: oldRoute, newRoute: newRoute);
    } else {
      // No modal is open; perform normal animated transition.
      Navigator.pushReplacement(context, newRoute);
    }
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final dark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: dark
                ? const [
                    Color(0xFF111218),
                    Color(0xFF1A1B24),
                    Color(0xFF221E30),
                  ]
                : const [
                    Color(0xFFF3EEFF),
                    Color(0xFFE8E0F7),
                    Color(0xFFF0ECFF),
                  ],
          ),
        ),
        child: SafeArea(
          child: SlideTransition(
            position: _slideAnim,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Spacer(flex: 2),

                // ── Meditation illustration ──
                FadeTransition(
                  opacity: _imageFadeAnim,
                  child: Container(
                    width: 220,
                    height: 220,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(30),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(30),
                      child: Image.asset(
                        'assets/welcome_illustration.png',
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 40),

                // ── Welcome text ──
                FadeTransition(
                  opacity: _textFadeAnim,
                  child: Column(
                    children: [
                      Text(
                        'Welcome to',
                        style: GoogleFonts.poppins(
                          fontSize: 16,
                          fontWeight: FontWeight.w400,
                          color: colors.onSurfaceVariant,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 6),
                      ShaderMask(
                        shaderCallback: (bounds) => const LinearGradient(
                          colors: [
                            Color(0xFF7C5CBF),
                            Color(0xFF9B7FD4),
                            Color(0xFF6B4FA0),
                          ],
                        ).createShader(bounds),
                        child: Text(
                          'Aura',
                          style: GoogleFonts.poppins(
                            fontSize: 46,
                            fontWeight: FontWeight.w700,
                            color: Colors.white, // Masked by shader
                            letterSpacing: 2,
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: colors.primaryContainer.withValues(
                            alpha: 0.45,
                          ),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          'Your mindful companion',
                          style: GoogleFonts.poppins(
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                            color: colors.primary,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const Spacer(flex: 3),

                // ── Subtle loading dots ──
                FadeTransition(opacity: _textFadeAnim, child: _PulsingDots()),

                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Three softly pulsing dots to indicate loading
class _PulsingDots extends StatefulWidget {
  @override
  State<_PulsingDots> createState() => _PulsingDotsState();
}

class _PulsingDotsState extends State<_PulsingDots>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(3, (i) {
        return AnimatedBuilder(
          animation: _controller,
          builder: (_, _) {
            // Stagger each dot
            final double delay = i * 0.2;
            final double t = ((_controller.value - delay) % 1.0).clamp(
              0.0,
              1.0,
            );
            final double opacity = (1.0 - (2.0 * t - 1.0).abs()).clamp(
              0.3,
              1.0,
            );
            return Container(
              margin: const EdgeInsets.symmetric(horizontal: 4),
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Theme.of(
                  context,
                ).colorScheme.primary.withValues(alpha: opacity),
              ),
            );
          },
        );
      }),
    );
  }
}
