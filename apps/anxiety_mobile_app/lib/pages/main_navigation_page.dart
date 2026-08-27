import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../profile_page.dart';
import 'home_page.dart';
import 'dashboard_page.dart';
import 'component2_bootstrap_page.dart';
import 'longitudinal_context_page.dart';

class MainNavigationPage extends StatefulWidget {
  final String? userId;
  const MainNavigationPage({super.key, this.userId});

  @override
  State<MainNavigationPage> createState() => _MainNavigationPageState();
}

class _MainNavigationPageState extends State<MainNavigationPage> {
  int _currentIndex = 0;

  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    final userId = widget.userId ?? '';
    _pages = [
      HomePage(userId: widget.userId),
      DashboardPage(userId: widget.userId),
      Component2BootstrapPage(userId: widget.userId),
      LongitudinalContextPage(userId: userId),
      ProfilePage(isTab: true),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _currentIndex, children: _pages),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.08),
              blurRadius: 20,
              offset: const Offset(0, -4),
            ),
          ],
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(
                  0,
                  Icons.home_rounded,
                  Icons.home_outlined,
                  'Home',
                ),
                _buildNavItem(
                  1,
                  Icons.monitor_heart_rounded,
                  Icons.monitor_heart_outlined,
                  'Body',
                ),
                _buildNavItem(
                  2,
                  Icons.psychology_rounded,
                  Icons.psychology_outlined,
                  'Activity',
                ),
                _buildNavItem(
                  3,
                  Icons.fact_check_rounded,
                  Icons.fact_check_outlined,
                  'Check-ins',
                ),
                _buildNavItem(
                  4,
                  Icons.person_rounded,
                  Icons.person_outline_rounded,
                  'Profile',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(
    int index,
    IconData activeIcon,
    IconData inactiveIcon,
    String label,
  ) {
    final bool isActive = _currentIndex == index;

    final dark = Theme.of(context).brightness == Brightness.dark;
    final List<Color> gradients = dark
        ? [
            const Color(0xFF9AAEFF), // Home
            const Color(0xFFC5A1E8), // Physio
            Theme.of(context).colorScheme.primary, // Phenotype
            const Color(0xFF66D5B1), // Check-ins
            Theme.of(context).colorScheme.primary, // Profile
          ]
        : [
            const Color(0xFF667eea), // Home
            const Color(0xFF764ba2), // Physio
            const Color(0xFF5E60CE), // Phenotype
            const Color(0xFF2D9C79), // Check-ins
            const Color(0xFF5E60CE), // Profile
          ];

    return GestureDetector(
      onTap: () => setState(() => _currentIndex = index),
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOutCubic,
        padding: EdgeInsets.symmetric(
          horizontal: isActive ? 11 : 9,
          vertical: 8,
        ),
        decoration: BoxDecoration(
          color: isActive
              ? gradients[index].withValues(alpha: 0.12)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: Icon(
                isActive ? activeIcon : inactiveIcon,
                key: ValueKey(isActive),
                color: isActive ? gradients[index] : Colors.grey.shade400,
                size: 23,
              ),
            ),
            if (isActive) ...[
              const SizedBox(width: 5),
              Text(
                label,
                style: GoogleFonts.poppins(
                  fontSize: 10.5,
                  fontWeight: FontWeight.w600,
                  color: gradients[index],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
