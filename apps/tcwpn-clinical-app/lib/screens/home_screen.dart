import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';
import '../widgets/patient_card.dart';
import '../widgets/risk_badge.dart';
import 'patient_detail_screen.dart';
import 'patient_list_screen.dart';
import 'support_set_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _tab = 0;

  final List<Widget> _screens = const [
    _DashboardTab(),
    PatientListScreen(),
    SupportSetScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _tab, children: _screens),
      bottomNavigationBar: Container(
        decoration: const BoxDecoration(
          border: Border(top: BorderSide(color: AppColors.border, width: 0.8)),
        ),
        child: BottomNavigationBar(
          currentIndex: _tab,
          onTap: (i) => setState(() => _tab = i),
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.dashboard_outlined),
              activeIcon: Icon(Icons.dashboard_rounded),
              label: 'Dashboard',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.people_outline_rounded),
              activeIcon: Icon(Icons.people_rounded),
              label: 'Patients',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.dataset_outlined),
              activeIcon: Icon(Icons.dataset_rounded),
              label: 'Support set',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings_outlined),
              activeIcon: Icon(Icons.settings_rounded),
              label: 'Settings',
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Dashboard tab ────────────────────────────────────────────────────────────
class _DashboardTab extends StatelessWidget {
  const _DashboardTab();

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<PatientProvider>();
    final alerts   = provider.alertPatients;

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      body: CustomScrollView(
        slivers: [
          // App bar with greeting
          SliverAppBar(
            expandedHeight: 120,
            pinned: true,
            backgroundColor: AppColors.primary,
            flexibleSpace: FlexibleSpaceBar(
              titlePadding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
              title: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Good morning, Dr. Kaushalya',
                    style: const TextStyle(
                      fontSize: 13, color: Colors.white,
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                  Text(
                    'Today\'s overview',
                    style: const TextStyle(
                      fontSize: 18, color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            actions: [
              IconButton(
                icon: Stack(
                  children: [
                    const Icon(Icons.notifications_outlined,
                        color: Colors.white, size: 24),
                    if (alerts.isNotEmpty)
                      Positioned(
                        right: 0, top: 0,
                        child: Container(
                          width: 10, height: 10,
                          decoration: const BoxDecoration(
                            color: AppColors.riskHigh,
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                  ],
                ),
                onPressed: () {},
              ),
              const SizedBox(width: 4),
            ],
          ),

          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([

                // Summary metrics
                Row(
                  children: [
                    _MetricCard(
                      label: 'Total patients',
                      value: '${provider.patients.length}',
                      icon: Icons.people_rounded,
                      color: AppColors.primary,
                    ),
                    const SizedBox(width: 10),
                    _MetricCard(
                      label: 'Alerts today',
                      value: '${alerts.length}',
                      icon: Icons.warning_amber_rounded,
                      color: alerts.isEmpty
                          ? AppColors.riskLow
                          : AppColors.riskHigh,
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    _MetricCard(
                      label: 'Support notes',
                      value: '${provider.supportNotes.length}',
                      icon: Icons.dataset_rounded,
                      color: AppColors.info,
                    ),
                    const SizedBox(width: 10),
                    _MetricCard(
                      label: 'Model AUROC',
                      value: '0.963',
                      icon: Icons.analytics_rounded,
                      color: AppColors.primaryLight,
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Alerts section
                if (alerts.isNotEmpty) ...[
                  Row(
                    children: [
                      Container(
                        width: 4, height: 18,
                        decoration: BoxDecoration(
                          color: AppColors.riskHigh,
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        'Requires attention',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: AppColors.riskHighBg,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Text(
                          '${alerts.length}',
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: AppColors.riskHigh,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  ...alerts.map((p) => Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: PatientCard(
                      patient: p,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => PatientDetailScreen(patient: p),
                        ),
                      ),
                    ),
                  )),
                  const SizedBox(height: 12),
                ],

                // All patients
                Row(
                  children: [
                    Container(
                      width: 4, height: 18,
                      decoration: BoxDecoration(
                        color: AppColors.primary,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      'All patients',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                ...provider.patients.map((p) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: PatientCard(
                    patient: p,
                    onTap: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => PatientDetailScreen(patient: p),
                      ),
                    ),
                  ),
                )),
                const SizedBox(height: 24),

                // Model disclaimer
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppColors.infoBg,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                        color: AppColors.info.withOpacity(0.25), width: 0.8),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(Icons.info_outline_rounded,
                          size: 16, color: AppColors.info),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'TC-WPN v1.0 · MIMIC-IV + MIMIC-III meta-training · '
                          'NHSL adaptation pending. Clinical decision support only.',
                          style: const TextStyle(
                            fontSize: 12, color: AppColors.info, height: 1.5,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;

  const _MetricCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border, width: 0.8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 20, color: color),
            const SizedBox(height: 10),
            Text(
              value,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
