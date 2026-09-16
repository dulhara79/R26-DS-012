// lib/features/shell.dart
//
// ONE shell. The server-backed Dashboard is the primary clinician worklist.
// Current assessment, forecast, assignments and attention-event identity come
// from the Central Backend; this shell never recalculates patient risk.

import 'package:flutter/material.dart';

import '../data/repositories/central_backend_repositories.dart';
import '../state/dashboard_controller.dart';
import 'attention_events/activity_screen.dart';
import 'dashboard/server_dashboard_screen.dart';
import 'evidence/ask_care_screen.dart';
import 'patients/patients_screen.dart';
import 'settings/settings_screen.dart';

class AppShell extends StatefulWidget {
  final DashboardController? dashboardController;

  const AppShell({
    super.key,
    this.dashboardController,
  });

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _tab = 0;
  late final DashboardController _dashboardController;
  late final bool _ownsDashboardController;

  @override
  void initState() {
    super.initState();
    _ownsDashboardController = widget.dashboardController == null;
    _dashboardController = widget.dashboardController ??
        DashboardController(
          repository: CentralBackendDashboardRepository(),
          authRepository: CentralBackendAuthRepository(),
        );

    if (_ownsDashboardController) {
      _dashboardController.load();
    }
  }

  @override
  void dispose() {
    if (_ownsDashboardController) {
      _dashboardController.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _tab,
        children: [
          ServerDashboardScreen(controller: _dashboardController),
          const PatientsScreen(),
          const ActivityScreen.production(),
          const SettingsScreen(),
          const AskCareScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard_outlined),
            selectedIcon: Icon(Icons.dashboard_rounded),
            label: 'Dashboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.folder_shared_outlined),
            selectedIcon: Icon(Icons.folder_shared_rounded),
            label: 'Patients',
          ),
          NavigationDestination(
            icon: Icon(Icons.notifications_none_rounded),
            selectedIcon: Icon(Icons.notifications_rounded),
            label: 'Activity',
          ),
          NavigationDestination(
            icon: Icon(Icons.tune_outlined),
            selectedIcon: Icon(Icons.tune_rounded),
            label: 'Settings',
          ),
          NavigationDestination(
            icon: Icon(Icons.psychology_outlined),
            selectedIcon: Icon(Icons.psychology),
            label: 'Ask CARE',
          ),
        ],
      ),
    );
  }
}
