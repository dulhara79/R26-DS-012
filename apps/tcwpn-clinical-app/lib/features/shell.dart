// lib/features/shell.dart
//
// ONE shell. The server-backed Dashboard is the primary clinician worklist.
// Current assessment, forecast, assignments and attention-event identity come
// from the Central Backend; this shell never recalculates patient risk.

import 'dart:async';

import 'package:flutter/material.dart';

import '../core/notifications/attention_notification_gateway.dart';
import '../core/notifications/flutter_attention_notification_gateway.dart';
import '../data/api/session.dart';
import '../data/local/attention_notification_store.dart';
import '../data/repositories/central_backend_repositories.dart';
import '../state/attention_notification_controller.dart';
import '../state/dashboard_controller.dart';
import 'attention_events/activity_screen.dart';
import 'attention_events/attention_event_detail_screen.dart';
import 'dashboard/server_dashboard_screen.dart';
import 'evidence/ask_care_screen.dart';
import 'patients/patients_screen.dart';
import 'settings/settings_screen.dart';

class AppShell extends StatefulWidget {
  final DashboardController? dashboardController;
  final AttentionNotificationController? attentionNotificationController;
  final AttentionNotificationGateway? notificationGateway;
  final AttentionNotificationStore? notificationStore;

  const AppShell({
    super.key,
    this.dashboardController,
    this.attentionNotificationController,
    this.notificationGateway,
    this.notificationStore,
  });

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> with WidgetsBindingObserver {
  static const _pollInterval = Duration(seconds: 30);

  int _tab = 0;
  late final DashboardController _dashboardController;
  late final bool _ownsDashboardController;
  late final AttentionNotificationGateway _notificationGateway;
  late final AttentionNotificationStore _notificationStore;
  late final AttentionNotificationController _notificationController;
  StreamSubscription<String>? _notificationOpenSubscription;
  Timer? _notificationPollTimer;
  bool _notificationsEnabled = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _ownsDashboardController = widget.dashboardController == null;
    _dashboardController = widget.dashboardController ??
        DashboardController(
          repository: CentralBackendDashboardRepository(),
          authRepository: CentralBackendAuthRepository(),
        );

    _notificationGateway =
        widget.notificationGateway ?? attentionNotificationGateway;
    _notificationStore = widget.notificationStore ??
        SharedPreferencesAttentionNotificationStore(
          deliveryScope: Session.clinicianId ?? '__unknown_clinician__',
        );
    _notificationController = widget.attentionNotificationController ??
        AttentionNotificationController(
          repository: CentralBackendAttentionEventRepository(),
          store: _notificationStore,
          gateway: _notificationGateway,
        );

    _notificationOpenSubscription =
        _notificationGateway.openedEventIds.listen((eventId) {
      unawaited(_openNotificationEvent(eventId, consumePending: true));
    });

    if (_ownsDashboardController) {
      _dashboardController.load();
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_restorePendingNotificationOpen());
    });
    unawaited(_enableNotifications());
  }

  Future<void> _enableNotifications() async {
    try {
      final enabled = await _notificationGateway.requestPermission();
      if (!mounted) return;
      _notificationsEnabled = enabled;
      if (enabled) _startNotificationPolling();
    } catch (_) {
      // Activity remains available as the persistent server-backed event inbox.
      // Notification permission/plugin failure must never alter clinical state.
    }
  }

  void _startNotificationPolling() {
    if (!_notificationsEnabled) return;
    _notificationPollTimer?.cancel();
    unawaited(_notificationController.pollOnce());
    _notificationPollTimer = Timer.periodic(
      _pollInterval,
      (_) => unawaited(_notificationController.pollOnce()),
    );
  }

  void _stopNotificationPolling() {
    _notificationPollTimer?.cancel();
    _notificationPollTimer = null;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _startNotificationPolling();
    } else {
      _stopNotificationPolling();
    }
  }

  Future<void> _restorePendingNotificationOpen() async {
    final eventId = await _notificationStore.takePendingOpen();
    if (eventId == null || !mounted) return;
    await _openNotificationEvent(eventId, consumePending: false);
  }

  Future<void> _openNotificationEvent(
    String eventId, {
    required bool consumePending,
  }) async {
    final id = eventId.trim();
    if (id.isEmpty || !mounted) return;

    if (consumePending) {
      await _notificationStore.takePendingOpen();
    }
    if (!mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => AttentionEventDetailScreen.production(eventId: id),
      ),
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopNotificationPolling();
    unawaited(_notificationOpenSubscription?.cancel());
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
