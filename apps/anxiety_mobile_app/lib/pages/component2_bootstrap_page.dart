import 'dart:async';

import 'package:flutter/material.dart';

import '../services/background_service_helper.dart';
import '../services/component2_data_service.dart';
import 'participant_behavior_page.dart';

/// Syncs Component 2 display-safe data before opening the participant-facing
/// behavioural page. Network/API failures never block the UI: the page falls
/// back to cached data or the honest baseline-building state.
///
/// The page also re-syncs when the app returns to the foreground. This prevents
/// baseline/data-quality progress from remaining stale for participants who
/// keep the app installed for several days without recreating this tab.
class Component2BootstrapPage extends StatefulWidget {
  final String? userId;

  const Component2BootstrapPage({super.key, this.userId});

  @override
  State<Component2BootstrapPage> createState() =>
      _Component2BootstrapPageState();
}

class _Component2BootstrapPageState extends State<Component2BootstrapPage>
    with WidgetsBindingObserver {
  late Future<void> _bootstrapFuture;
  bool _refreshing = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _bootstrapFuture = _sync();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  Future<void> _sync() async {
    final participantId =
        widget.userId ?? await BackgroundServiceHelper.getCachedId();
    await Component2DataService.sync(participantId);
  }

  Future<void> _refreshFromBackend() async {
    if (_refreshing || !mounted) return;
    _refreshing = true;
    final next = _sync();
    setState(() => _bootstrapFuture = next);
    try {
      await next;
    } finally {
      _refreshing = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(_refreshFromBackend());
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _bootstrapFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return Scaffold(
            backgroundColor: Theme.of(context).scaffoldBackgroundColor,
            body: Center(
              child: CircularProgressIndicator(
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
          );
        }

        return ParticipantBehaviorPage(
          // A completed remote sync gets a new Future object, so this key
          // recreates the child and makes it reload the refreshed cache.
          key: ValueKey(_bootstrapFuture),
          userId: widget.userId,
        );
      },
    );
  }
}
