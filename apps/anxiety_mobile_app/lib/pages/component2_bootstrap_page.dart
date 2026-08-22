import 'package:flutter/material.dart';

import '../services/background_service_helper.dart';
import '../services/component2_data_service.dart';
import 'participant_behavior_page.dart';

/// Syncs Component 2 display-safe data before opening the participant-facing
/// behavioural page. Network/API failures never block the UI: the page falls
/// back to cached data or the honest baseline-building state.
class Component2BootstrapPage extends StatefulWidget {
  final String? userId;

  const Component2BootstrapPage({super.key, this.userId});

  @override
  State<Component2BootstrapPage> createState() =>
      _Component2BootstrapPageState();
}

class _Component2BootstrapPageState extends State<Component2BootstrapPage> {
  late Future<void> _bootstrapFuture;

  @override
  void initState() {
    super.initState();
    _bootstrapFuture = _sync();
  }

  Future<void> _sync() async {
    final participantId =
        widget.userId ?? await BackgroundServiceHelper.getCachedId();
    await Component2DataService.sync(participantId);
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _bootstrapFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Scaffold(
            backgroundColor: Color(0xFFF7F5FF),
            body: Center(
              child: CircularProgressIndicator(color: Color(0xFF6D5BD0)),
            ),
          );
        }

        return ParticipantBehaviorPage(
          key: ValueKey(_bootstrapFuture),
          userId: widget.userId,
        );
      },
    );
  }
}
