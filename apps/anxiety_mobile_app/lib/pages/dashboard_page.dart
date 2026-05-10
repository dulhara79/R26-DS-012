import 'dart:async';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../theme/app_theme.dart';
import '../background_service_helper.dart';
import '../ema_and_gad7.dart';
import '../services/background/background_service.dart';

class DashboardPage extends StatefulWidget {
  final String userId;
  const DashboardPage({super.key, required this.userId});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  double _currentPressure = 0.0;
  bool _isPressed = false;
  DateTime? _lastSentAt;
  double _lastPressureSent = 0.0;
  static const Duration _minSendInterval = Duration(milliseconds: 300);
  static const double _minPressureDelta = 0.03;
  bool _isServiceRunning = false;
  bool _isBatteryOptimized = false;
  Timer? _statusTimer;

  @override
  void initState() {
    super.initState();
    _checkSystemStatus();
    _statusTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _checkSystemStatus(),
    );
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkGad7OnOpen());
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    super.dispose();
  }

  Future<void> _checkSystemStatus() async {
    final running = await BackgroundServiceHelper.isServiceRunning();
    final optimized = await Permission.ignoreBatteryOptimizations.isDenied;

    if (mounted) {
      setState(() {
        _isServiceRunning = running;
        _isBatteryOptimized = optimized;
      });
    }

    // Auto-restart if dead
    if (!running) {
      try {
        await initializeService();
      } catch (e) {
        debugPrint('Auto-restart service failed: $e');
      }
    }
  }

  Future<void> _fixBatteryOptimization() async {
    await Permission.ignoreBatteryOptimizations.request();
    _checkSystemStatus();
  }

  Future<void> _checkGad7OnOpen() async {
    if (await isGad7DueThisWeek()) {
      await Future.delayed(const Duration(seconds: 2));
      if (mounted) {
        final shouldOpen = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Weekly Assessment Due'),
            content: const Text(
              'Your weekly GAD-7 anxiety questionnaire is ready. It takes about 2 minutes.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Later'),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('Start Now'),
              ),
            ],
          ),
        );
        if (shouldOpen == true && mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const Gad7Screen()),
          );
        }
      }
    }
  }

  void _handleTouch(PointerEvent event, bool isPressed) async {
    setState(() {
      _isPressed = isPressed;
      _currentPressure = isPressed ? event.pressure : 0.0;
    });
    if (isPressed) {
      final now = DateTime.now();
      bool enoughTime =
          _lastSentAt == null ||
          now.difference(_lastSentAt!) >= _minSendInterval;
      bool enoughDelta =
          (event.pressure - _lastPressureSent).abs() >= _minPressureDelta;
      if (enoughTime && enoughDelta) {
        _lastSentAt = now;
        _lastPressureSent = event.pressure;
        await BackgroundServiceHelper.sendToSheet(
          widget.userId,
          "Touch_Event",
          "Pressure:${event.pressure.toStringAsFixed(2)}",
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Monitoring Dashboard"),
        actions: [
          IconButton(
            icon: const Icon(Icons.assignment_outlined),
            tooltip: 'Weekly GAD-7',
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const Gad7Screen()),
            ),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: _isServiceRunning
                    ? const Color(0xFFE8F5E9)
                    : const Color(0xFFFFEBEE),
                borderRadius: BorderRadius.circular(30),
                border: Border.all(
                  color: _isServiceRunning
                      ? Colors.green.shade200
                      : Colors.red.shade200,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      color: _isServiceRunning ? Colors.green : Colors.red,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _isServiceRunning
                        ? "System Active & Recording"
                        : "System Inactive - Restarting...",
                    style: TextStyle(
                      color: _isServiceRunning
                          ? Colors.green.shade800
                          : Colors.red.shade800,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            if (_isBatteryOptimized) ...[
              const SizedBox(height: 12),
              GestureDetector(
                onTap: _fixBatteryOptimization,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.orange.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.orange.shade200),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.warning_amber_rounded, color: Colors.orange),
                      SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          "Battery optimization is ON. This may stop data collection. Tap to fix.",
                          style: TextStyle(fontSize: 13, color: Colors.orange),
                        ),
                      ),
                      Icon(
                        Icons.chevron_right,
                        color: Colors.orange,
                        size: 20,
                      ),
                    ],
                  ),
                ),
              ),
            ],
            const Spacer(),
            const Text(
              "Anxiety Event Recorder",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              "If you feel anxious, press and hold the sensor below. Vary your pressure to match the intensity.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey.shade600, fontSize: 15),
            ),
            const SizedBox(height: 40),
            Listener(
              onPointerDown: (e) => _handleTouch(e, true),
              onPointerMove: (e) => _handleTouch(e, true),
              onPointerUp: (e) => _handleTouch(e, false),
              onPointerCancel: (e) => _handleTouch(e, false),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 100),
                width: 260,
                height: 260,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: _isPressed
                        ? [AppTheme.kPrimaryDeep, AppTheme.kAccentBlue]
                        : [Colors.white, Colors.grey.shade100],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: AppTheme.kPrimaryDeep.withValues(
                        alpha: _isPressed ? 0.4 : 0.1,
                      ),
                      blurRadius: _isPressed ? 30 : 20,
                      spreadRadius: _isPressed ? 5 : 0,
                      offset: const Offset(0, 10),
                    ),
                    if (!_isPressed)
                      const BoxShadow(
                        color: Colors.white,
                        blurRadius: 10,
                        offset: Offset(-5, -5),
                      ),
                  ],
                  border: Border.all(
                    color: _isPressed ? AppTheme.kPrimaryDeep : Colors.grey.shade200,
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.fingerprint,
                      size: 60,
                      color: _isPressed
                          ? Colors.white
                          : AppTheme.kPrimaryDeep.withValues(alpha: 0.5),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      _isPressed
                          ? "${(_currentPressure * 100).toInt()}% Intensity"
                          : "Press Here",
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: _isPressed ? Colors.white : Colors.grey.shade500,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const Spacer(),
            Text(
              "ID: ${widget.userId}",
              style: TextStyle(color: Colors.grey.shade400, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
