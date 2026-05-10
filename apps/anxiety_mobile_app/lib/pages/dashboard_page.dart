import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';

import '../theme/app_theme.dart';
import '../background_service_helper.dart';
import '../services/notification_helper.dart';
import '../services/rating_settings.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import '../ema_and_gad7.dart';
import '../profile_page.dart';
import '../main.dart';

class DashboardPage extends StatefulWidget {
  final String? userId;
  const DashboardPage({super.key, this.userId});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage>
    with TickerProviderStateMixin {
  String _cachedId = "";
  double _currentPressure = 0.0;
  bool _isPressed = false;
  DateTime? _lastSentAt;
  bool _isServiceRunning = false;
  bool _isOptimized = false;

  // Vitals State
  int _heartRate = 72;
  double _breathingRate = 16.0;
  double _bodyTemp = 36.6;
  double _motion = 0.02;
  Timer? _vitalsTimer;

  late AnimationController _breatheController;
  late Animation<double> _breatheAnimation;

  @override
  void initState() {
    super.initState();
    _loadCachedId();

    _breatheController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat(reverse: true);

    _breatheAnimation = Tween<double>(begin: 1.0, end: 1.15).animate(
      CurvedAnimation(parent: _breatheController, curve: Curves.easeInOutQuad),
    );

    // Setup notification click listener
    NotificationHelper.onNotificationClick = (payload) {
      if (mounted) {
        if (payload != null && payload.startsWith('ema_rating_')) {
          final period = payload.replaceFirst('ema_rating_', '');
          showEmaSheet(period);
        } else if (payload == 'gad7_weekly') {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const Gad7Screen()),
          );
        } else if (payload == 'pss10_monthly') {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const Pss10Screen()),
          );
        } else {
          showRatingDialog();
        }
      }
    };

    _startStatusCheck();
    _startVitalsSimulation();
  }

  void _startVitalsSimulation() {
    _vitalsTimer = Timer.periodic(const Duration(seconds: 3), (timer) {
      if (!mounted) return;
      setState(() {
        // Simple random walk simulation
        final random = Random();
        _heartRate = (70 + random.nextInt(15));
        _breathingRate = (14 + random.nextDouble() * 4);
        _bodyTemp = (36.4 + random.nextDouble() * 0.5);
        _motion = random.nextDouble() * 0.1;
      });
    });
  }

  void _startStatusCheck() {
    Timer.periodic(const Duration(seconds: 10), (timer) async {
      if (!mounted) {
        timer.cancel();
        return;
      }
      final isRunning = await FlutterBackgroundService().isRunning();
      final optimized = await Permission.ignoreBatteryOptimizations.isDenied;

      if (mounted) {
        setState(() {
          _isServiceRunning = isRunning;
          _isOptimized = optimized;
        });

        // Auto-restart logic
        if (!isRunning) {
          FlutterBackgroundService().startService();
        }
      }
    });
  }

  @override
  void dispose() {
    _breatheController.dispose();
    _vitalsTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadCachedId() async {
    if (widget.userId != null) {
      setState(() => _cachedId = widget.userId!);
    } else {
      String id = await BackgroundServiceHelper.getCachedId();
      if (mounted) setState(() => _cachedId = id);
    }
  }

  Future<void> showRatingDialog() async {
    final prefs = await SharedPreferences.getInstance();
    String uid = prefs.getString('user_id') ?? "No_User_ID";

    int? selected = await showDialog<int>(
      context: context,
      builder: (ctx) {
        return SimpleDialog(
          title: const Text('Daily Check-in'),
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Wrap(
                alignment: WrapAlignment.center,
                spacing: 12,
                children: List.generate(6, (i) {
                  return InkWell(
                    onTap: () => Navigator.pop(ctx, i),
                    borderRadius: BorderRadius.circular(10),
                    child: Container(
                      padding: const EdgeInsets.all(15),
                      decoration: BoxDecoration(
                        color: AppTheme.kBgBottom,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        "$i",
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  );
                }),
              ),
            ),
          ],
        );
      },
    );

    if (selected != null) {
      await BackgroundServiceHelper.sendToSheet(
        uid,
        "Stress_Rating",
        selected.toString(),
      );
      String today = DateFormat('yyyy-MM-dd').format(DateTime.now());
      await prefs.setString('last_rating_submitted', today);
    }
  }

  void showEmaSheet(String period) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => EmaRatingSheet(timePeriod: period),
    );
  }

  void _handleTouch(PointerEvent event, bool isPressed) async {
    setState(() {
      _isPressed = isPressed;
      // Fallback for screens without pressure sensitivity
      _currentPressure = isPressed
          ? (event.pressure == 0 || event.pressure == 1.0
                ? 0.5
                : event.pressure)
          : 0.0;
    });

    if (isPressed) {
      _breatheController.stop();
      final now = DateTime.now();
      bool enoughTime =
          _lastSentAt == null ||
          now.difference(_lastSentAt!) >= const Duration(milliseconds: 300);

      if (enoughTime) {
        _lastSentAt = now;
        await BackgroundServiceHelper.sendToSheet(
          _cachedId,
          "Touch_Event",
          "Pressure:${_currentPressure.toStringAsFixed(2)}",
        );
      }
    } else {
      _breatheController.repeat(reverse: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.person_outline_rounded, color: AppTheme.kTextDark),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const ProfilePage()),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.settings_rounded, color: AppTheme.kTextDark),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const RatingSettingsPage()),
            ),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFFE0C3FC), Color(0xFF8EC5FC)],
            stops: [0.2, 1.0],
          ),
        ),
        child: SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_isOptimized)
                GestureDetector(
                  onTap: () => openAppSettings(),
                  child: Container(
                    width: double.infinity,
                    margin: const EdgeInsets.symmetric(horizontal: 20),
                    padding: const EdgeInsets.symmetric(
                      vertical: 10,
                      horizontal: 16,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.orangeAccent.withOpacity(0.9),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.warning_amber_rounded, color: Colors.white),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            "Battery optimization is active. Tap to set to 'Unrestricted' for continuous recording.",
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              const Spacer(flex: 2),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 12,
                    height: 12,
                    decoration: BoxDecoration(
                      color: _isServiceRunning ? Colors.greenAccent : Colors.redAccent,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: (_isServiceRunning ? Colors.greenAccent : Colors.redAccent).withOpacity(0.5),
                          blurRadius: 8,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    _isServiceRunning ? "System Active & Recording" : "System Inactive - Restarting...",
                    style: GoogleFonts.poppins(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: Colors.white.withOpacity(0.9),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Text(
                "How are you feeling?",
                style: GoogleFonts.poppins(
                  fontSize: 24,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                "Hold the orb to match your anxiety level.",
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  color: Colors.white.withOpacity(0.9),
                ),
              ),
              const SizedBox(height: 20),
              // Vitals Monitor Grid
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24.0),
                child: Column(
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: _buildVitalCard(
                            "Heart Rate",
                            "$_heartRate",
                            "BPM",
                            Icons.favorite_rounded,
                            Colors.pinkAccent,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _buildVitalCard(
                            "Breathing",
                            _breathingRate.toStringAsFixed(1),
                            "RPM",
                            Icons.air_rounded,
                            Colors.tealAccent,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: _buildVitalCard(
                            "Body Temp",
                            _bodyTemp.toStringAsFixed(1),
                            "°C",
                            Icons.thermostat_rounded,
                            Colors.orangeAccent,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _buildVitalCard(
                            "Motion",
                            _motion.toStringAsFixed(2),
                            "G",
                            Icons.directions_run_rounded,
                            Colors.cyanAccent,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const Spacer(flex: 3),
              Listener(
                onPointerDown: (e) => _handleTouch(e, true),
                onPointerMove: (e) => _handleTouch(e, true),
                onPointerUp: (e) => _handleTouch(e, false),
                onPointerCancel: (e) => _handleTouch(e, false),
                child: AnimatedBuilder(
                  animation: _breatheController,
                  builder: (context, child) {
                    double scale = _isPressed
                        ? 0.9 + (_currentPressure * 0.3)
                        : _breatheAnimation.value;

                    return Transform.scale(
                      scale: scale,
                      child: Container(
                        width: 280,
                        height: 280,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: _isPressed
                                ? [
                                    const Color(0xFFFF9A9E),
                                    const Color(0xFFFECFEF),
                                  ]
                                : [
                                    Colors.white.withOpacity(0.9),
                                    Colors.white.withOpacity(0.4),
                                  ],
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.white.withOpacity(0.4),
                              blurRadius: 40,
                              spreadRadius: 10,
                            ),
                            BoxShadow(
                              color: AppTheme.kPrimaryDeep.withOpacity(0.2),
                              blurRadius: 60,
                              spreadRadius: 5,
                              offset: const Offset(0, 20),
                            ),
                          ],
                        ),
                        child: Center(
                          child: Icon(
                            _isPressed ? Icons.favorite : Icons.fingerprint,
                            size: 60,
                            color: _isPressed
                                ? Colors.white
                                : AppTheme.kPrimaryDeep.withOpacity(0.5),
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const Spacer(flex: 3),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 10,
                ),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(30),
                  border: Border.all(color: Colors.white.withOpacity(0.3)),
                ),
                child: Text(
                  _cachedId.isNotEmpty
                      ? "ID: $_cachedId • Monitoring Active"
                      : "Initializing...",
                  style: GoogleFonts.poppins(fontSize: 12, color: Colors.white),
                ),
              ),
              const SizedBox(height: 30),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildVitalCard(String label, String value, String unit, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.2)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.poppins(
                    fontSize: 10,
                    color: Colors.white.withOpacity(0.7),
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    Text(
                      value,
                      style: GoogleFonts.poppins(
                        fontSize: 18,
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(width: 2),
                    Text(
                      unit,
                      style: GoogleFonts.poppins(
                        fontSize: 9,
                        color: Colors.white.withOpacity(0.5),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
