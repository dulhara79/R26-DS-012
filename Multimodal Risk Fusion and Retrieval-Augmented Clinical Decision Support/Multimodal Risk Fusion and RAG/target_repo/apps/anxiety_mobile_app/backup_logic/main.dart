import 'dart:io';
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:usage_stats/usage_stats.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'background_service.dart';
import 'background_service_helper.dart';
import 'notification_helper.dart';
import 'rating_settings.dart';
import 'profile_page.dart';
import 'ema_and_gad7.dart'; // FIX: removed duplicate 'package:anxiety_mobile_app/ema_and_gad7.dart' import
import 'package:connectivity_plus/connectivity_plus.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

const Color kPrimaryColor = Color(0xFF00695C);
const Color kSecondaryColor = Color(0xFFB2DFDB);
const Color kAccentColor = Color(0xFF009688);
const Color kSurfaceColor = Colors.white;
const Color kBackgroundColor = Color(0xFFF5F7FA);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(const AnxietyResearchApp());

  try {
    await NotificationHelper.init(onPayload: _handleNotificationPayload);
  } catch (e) {
    debugPrint('NotificationHelper init failed: $e');
  }

  try {
    await BackgroundServiceHelper.retryOfflineQueue();
  } catch (e) {
    debugPrint('Retry offline queue on init failed: $e');
  }

  try {
    Connectivity().onConnectivityChanged.listen((
      List<ConnectivityResult> results,
    ) async {
      final isConnected = results.any((r) => r != ConnectivityResult.none);
      if (isConnected) {
        try {
          await BackgroundServiceHelper.retryOfflineQueue();
        } catch (e) {
          debugPrint('Retry offline queue on connectivity change failed: $e');
        }
      }
    });
  } catch (e) {
    debugPrint('Connectivity listener setup failed: $e');
  }
}

void _handleNotificationPayload(String? payload) {
  if (payload == null) return;
  final ctx = navigatorKey.currentContext;
  if (ctx == null) return;

  if (payload.startsWith('ema_')) {
    final period = payload.replaceFirst('ema_', '');
    showModalBottomSheet(
      context: ctx,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Padding(
        padding: EdgeInsets.only(bottom: MediaQuery.of(ctx).viewInsets.bottom),
        child: EmaRatingSheet(timePeriod: period),
      ),
    );
  } else if (payload == 'gad7') {
    Navigator.push(ctx, MaterialPageRoute(builder: (_) => const Gad7Screen()));
  }
}

class AnxietyResearchApp extends StatelessWidget {
  const AnxietyResearchApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      navigatorKey: navigatorKey,
      title: 'SLIIT Anxiety Research',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: kBackgroundColor,
        colorScheme: ColorScheme.fromSeed(
          seedColor: kPrimaryColor,
          primary: kPrimaryColor,
          secondary: kAccentColor,
          surface: kSurfaceColor,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: kBackgroundColor,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: TextStyle(
            color: Colors.black87,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
          iconTheme: IconThemeData(color: Colors.black87),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: kPrimaryColor,
            foregroundColor: Colors.white,
            elevation: 2,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none,
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: kPrimaryColor, width: 2),
          ),
          contentPadding: const EdgeInsets.all(20),
        ),
      ),
      routes: {
        '/dashboard': (_) => const DashboardPage(),
        '/profile': (_) => const ProfilePage(),
        '/gad7': (_) => const Gad7Screen(),
      },
      home: const SplashRouter(),
    );
  }
}

class SplashRouter extends StatefulWidget {
  const SplashRouter({super.key});

  @override
  State<SplashRouter> createState() => _SplashRouterState();
}

class _SplashRouterState extends State<SplashRouter> {
  @override
  void initState() {
    super.initState();
    _route();
  }

  Future<void> _route() async {
    await Future.delayed(const Duration(milliseconds: 400));
    final prefs = await SharedPreferences.getInstance();
    final userId = prefs.getString('user_id');
    final profileComplete = prefs.getBool('profile_complete') ?? false;

    if (!mounted) return;

    if (userId == null || userId.isEmpty) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const LoginPage()),
      );
    } else if (!profileComplete) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const ProfilePage()),
      );
    } else {
      // Ensure service is running if already logged in
      try {
        await initializeService();
      } catch (e) {
        debugPrint('Service init failed in SplashRouter: $e');
      }

      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const DashboardPage()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: CircularProgressIndicator(color: kPrimaryColor)),
    );
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _idController = TextEditingController();
  bool _permissionsGranted = false;
  bool _isLoading = false;

  Future<void> _requestPermissions() async {
    setState(() => _isLoading = true);

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      PermissionStatus foreground = await Permission.location.request();
      if (foreground.isPermanentlyDenied) {
        await _showOpenSettingsDialog(
          'Location permission is permanently denied. Please enable it in system settings.',
        );
      }
      if (foreground.isGranted) {
        PermissionStatus background = await Permission.locationAlways.request();
        if (background.isPermanentlyDenied) {
          await _showOpenSettingsDialog(
            'Background location is permanently denied. Please enable "Allow all the time" in system settings.',
          );
        }
      }

      await [
        Permission.phone,
        Permission.sms,
        Permission.notification,
      ].request();

      var batteryStatus = await Permission.ignoreBatteryOptimizations.status;
      if (!batteryStatus.isGranted) {
        await Permission.ignoreBatteryOptimizations.request();
      }

      bool isUsageGranted = await UsageStats.checkUsagePermission() ?? false;
      if (!isUsageGranted) {
        await UsageStats.grantUsagePermission();
      }
    }

    await Future.delayed(const Duration(seconds: 1));
    setState(() {
      _permissionsGranted = true;
      _isLoading = false;
    });
  }

  Future<void> _login() async {
    if (_idController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Please enter your Participant ID"),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user_id', _idController.text.trim());
    await initializeService();

    try {
      await BackgroundServiceHelper.retryOfflineQueue();
    } catch (e) {
      debugPrint('Retry offline queue after login failed: $e');
    }

    if (mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const ProfilePage()),
      );
    }
  }

  Future<void> _showOpenSettingsDialog(String message) async {
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Permission Required'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.of(ctx).pop();
              await openAppSettings();
            },
            child: const Text('Open Settings'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Icon(
                  Icons.health_and_safety,
                  size: 64,
                  color: kPrimaryColor,
                ),
                const SizedBox(height: 20),
                const Text(
                  "Research Companion",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  "SLIIT Anxiety Monitoring Study",
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.grey.shade600),
                ),
                const SizedBox(height: 40),
                _buildCard(
                  title: "1. System Configuration",
                  isActive: !_permissionsGranted,
                  child: Column(
                    children: [
                      Text(
                        "This app requires background access to collect behavioral signals for the research.",
                        style: TextStyle(
                          color: Colors.grey.shade700,
                          height: 1.5,
                        ),
                      ),
                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton.icon(
                          onPressed: _permissionsGranted
                              ? null
                              : _requestPermissions,
                          icon: _isLoading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    color: Colors.white,
                                    strokeWidth: 2,
                                  ),
                                )
                              : Icon(
                                  _permissionsGranted
                                      ? Icons.check_circle
                                      : Icons.shield_moon,
                                ),
                          label: Text(
                            _permissionsGranted
                                ? "Configuration Complete"
                                : "Grant Secure Access",
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: _permissionsGranted
                                ? Colors.green
                                : kPrimaryColor,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                Opacity(
                  opacity: _permissionsGranted ? 1.0 : 0.5,
                  child: _buildCard(
                    title: "2. Participant Identification",
                    isActive: _permissionsGranted,
                    child: Column(
                      children: [
                        TextField(
                          controller: _idController,
                          enabled: _permissionsGranted,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(
                            labelText: "Enter Participant ID",
                            prefixIcon: Icon(Icons.person_outline),
                          ),
                        ),
                        const SizedBox(height: 20),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: _permissionsGranted ? _login : null,
                            child: const Text("Initialize Session"),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                const Text(
                  "Your data is encrypted and used solely for research purposes.",
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 12, color: Colors.grey),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCard({
    required String title,
    required Widget child,
    required bool isActive,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
        border: isActive
            ? Border.all(color: kPrimaryColor.withValues(alpha: 0.3))
            : null,
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: isActive ? Colors.black87 : Colors.grey,
            ),
          ),
          const Divider(height: 24),
          child,
        ],
      ),
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  double _currentPressure = 0.0;
  bool _isPressed = false;
  String _cachedId = "";
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
    _loadCachedId();
    _checkSystemStatus();
    _statusTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _checkSystemStatus(),
    );
    NotificationHelper.onNotificationClick = _handleNotificationPayload;
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

  void _handleNotificationPayload(String? payload) {
    _handleNotificationPayloadGlobal(payload);
  }

  Future<void> _loadCachedId() async {
    String id = await BackgroundServiceHelper.getCachedId();
    if (mounted) setState(() => _cachedId = id);
  }

  void _handleTouch(PointerEvent event, bool isPressed) async {
    setState(() {
      _isPressed = isPressed;
      _currentPressure = isPressed ? event.pressure : 0.0;
    });
    if (isPressed) {
      final now = DateTime.now();
      final prefs = await SharedPreferences.getInstance();
      String uid = prefs.getString('user_id') ?? "Unknown";
      bool enoughTime =
          _lastSentAt == null ||
          now.difference(_lastSentAt!) >= _minSendInterval;
      bool enoughDelta =
          (event.pressure - _lastPressureSent).abs() >= _minPressureDelta;
      if (enoughTime && enoughDelta) {
        _lastSentAt = now;
        _lastPressureSent = event.pressure;
        await BackgroundServiceHelper.sendToSheet(
          uid,
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
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const RatingSettingsPage()),
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
                        ? [kPrimaryColor, kAccentColor]
                        : [Colors.white, Colors.grey.shade100],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: kPrimaryColor.withValues(
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
                    color: _isPressed ? kPrimaryColor : Colors.grey.shade200,
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
                          : kPrimaryColor.withValues(alpha: 0.5),
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
              _cachedId.isNotEmpty ? "ID: $_cachedId" : "ID: (loading)",
              style: TextStyle(color: Colors.grey.shade400, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}

void _handleNotificationPayloadGlobal(String? payload) {
  if (payload == null) return;
  final ctx = navigatorKey.currentContext;
  if (ctx == null) return;

  if (payload.startsWith('ema_')) {
    final period = payload.replaceFirst('ema_', '');
    showModalBottomSheet(
      context: ctx,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Padding(
        padding: EdgeInsets.only(bottom: MediaQuery.of(ctx).viewInsets.bottom),
        child: EmaRatingSheet(timePeriod: period),
      ),
    );
  } else if (payload == 'gad7') {
    Navigator.push(ctx, MaterialPageRoute(builder: (_) => const Gad7Screen()));
  }
}
