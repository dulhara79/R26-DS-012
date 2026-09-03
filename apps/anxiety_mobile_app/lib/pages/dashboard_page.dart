import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';

import '../theme/app_theme.dart';
import '../background_service_helper.dart';
import '../background_service.dart' as background_service;
import '../services/chest_strap_service.dart';

import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:app_settings/app_settings.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

import '../services/api_service.dart';
import '../services/anxiety_feedback_service.dart';
import '../services/forecast_message_policy.dart';
import '../services/notification_helper.dart';
import 'baseline_calibration_page.dart';
import '../services/fusion_risk_service.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard Page — Physiological Monitoring
// ─────────────────────────────────────────────────────────────────────────────

class _TimedAnxietyReading {
  final DateTime time;
  final double score;

  const _TimedAnxietyReading(this.time, this.score);
}

class DashboardPage extends StatefulWidget {
  final String? userId;
  const DashboardPage({super.key, this.userId});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage>
    with TickerProviderStateMixin, WidgetsBindingObserver {
  String _cachedId = "";
  bool _isOptimized = false;
  bool _chestStrapConnected = false;
  bool _notificationsEnabled = true;

  // ── Prediction Pipeline State ──────────────────────────────
  String _predictionStatus =
      "loading"; // "loading", "buffering", "not_calibrated", "success", "error"
  List<double> _forecastData = [];
  String _statusMessage = "";
  Timer? _predictionTimer;
  double _forecastCoverage = 0.0;
  List<Map<String, dynamic>> _historyData = [];
  String _historyStatus = 'loading';
  String _historyMessage = '';
  String _historyMetric = 'risk_index';
  double? _currentModelRisk;
  // Final score returned by the teammate's fusion model when configured.
  double? _fusionRiskScore;
  final List<_TimedAnxietyReading> _recentAnxietyReadings = [];
  final ScrollController _forecastScrollController = ScrollController();
  bool _keepForecastAtLatest = true;

  // ── Chest Strap Live Data ──────────────────────────────────
  ChestStrapReading? _currentReading;
  StreamSubscription<ChestStrapReading>? _readingSubscription;
  StreamSubscription<BluetoothAdapterState>? _btStateSubscription;
  bool _isBluetoothDialogShowing = false;

  // ── Animation Controllers ───────────────────────────────────
  late AnimationController _entryController;
  late Animation<double> _entryFade;
  late Animation<Offset> _entrySlide;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadCachedId();

    // Entry animation
    _entryController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _entryFade = CurvedAnimation(
      parent: _entryController,
      curve: Curves.easeOut,
    );
    _entrySlide = Tween<Offset>(begin: const Offset(0, 0.08), end: Offset.zero)
        .animate(
          CurvedAnimation(parent: _entryController, curve: Curves.easeOutCubic),
        );
    _entryController.forward();

    // A persisted reading is historical context, not a live risk score.
    final chestStrap = ChestStrapService();
    _chestStrapConnected = chestStrap.isConnected;
    _currentReading = chestStrap.hasLiveWornReading
        ? chestStrap.lastReading
        : null;

    // Listen for live chest strap data
    _readingSubscription = ChestStrapService().readingsStream.listen((reading) {
      if (mounted) {
        setState(() => _currentReading = reading);
        _uploadChestStrapData(reading);
      }
    });

    // Listen for connection state changes to update UI
    ChestStrapService().connectionState.addListener(_onConnectionChanged);
    ChestStrapService().liveReadingAvailable.addListener(
      _onLiveReadingAvailabilityChanged,
    );

    _btStateSubscription = FlutterBluePlus.adapterState.listen((state) {
      if (state == BluetoothAdapterState.on && _isBluetoothDialogShowing) {
        if (mounted) {
          Navigator.of(context, rootNavigator: true).pop();
        }
        _checkBluetoothConnection();
      }
    });

    _startStatusCheck();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkBluetoothConnection();
      _checkNotificationPermission();
    });
  }

  Future<void> _checkNotificationPermission() async {
    final enabled = await NotificationHelper.ensurePermissions();
    if (mounted) setState(() => _notificationsEnabled = enabled);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(_checkNotificationPermission());
    }
  }

  void _onConnectionChanged() {
    if (mounted) {
      setState(() {
        _chestStrapConnected = ChestStrapService().isConnected;
        if (!_chestStrapConnected) {
          _currentReading = null;
        }
      });
    }
  }

  void _onLiveReadingAvailabilityChanged() {
    if (!mounted) return;
    setState(() {
      if (!ChestStrapService().hasLiveWornReading) {
        _currentReading = null;
      }
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    ChestStrapService().connectionState.removeListener(_onConnectionChanged);
    ChestStrapService().liveReadingAvailable.removeListener(
      _onLiveReadingAvailabilityChanged,
    );
    _entryController.dispose();
    _predictionTimer?.cancel();
    _btStateSubscription?.cancel();
    _readingSubscription?.cancel();
    _forecastScrollController.dispose();
    super.dispose();
  }

  // ── Chest Strap Bluetooth Flow ───────────────────────────────

  Future<void> _checkBluetoothConnection() async {
    debugPrint('[Dashboard] _checkBluetoothConnection called');

    // Already connected? No need to do anything.
    if (ChestStrapService().isConnected) {
      debugPrint('[Dashboard] Already connected to chest strap, skipping.');
      return;
    }

    // Check if Bluetooth adapter is on
    final adapterState = await FlutterBluePlus.adapterState.first;
    debugPrint('[Dashboard] Adapter state: $adapterState');
    if (adapterState != BluetoothAdapterState.on) {
      _showBluetoothOffDialog();
      return;
    }

    // Check permissions
    bool scanGranted = await Permission.bluetoothScan.isGranted;
    bool connectGranted = await Permission.bluetoothConnect.isGranted;
    // On Android 11 and below, location permission is also needed for BLE scanning
    bool locationGranted = await Permission.locationWhenInUse.isGranted;
    debugPrint(
      '[Dashboard] Permissions - scan: $scanGranted, connect: $connectGranted, location: $locationGranted',
    );

    if (!scanGranted || !connectGranted || !locationGranted) {
      _showBluetoothPrompt();
    } else {
      // Permissions granted, start scanning
      debugPrint('[Dashboard] All permissions granted. Starting scan...');
      _startChestStrapScan();
    }
  }

  void _showBluetoothOffDialog() {
    if (_isBluetoothDialogShowing) return;
    _isBluetoothDialogShowing = true;
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: Text(
          'Bluetooth is Off',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
        ),
        content: Text(
          'Please turn on Bluetooth to connect your chest strap and see your current anxiety level.',
          style: GoogleFonts.poppins(),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              // Continue without strap
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Using saved information for now.'),
                ),
              );
            },
            child: Text('Skip', style: GoogleFonts.poppins(color: Colors.red)),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await AppSettings.openAppSettings(
                type: AppSettingsType.bluetooth,
              );
              // Re-check after user returns from settings
              Future.delayed(
                const Duration(seconds: 2),
                _checkBluetoothConnection,
              );
            },
            child: Text(
              'Turn On',
              style: GoogleFonts.poppins(color: AppTheme.kPrimaryDeep),
            ),
          ),
        ],
      ),
    ).then((_) {
      _isBluetoothDialogShowing = false;
    });
  }

  void _showBluetoothPrompt() {
    if (_isBluetoothDialogShowing) return;
    _isBluetoothDialogShowing = true;
    debugPrint('[Dashboard] Showing Bluetooth permission prompt');
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: Text(
          'Connect Chest Strap',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
        ),
        content: Text(
          'Aura needs Bluetooth access to connect to your chest strap and show live readings.',
          style: GoogleFonts.poppins(),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _showDenyWarning();
            },
            child: Text('Deny', style: GoogleFonts.poppins(color: Colors.red)),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await Permission.bluetoothScan.request();
              await Permission.bluetoothConnect.request();
              await Permission.locationWhenInUse.request();
              bool scanOk = await Permission.bluetoothScan.isGranted;
              bool connectOk = await Permission.bluetoothConnect.isGranted;
              bool locationOk = await Permission.locationWhenInUse.isGranted;
              debugPrint(
                '[Dashboard] After permission request - scan: $scanOk, connect: $connectOk, location: $locationOk',
              );
              if (scanOk && connectOk && locationOk) {
                _startChestStrapScan();
              } else {
                _showDenyWarning();
              }
            },
            child: Text(
              'Allow',
              style: GoogleFonts.poppins(color: AppTheme.kPrimaryDeep),
            ),
          ),
        ],
      ),
    ).then((_) {
      _isBluetoothDialogShowing = false;
    });
  }

  void _showDenyWarning() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: Text(
          'Are you sure?',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
        ),
        content: Text(
          'Without the chest strap, Aura cannot show live body readings. You can continue, but your current anxiety level will be unavailable.\n\nDo you want to continue?',
          style: GoogleFonts.poppins(),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _showAskAgainPrompt();
            },
            child: Text(
              'No, go back',
              style: GoogleFonts.poppins(color: Colors.red),
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Using saved information for now.'),
                ),
              );
            },
            child: Text(
              'Yes, that\'s fine',
              style: GoogleFonts.poppins(color: AppTheme.kPrimaryDeep),
            ),
          ),
        ],
      ),
    );
  }

  void _showAskAgainPrompt() {
    if (_isBluetoothDialogShowing) return;
    _isBluetoothDialogShowing = true;
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: Text(
          'Bluetooth Needed',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
        ),
        content: Text(
          'Please let Aura use Bluetooth to connect to your chest strap.',
          style: GoogleFonts.poppins(),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Using saved information for now.'),
                ),
              );
            },
            child: Text('Skip', style: GoogleFonts.poppins(color: Colors.red)),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await AppSettings.openAppSettings(
                type: AppSettingsType.bluetooth,
              );
              Future.delayed(
                const Duration(seconds: 2),
                _checkBluetoothConnection,
              );
            },
            child: Text(
              'Turn on Bluetooth',
              style: GoogleFonts.poppins(color: AppTheme.kPrimaryDeep),
            ),
          ),
        ],
      ),
    ).then((_) {
      _isBluetoothDialogShowing = false;
    });
  }

  void _startChestStrapScan() {
    debugPrint('[Dashboard] _startChestStrapScan called');
    ChestStrapService().startScan().then((_) {
      debugPrint(
        '[Dashboard] startScan() completed. isConnected: ${ChestStrapService().isConnected}',
      );
      if (mounted && ChestStrapService().isConnected) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Chest strap connected.'),
            backgroundColor: Colors.green,
          ),
        );
      } else if (mounted && !ChestStrapService().isConnected) {
        debugPrint('[Dashboard] Scan timed out without finding ChestStrap_V3');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Chest strap not found. Make sure it is turned on and nearby.',
            ),
            backgroundColor: Colors.orange,
          ),
        );
      }
    });
  }

  // ── Data & Service Helpers ──────────────────────────────────

  Future<void> _loadCachedId() async {
    if (widget.userId != null) {
      setState(() => _cachedId = widget.userId!);
    } else {
      String id = await BackgroundServiceHelper.getCachedId();
      if (mounted) setState(() => _cachedId = id);
    }
    _startPredictionPolling();
    _fetchHistory();
  }

  // ── Forecast API Methods ────────────────────────────────────

  void _startPredictionPolling() {
    _predictionTimer?.cancel();
    _fetchForecast();
    // Poll the FastAPI prediction endpoint every 30 seconds
    _predictionTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      _fetchForecast();
    });
  }

  Future<void> _fetchForecast() async {
    if (_cachedId.isEmpty) return;

    final result = await ApiService.getEscalationForecast(_cachedId);
    if (!mounted) return;

    final status = result['status'] as String?;
    final message = result['message'] as String? ?? "";

    if (status == 'success') {
      final List? riskForecast = result['risk_forecast'] as List?;
      final List? forecastHorizons =
          result['forecast_horizons_minutes'] as List?;
      if (riskForecast == null ||
          riskForecast.length != 2 ||
          riskForecast.any(
            (value) => value is! num || !value.toDouble().isFinite,
          ) ||
          forecastHorizons == null ||
          forecastHorizons.length != 2 ||
          forecastHorizons.any(
            (value) => value is! num || !value.toDouble().isFinite,
          )) {
        setState(() {
          _predictionStatus = 'error';
          _forecastData = [];
          _currentModelRisk = null;
          _forecastCoverage = 0.0;
          _statusMessage =
              'The +5 and +10 minute forecast is not available right now.';
        });
        return;
      }
      final parsedForecast = riskForecast
          .cast<num>()
          .map((value) => value.toDouble())
          .toList();
      final parsedHorizons = forecastHorizons
          .cast<num>()
          .map((value) => value.toDouble())
          .toList();
      if (parsedHorizons[0] != 5.0 || parsedHorizons[1] != 10.0) {
        setState(() {
          _predictionStatus = 'error';
          _forecastData = [];
          _currentModelRisk = null;
          _forecastCoverage = 0.0;
          _statusMessage = 'The forecast horizons are not recognised.';
        });
        return;
      }
      final currentRisk =
          (result['current_risk_index'] as num?)?.toDouble() ??
          (ChestStrapService().hasLiveWornReading
              ? ChestStrapService().lastReading?.riskScore
              : null);
      final now = DateTime.now();
      if (currentRisk != null) {
        _recentAnxietyReadings.add(
          _TimedAnxietyReading(now, currentRisk.clamp(0.0, 100.0).toDouble()),
        );
        _recentAnxietyReadings.removeWhere(
          (reading) =>
              now.difference(reading.time) > const Duration(minutes: 30),
        );
      }

      setState(() {
        _predictionStatus = "success";
        _forecastData = parsedForecast;
        _currentModelRisk = currentRisk;
        _forecastCoverage = 1.0;
        _statusMessage = message;
      });
      AnxietyFeedbackService().observeForecastResponse(result);
      if (_keepForecastAtLatest) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted || !_forecastScrollController.hasClients) return;
          _forecastScrollController.jumpTo(
            _forecastScrollController.position.maxScrollExtent,
          );
        });
      }

      // Notify the central backend to fetch C1's latest prediction, then
      // refresh fusion so the home page does not wait for its polling interval.
      if (parsedForecast.isNotEmpty && _cachedId.isNotEmpty) {
        ApiService.submitPhysiologicalWindow(
          participantId: _cachedId,
        ).then((sent) {
          if (sent) FusionRiskService.instance.fetch();
        });
      }
    } else if (status == 'buffering') {
      final coverage = ((result['coverage'] as num?)?.toDouble() ?? 0.0)
          .clamp(0.0, 1.0)
          .toDouble();
      setState(() {
        _predictionStatus = "buffering";
        _forecastData = [];
        _currentModelRisk = null;
        _fusionRiskScore = null;
        _forecastCoverage = coverage;
        _statusMessage = message;
      });
    } else if (status == 'not_calibrated') {
      setState(() {
        _predictionStatus = "not_calibrated";
        _forecastData = [];
        _currentModelRisk = null;
        _fusionRiskScore = null;
        _forecastCoverage = 0.0;
        _statusMessage = message;
      });
    } else {
      // API Offline/Error state
      setState(() {
        _predictionStatus = "error";
        _forecastData = [];
        _currentModelRisk = null;
        _fusionRiskScore = null;
        _forecastCoverage = 0.0;
        _statusMessage = message;
      });
    }
  }

  double _scaleForecastValue(double value) {
    return value.clamp(0.0, 100.0);
  }

  List<double> get _effectiveForecastData {
    if (_forecastData.isNotEmpty) return _forecastData;
    // No real data available yet
    return [];
  }

  double? _historyNumber(Map<String, dynamic> row, List<String> keys) {
    for (final key in keys) {
      final value = row[key];
      if (value is num) return value.toDouble();
      if (value is String) {
        final parsed = double.tryParse(value);
        if (parsed != null) return parsed;
      }
    }
    return null;
  }

  DateTime? _historyTimestamp(Map<String, dynamic> row) {
    for (final key in ['timestamp', '_time', 'time', 'datetime', 'date']) {
      final value = row[key];
      if (value is num) {
        if (!value.isFinite) continue;
        final absolute = value.abs();
        final int milliseconds;
        if (absolute >= 100000000000000000) {
          milliseconds = value.toInt() ~/ 1000000;
        } else if (absolute >= 100000000000000) {
          milliseconds = value.toInt() ~/ 1000;
        } else if (absolute >= 100000000000) {
          milliseconds = value.toInt();
        } else {
          milliseconds = value.toInt() * 1000;
        }
        try {
          return DateTime.fromMillisecondsSinceEpoch(
            milliseconds,
            isUtc: true,
          ).toLocal();
        } on RangeError {
          continue;
        }
      }
      if (value is String && value.isNotEmpty) {
        final parsed = DateTime.tryParse(value);
        if (parsed != null) return parsed.toLocal();
      }
    }
    return null;
  }

  List<Map<String, dynamic>> _normaliseHistoryRows(List<dynamic> rawRows) {
    final grouped = <String, Map<String, List<double>>>{};

    for (final rawRow in rawRows.whereType<Map>()) {
      final row = Map<String, dynamic>.from(rawRow);
      final timestamp = _historyTimestamp(row);
      if (timestamp == null) continue;
      final date = DateFormat('yyyy-MM-dd').format(timestamp);
      final bucket = grouped.putIfAbsent(date, () => <String, List<double>>{});

      final values = <String, double?>{
        'risk_index': _historyNumber(row, ['risk_index', 'risk_score', 'risk']),
        'mean_hr': _historyNumber(row, ['mean_hr', 'mean_HR', 'meanHR']),
        'mean_br': _historyNumber(row, ['mean_br', 'mean_BR', 'meanBR']),
        'mean_temp': _historyNumber(row, [
          'mean_temp',
          'mean_temperature',
          'temperature',
        ]),
        'mean_motion': _historyNumber(row, [
          'mean_motion',
          'std_acc_mag',
          'mean_acc_mag',
          'motion',
        ]),
      };

      for (final entry in values.entries) {
        final value = entry.value;
        if (value != null && value.isFinite) {
          bucket.putIfAbsent(entry.key, () => <double>[]).add(value);
        }
      }
    }

    final dates = grouped.keys.toList()..sort();
    return dates.map((date) {
      final row = <String, dynamic>{'date': date};
      for (final entry in grouped[date]!.entries) {
        if (entry.value.isNotEmpty) {
          row[entry.key] =
              entry.value.reduce((a, b) => a + b) / entry.value.length;
        }
      }
      return row;
    }).toList();
  }

  Future<void> _fetchHistory() async {
    if (_cachedId.isEmpty) return;
    if (mounted) {
      setState(() {
        _historyStatus = 'loading';
        _historyMessage = '';
      });
    }
    final result = await ApiService.getPhysiologicalHistory(_cachedId);
    if (!mounted) return;
    if (result['status'] == 'success') {
      final rows = _normaliseHistoryRows(
        (result['history'] as List?)?.cast<dynamic>() ?? <dynamic>[],
      );
      setState(() {
        _historyData = rows;
        _historyStatus = rows.isEmpty ? 'empty' : 'success';
        _historyMessage = '';
      });
    } else {
      setState(() {
        _historyStatus = 'error';
        _historyMessage =
            result['message'] as String? ??
            'The history service did not return data.';
      });
    }
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
          _isOptimized = optimized;
        });
        if (!isRunning) {
          await background_service.startBackgroundServiceIfPermitted();
        }
      }
    });
  }

  Future<void> _uploadChestStrapData(ChestStrapReading reading) async {
    if (_cachedId.isEmpty) return;
    final data = {
      'mean_HR': reading.meanHR.toStringAsFixed(1),
      'mean_RR': reading.meanRR.toStringAsFixed(2),
      'SDNN': reading.sdnn.toStringAsFixed(2),
      'RMSSD': reading.rmssd.toStringAsFixed(2),
      'mean_BR': reading.meanBR.toStringAsFixed(1),
      'std_BR': reading.stdBR.toStringAsFixed(2),
      'mean_temp': reading.meanTemp.toStringAsFixed(2),
      'std_temp': reading.stdTemp.toStringAsFixed(2),
      'mean_acc_mag': reading.meanAccMag.toStringAsFixed(4),
      'std_acc_mag': reading.stdAccMag.toStringAsFixed(4),
      'risk_score': reading.riskScore.toStringAsFixed(1),
      'risk_label': reading.riskLabel,
      'is_worn': reading.isWorn ? 1 : 0,
      'timestamp': DateTime.now().toIso8601String(),
    };
    await BackgroundServiceHelper.sendToSheet(
      _cachedId,
      'ChestStrap_Vitals',
      data.toString(),
    );
  }

  // ── Risk Score Color ────────────────────────────────────────
  Color _riskColor(double score) {
    if (score <= 20) return const Color(0xFF4CAF50);
    if (score <= 45) return const Color(0xFFFFA726);
    if (score <= 70) return const Color(0xFFFF7043);
    return const Color(0xFFEF5350);
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'Normal':
      case 'Still':
      case 'Low':
        return const Color(0xFF4CAF50);
      case 'Moderate':
      case 'Elevated':
      case 'Restless':
        return const Color(0xFFFFA726);
      case 'High':
      case 'Agitated':
        return const Color(0xFFEF5350);
      default:
        return Colors.grey;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // BUILD
  // ═══════════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    final hasLiveReading =
        _chestStrapConnected && (_currentReading?.isWorn ?? false);
    final risk = hasLiveReading ? _currentReading!.riskScore : 0.0;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          'Body Readings',
          style: GoogleFonts.poppins(
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: Theme.of(context).colorScheme.onSurface,
          ),
        ),
        automaticallyImplyLeading: false,
      ),
      body: SlideTransition(
        position: _entrySlide,
        child: FadeTransition(opacity: _entryFade, child: _buildBody(risk)),
      ),
    );
  }

  Widget _buildBody(double risk) {
    if (!_chestStrapConnected) {
      return _buildDisconnectedScreen();
    }

    switch (_predictionStatus) {
      case 'loading':
        return _buildLoadingScreen();
      case 'not_calibrated':
        return _buildCalibrationRequiredScreen();
      case 'buffering':
        return _buildDashboardList(risk);
      case 'error':
      case 'success':
      default:
        return _buildDashboardList(risk);
    }
  }

  Widget _buildDashboardList(double risk) {
    final isWorn = _chestStrapConnected && (_currentReading?.isWorn ?? false);
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 30),
      children: [
        // ── Connection Offline Warning Banner ──
        if (_predictionStatus == 'error') _buildOfflineBanner(),

        // ── Service Status Strip ──
        _buildServiceStrip(),
        if (isWorn) ...[const SizedBox(height: 10), _buildAdviceCard(risk)],
        if (!_notificationsEnabled) ...[
          const SizedBox(height: 10),
          _buildNotificationWarning(),
        ],
        if (_isOptimized) ...[
          const SizedBox(height: 10),
          _buildBatteryWarning(),
        ],

        const SizedBox(height: 20),

        // ── KPI Grid (2x2) ──
        Row(
          children: [
            Expanded(
              child: _KpiCard(
                label: 'Heart Rate',
                value: isWorn
                    ? _currentReading!.meanHR.toStringAsFixed(0)
                    : '--',
                unit: 'bpm',
                icon: Icons.monitor_heart_rounded,
                status: _currentReading?.hrStatus ?? 'N/A',
                statusColor: _statusColor(_currentReading?.hrStatus ?? 'N/A'),
                gradient: const [Color(0xFFFF6B6B), Color(0xFFee5a24)],
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: _KpiCard(
                label: 'Breathing',
                value: isWorn
                    ? _currentReading!.meanBR.toStringAsFixed(0)
                    : '--',
                unit: 'br/min',
                icon: Icons.air_rounded,
                status: _currentReading?.brStatus ?? 'N/A',
                statusColor: _statusColor(_currentReading?.brStatus ?? 'N/A'),
                gradient: const [Color(0xFF4facfe), Color(0xFF00f2fe)],
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        Row(
          children: [
            Expanded(
              child: _KpiCard(
                label: 'Temperature',
                value: isWorn
                    ? (_currentReading!.meanTemp - 3).toStringAsFixed(1)
                    : '--',
                unit: '°C',
                icon: Icons.thermostat_rounded,
                status: _currentReading?.tempStatus ?? 'N/A',
                statusColor: _statusColor(_currentReading?.tempStatus ?? 'N/A'),
                gradient: const [Color(0xFFF6D365), Color(0xFFFDA085)],
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: _KpiCard(
                label: 'Motion',
                value: isWorn
                    ? _currentReading!.stdAccMag.toStringAsFixed(3)
                    : '--',
                unit: 'g',
                icon: Icons.directions_walk_rounded,
                status: _currentReading?.motionStatus ?? 'N/A',
                statusColor: _statusColor(
                  _currentReading?.motionStatus ?? 'N/A',
                ),
                gradient: const [Color(0xFFA18CD1), Color(0xFFFBC2EB)],
              ),
            ),
          ],
        ),

        const SizedBox(height: 24),
        _buildChartsSection(),
        const SizedBox(height: 20),

        // ── Footer ──
        Center(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: Theme.of(
                context,
              ).colorScheme.surfaceContainerHighest.withValues(alpha: 0.75),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              _cachedId.isNotEmpty
                  ? 'ID: $_cachedId  •  Live readings'
                  : 'Starting...',
              style: GoogleFonts.poppins(
                fontSize: 11,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ── Connection Status / Screen Swapping Builders ────────────

  Widget _buildOfflineBanner() {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.tertiaryContainer,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Theme.of(context).colorScheme.tertiary.withValues(alpha: 0.5),
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.wifi_off_rounded,
            color: Theme.of(context).colorScheme.tertiary,
            size: 18,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _statusMessage.isEmpty
                  ? 'Forecast unavailable. Connect the chest strap and try again.'
                  : _statusMessage,
              style: GoogleFonts.poppins(
                fontSize: 11,
                color: Theme.of(context).colorScheme.onTertiaryContainer,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingScreen() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(
            valueColor: AlwaysStoppedAnimation<Color>(
              Theme.of(context).colorScheme.primary,
            ),
          ),
          const SizedBox(height: 20),
          Text(
            'Preparing your forecast...',
            style: GoogleFonts.poppins(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCalibrationRequiredScreen() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(24),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.04),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.tertiaryContainer,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.tune_rounded,
                  color: Theme.of(context).colorScheme.tertiary,
                  size: 48,
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'One-Time Setup Needed',
                style: GoogleFonts.poppins(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Text(
                'Aura needs to learn what your readings look like while you are calm. This takes 3 minutes and helps make your anxiety results more accurate.',
                style: GoogleFonts.poppins(
                  fontSize: 13,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                  height: 1.6,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 30),
              ElevatedButton.icon(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) =>
                          BaselineCalibrationPage(userId: _cachedId),
                    ),
                  ).then((_) => _fetchForecast());
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.kPrimaryDeep,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 28,
                    vertical: 14,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  elevation: 2,
                ),
                icon: const Icon(Icons.play_arrow_rounded, size: 22),
                label: Text(
                  'Start 3-Minute Setup',
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDisconnectedScreen() {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(24),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.04),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.bluetooth_disabled_rounded,
                size: 64,
                color: Colors.grey,
              ),
              const SizedBox(height: 16),
              Text(
                'Chest Strap Disconnected',
                style: GoogleFonts.poppins(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'Turn on the chest strap and connect it to see live readings.',
                textAlign: TextAlign.center,
                style: GoogleFonts.poppins(
                  fontSize: 13,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _startChestStrapScan,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.kPrimaryDeep,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    vertical: 14,
                    horizontal: 24,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                icon: const Icon(Icons.bluetooth_searching_rounded, size: 20),
                label: Text(
                  'Scan & Connect',
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 28),
        Text(
          'Past 30 Days',
          style: GoogleFonts.poppins(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Theme.of(context).colorScheme.onSurface,
          ),
        ),
        const SizedBox(height: 16),
        _buildHistoryCard(),
      ],
    );
  }

  Widget _buildBufferingScreen() {
    final progress = _forecastCoverage.clamp(0.0, 1.0).toDouble();
    final collectedMinutes = (progress * 10).round();

    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(24),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.04),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Building Your Forecast',
                style: GoogleFonts.poppins(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'Collecting 10 consecutive one-minute readings',
                style: GoogleFonts.poppins(
                  fontSize: 12,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 36),

              // Countdown Ring
              Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 120,
                    height: 120,
                    child: CircularProgressIndicator(
                      value: progress,
                      strokeWidth: 8,
                      backgroundColor: Theme.of(
                        context,
                      ).colorScheme.surfaceContainerHighest,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        Theme.of(context).colorScheme.primary,
                      ),
                    ),
                  ),
                  Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '$collectedMinutes/10',
                        style: GoogleFonts.poppins(
                          fontSize: 32,
                          fontWeight: FontWeight.w800,
                          color: Theme.of(context).colorScheme.onSurface,
                          height: 1.0,
                        ),
                      ),
                      Text(
                        'minutes',
                        style: GoogleFonts.poppins(
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 36),

              Text(
                _statusMessage.isEmpty
                    ? 'Keep the chest strap connected. The forecast will appear after 10 consecutive valid one-minute readings.'
                    : _statusMessage,
                style: GoogleFonts.poppins(
                  fontSize: 13,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                  height: 1.6,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              const Divider(),
              const SizedBox(height: 16),

              // Pipeline steps check-list
              Column(
                children: [
                  _buildPipelineStepRow(
                    _chestStrapConnected,
                    'Chest strap connected',
                  ),
                  const SizedBox(height: 10),
                  _buildPipelineStepRow(true, 'Personal resting level ready'),
                  const SizedBox(height: 10),
                  _buildPipelineStepRow(
                    progress >= 1.0,
                    'Collecting 10 consecutive valid minutes',
                    trailing: '$collectedMinutes/10',
                  ),
                  const SizedBox(height: 10),
                  _buildPipelineStepRow(
                    false,
                    'Preparing the +5 and +10 minute forecast',
                    trailing: progress >= 1.0 ? 'Connecting...' : 'Pending',
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPipelineStepRow(bool complete, String text, {String? trailing}) {
    return Row(
      children: [
        Icon(
          complete
              ? Icons.check_circle_rounded
              : Icons.radio_button_unchecked_rounded,
          color: complete ? Colors.green : Colors.grey.shade400,
          size: 18,
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            text,
            style: GoogleFonts.poppins(
              fontSize: 11.5,
              color: complete
                  ? Theme.of(context).colorScheme.onSurface
                  : Theme.of(context).colorScheme.onSurfaceVariant,
              fontWeight: complete ? FontWeight.w500 : FontWeight.normal,
            ),
          ),
        ),
        if (trailing != null)
          Text(
            trailing,
            style: GoogleFonts.poppins(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: complete
                  ? Colors.green
                  : Theme.of(context).colorScheme.primary,
            ),
          ),
      ],
    );
  }

  Widget _buildForecastChart() {
    final List<double> forecast = _effectiveForecastData;
    if (forecast.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.04),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(
                    Icons.insights_rounded,
                    color: Theme.of(context).colorScheme.primary,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  'Your Next 10 Minutes',
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 40),
            Icon(
              Icons.hourglass_empty_rounded,
              size: 40,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
            const SizedBox(height: 12),
            Text(
              'Waiting for chest strap readings...',
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            Text(
              'Your outlook will appear after 10 consecutive valid one-minute readings',
              style: GoogleFonts.poppins(
                fontSize: 11,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      );
    }
    final liveCurrentRisk =
        _chestStrapConnected && (_currentReading?.isWorn ?? false)
        ? _currentReading?.riskScore
        : null;
    final currentRisk =
        (liveCurrentRisk ??
                _currentModelRisk ??
                _scaleForecastValue(forecast.first))
            .clamp(0.0, 100.0)
            .toDouble();
    const forecastHorizons = <double>[5.0, 10.0];
    final List<FlSpot> spots = [
      FlSpot(0, currentRisk),
      ...List.generate(forecast.length, (index) {
        final yVal = _scaleForecastValue(forecast[index]);
        return FlSpot(forecastHorizons[index], yVal);
      }),
    ];
    final forecastSummary = describeForecast(
      currentRisk: currentRisk,
      forecast: forecast.map(_scaleForecastValue).toList(),
    );
    final predictedPeak = forecastSummary.predictedPeak;
    final currentElevated =
        forecastSummary.tone == ForecastMessageTone.elevated;
    final trendColor = forecastSummary.isUrgent
        ? const Color(0xFFEF5350)
        : currentElevated
        ? const Color(0xFFFFA726)
        : const Color(0xFF4CAF50);
    final trendIcon = forecastSummary.isUrgent
        ? Icons.self_improvement_rounded
        : currentElevated
        ? Icons.info_outline_rounded
        : Icons.check_circle_outline_rounded;
    final trendTitle = forecastSummary.title;
    final trendDetail = forecastSummary.isUrgent
        ? 'This is a model estimate, not a diagnosis. Take a slow breath and notice how you feel.'
        : 'This model estimate updates as new body readings arrive.';
    final lineColor = _riskColor(max(currentRisk, predictedPeak));

    final now = DateTime.now();
    final pastSpots =
        _recentAnxietyReadings
            .map(
              (reading) => FlSpot(
                -now.difference(reading.time).inSeconds / 60.0,
                reading.score,
              ),
            )
            .where((spot) => spot.x <= -0.2)
            .toList()
          ..sort((a, b) => a.x.compareTo(b.x));
    final visibleSpots = <FlSpot>[...pastSpots, ...spots];
    final minX = pastSpots.isEmpty
        ? 0.0
        : min(-2.0, pastSpots.first.x.floorToDouble());

    double minY = 0.0;
    double maxY = 100.0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.kPrimaryDeep.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.insights_rounded,
                  color: AppTheme.kPrimaryDeep,
                  size: 20,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Your Next 10 Minutes',
                      style: GoogleFonts.poppins(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: Theme.of(context).colorScheme.onSurface,
                      ),
                    ),
                    Text(
                      'A model estimate from recent body readings',
                      style: GoogleFonts.poppins(
                        fontSize: 11,
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: trendColor.withValues(alpha: 0.10),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: trendColor.withValues(alpha: 0.25)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(trendIcon, color: trendColor, size: 20),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        trendTitle,
                        style: GoogleFonts.poppins(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w600,
                          color: trendColor,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        trendDetail,
                        style: GoogleFonts.poppins(
                          fontSize: 10,
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'Swipe left to see earlier readings.',
            style: GoogleFonts.poppins(
              fontSize: 10,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 8),
          LayoutBuilder(
            builder: (context, constraints) {
              final pixelsPerMinute = constraints.maxWidth / 10.0;
              final chartWidth = max(
                constraints.maxWidth,
                (10.0 - minX) * pixelsPerMinute,
              );
              return NotificationListener<ScrollNotification>(
                onNotification: (notification) {
                  if (notification is UserScrollNotification &&
                      _forecastScrollController.hasClients) {
                    final position = _forecastScrollController.position;
                    _keepForecastAtLatest =
                        position.maxScrollExtent - position.pixels < 36;
                  }
                  return false;
                },
                child: SingleChildScrollView(
                  controller: _forecastScrollController,
                  scrollDirection: Axis.horizontal,
                  child: SizedBox(
                    width: chartWidth,
                    height: 180,
                    child: LineChart(
                      LineChartData(
                        minX: minX,
                        maxX: 10,
                        minY: minY,
                        maxY: maxY,
                        gridData: FlGridData(
                          show: true,
                          drawVerticalLine: false,
                          getDrawingHorizontalLine: (value) {
                            return FlLine(
                              color: Colors.grey.withValues(alpha: 0.08),
                              strokeWidth: 1,
                              dashArray: [5, 5],
                            );
                          },
                        ),
                        titlesData: FlTitlesData(
                          show: true,
                          rightTitles: const AxisTitles(
                            sideTitles: SideTitles(showTitles: false),
                          ),
                          topTitles: const AxisTitles(
                            sideTitles: SideTitles(showTitles: false),
                          ),
                          bottomTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              reservedSize: 22,
                              getTitlesWidget: (value, meta) {
                                if (value.abs() < 0.05) {
                                  return Padding(
                                    padding: const EdgeInsets.only(top: 4.0),
                                    child: Text(
                                      'Now',
                                      style: GoogleFonts.poppins(
                                        fontSize: 9.5,
                                        color: Theme.of(
                                          context,
                                        ).colorScheme.onSurface,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  );
                                }
                                if (value < minX ||
                                    value > 10 ||
                                    value % 2 != 0) {
                                  return const SizedBox.shrink();
                                }
                                return Padding(
                                  padding: const EdgeInsets.only(top: 4.0),
                                  child: Text(
                                    value < 0
                                        ? '${value.abs().toInt()}m ago'
                                        : '+${value.toInt()}m',
                                    style: GoogleFonts.poppins(
                                      fontSize: 9.5,
                                      color: Theme.of(
                                        context,
                                      ).colorScheme.onSurfaceVariant,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
                          leftTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              reservedSize: 32,
                              interval: 25,
                              getTitlesWidget: (value, meta) {
                                if (value < 0 || value > 100) {
                                  return const SizedBox.shrink();
                                }
                                return Text(
                                  '${value.toInt()}%',
                                  style: GoogleFonts.poppins(
                                    fontSize: 9.5,
                                    color: Theme.of(
                                      context,
                                    ).colorScheme.onSurfaceVariant,
                                    fontWeight: FontWeight.w500,
                                  ),
                                );
                              },
                            ),
                          ),
                        ),
                        borderData: FlBorderData(show: false),
                        rangeAnnotations: RangeAnnotations(
                          horizontalRangeAnnotations: [
                            HorizontalRangeAnnotation(
                              y1: 0,
                              y2: 20,
                              color: const Color(
                                0xFF4CAF50,
                              ).withValues(alpha: 0.055),
                            ),
                            HorizontalRangeAnnotation(
                              y1: 20,
                              y2: 45,
                              color: const Color(
                                0xFFFFC107,
                              ).withValues(alpha: 0.055),
                            ),
                            HorizontalRangeAnnotation(
                              y1: 45,
                              y2: 70,
                              color: const Color(
                                0xFFFF7043,
                              ).withValues(alpha: 0.055),
                            ),
                            HorizontalRangeAnnotation(
                              y1: 70,
                              y2: 100,
                              color: const Color(
                                0xFFEF5350,
                              ).withValues(alpha: 0.065),
                            ),
                          ],
                        ),
                        extraLinesData: ExtraLinesData(
                          horizontalLines: [
                            HorizontalLine(
                              y: 20,
                              color: const Color(
                                0xFF4CAF50,
                              ).withValues(alpha: 0.25),
                              strokeWidth: 1.5,
                              dashArray: [4, 4],
                              label: HorizontalLineLabel(
                                show: true,
                                alignment: Alignment.topRight,
                                style: GoogleFonts.poppins(
                                  fontSize: 8,
                                  color: const Color(0xFF4CAF50),
                                  fontWeight: FontWeight.w600,
                                ),
                                labelResolver: (line) => 'Low',
                              ),
                            ),
                            HorizontalLine(
                              y: 45,
                              color: const Color(
                                0xFFFFA726,
                              ).withValues(alpha: 0.25),
                              strokeWidth: 1.5,
                              dashArray: [4, 4],
                              label: HorizontalLineLabel(
                                show: true,
                                alignment: Alignment.topRight,
                                style: GoogleFonts.poppins(
                                  fontSize: 8,
                                  color: const Color(0xFFFFA726),
                                  fontWeight: FontWeight.w600,
                                ),
                                labelResolver: (line) => 'Elevated',
                              ),
                            ),
                            HorizontalLine(
                              y: 70,
                              color: const Color(
                                0xFFEF5350,
                              ).withValues(alpha: 0.25),
                              strokeWidth: 1.5,
                              dashArray: [4, 4],
                              label: HorizontalLineLabel(
                                show: true,
                                alignment: Alignment.topRight,
                                style: GoogleFonts.poppins(
                                  fontSize: 8,
                                  color: const Color(0xFFEF5350),
                                  fontWeight: FontWeight.w600,
                                ),
                                labelResolver: (line) => 'High',
                              ),
                            ),
                          ],
                        ),
                        lineBarsData: [
                          LineChartBarData(
                            spots: visibleSpots,
                            isCurved: true,
                            color: lineColor,
                            barWidth: 4,
                            isStrokeCapRound: true,
                            dotData: FlDotData(
                              show: true,
                              getDotPainter: (spot, percent, barData, index) {
                                return FlDotCirclePainter(
                                  radius: spot.x == 0 ? 5 : 3.5,
                                  color: Colors.white,
                                  strokeWidth: 2,
                                  strokeColor: lineColor,
                                );
                              },
                            ),
                            belowBarData: BarAreaData(
                              show: true,
                              gradient: LinearGradient(
                                colors: [
                                  lineColor.withValues(alpha: 0.18),
                                  lineColor.withValues(alpha: 0.0),
                                ],
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                              ),
                            ),
                          ),
                        ],
                        lineTouchData: LineTouchData(
                          touchTooltipData: LineTouchTooltipData(
                            getTooltipItems: (touchedSpots) {
                              return touchedSpots.map((spot) {
                                String riskLabel = 'Low';
                                if (spot.y > 70) {
                                  riskLabel = 'High';
                                } else if (spot.y > 45) {
                                  riskLabel = 'Elevated';
                                } else if (spot.y > 20) {
                                  riskLabel = 'Moderate';
                                }

                                return LineTooltipItem(
                                  '${spot.x.abs() < 0.05
                                      ? 'Now'
                                      : spot.x < 0
                                      ? '${spot.x.abs().toStringAsFixed(0)} min ago'
                                      : 'In ${spot.x.toInt()} min'}\n${spot.y.toStringAsFixed(0)}% ($riskLabel)',
                                  GoogleFonts.poppins(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w600,
                                    fontSize: 10,
                                  ),
                                );
                              }).toList();
                            },
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Widget Builders
  // ─────────────────────────────────────────────────────────────

  Widget _buildServiceStrip() {
    final bool isWorn = _currentReading?.isWorn ?? false;
    final colors = Theme.of(context).colorScheme;

    Color bgColor = colors.errorContainer;
    Color borderColor = colors.error.withValues(alpha: 0.5);
    Color dotColor = colors.error;
    Color textColor = colors.onErrorContainer;
    String statusText = 'Chest Strap Disconnected';

    if (_chestStrapConnected) {
      if (isWorn) {
        bgColor = colors.tertiaryContainer;
        borderColor = colors.tertiary.withValues(alpha: 0.5);
        dotColor = colors.tertiary;
        textColor = colors.onTertiaryContainer;
        statusText = 'Chest strap connected and active';
      } else {
        bgColor = colors.secondaryContainer;
        borderColor = colors.secondary.withValues(alpha: 0.5);
        dotColor = colors.secondary;
        textColor = colors.onSecondaryContainer;
        statusText = 'Chest strap connected. Please put it on.';
      }
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: borderColor),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: dotColor, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(
            statusText,
            style: GoogleFonts.poppins(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: textColor,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBatteryWarning() {
    return GestureDetector(
      onTap: () => openAppSettings(),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.tertiaryContainer,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: Theme.of(
              context,
            ).colorScheme.tertiary.withValues(alpha: 0.5),
          ),
        ),
        child: Row(
          children: [
            Icon(
              Icons.battery_alert_rounded,
              color: Theme.of(context).colorScheme.tertiary,
              size: 18,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Battery settings may stop background updates. Tap to fix this.',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  color: Theme.of(context).colorScheme.onTertiaryContainer,
                ),
              ),
            ),
            Icon(
              Icons.chevron_right_rounded,
              color: Theme.of(context).colorScheme.tertiary,
              size: 18,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNotificationWarning() {
    return GestureDetector(
      onTap: () => openAppSettings(),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.errorContainer,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: Theme.of(context).colorScheme.error.withValues(alpha: 0.5),
          ),
        ),
        child: Row(
          children: [
            Icon(
              Icons.notifications_off_rounded,
              color: Theme.of(context).colorScheme.error,
              size: 18,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Anxiety check-in notifications are turned off. Tap to turn them on.',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  color: Theme.of(context).colorScheme.onErrorContainer,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            Icon(
              Icons.open_in_new_rounded,
              color: Colors.red.shade400,
              size: 18,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAdviceCard(double risk) {
    String title = "";
    String advice = "";
    IconData icon = Icons.lightbulb_outline_rounded;
    Color color = Colors.blue;

    // This banner describes what the live body readings show right now. The
    // separate forecast card below is responsible for future changes.
    final currentRisk = risk.clamp(0.0, 100.0).toDouble();
    if (currentRisk > 70) {
      title = "Take a moment to check in";
      advice =
          "Your recent body readings are stronger than usual. If you can, pause, breathe slowly, or contact someone you trust.";
      color = const Color(0xFFEF5350);
      icon = Icons.warning_amber_rounded;
    } else if (currentRisk <= 20) {
      title = "Your readings look settled";
      advice =
          "Your recent body readings are within your lower range. Keep doing what helps you feel comfortable.";
      color = const Color(0xFF4CAF50);
      icon = Icons.spa_rounded;
    } else if (currentRisk <= 45) {
      title = "A gentle check-in may help";
      advice =
          "Your recent readings have shifted a little. Consider a short pause and a slow breath.";
      color = const Color(0xFFFFA726);
      icon = Icons.self_improvement_rounded;
    } else {
      title = "Take a gentle pause";
      advice =
          "Your recent readings are above your usual range. Try some water, a short break, or a calming exercise if that feels helpful.";
      color = const Color(0xFFFF7043);
      icon = Icons.warning_amber_rounded;
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: color,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  advice,
                  style: GoogleFonts.poppins(
                    fontSize: 12,
                    color: Theme.of(
                      context,
                    ).colorScheme.onSurface.withValues(alpha: 0.8),
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChartsSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildForecastChart(),
        const SizedBox(height: 28),
        Text(
          'Past 30 Days',
          style: GoogleFonts.poppins(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Theme.of(context).colorScheme.onSurface,
          ),
        ),
        const SizedBox(height: 16),
        _buildHistoryCard(),
      ],
    );
  }

  String get _historyMetricLabel {
    switch (_historyMetric) {
      case 'mean_hr':
        return 'Heart rate';
      case 'mean_br':
        return 'Breathing';
      case 'mean_temp':
        return 'Temperature';
      case 'mean_motion':
        return 'Motion';
      default:
        return 'Anxiety score';
    }
  }

  String get _historyMetricUnit {
    switch (_historyMetric) {
      case 'mean_hr':
        return 'bpm';
      case 'mean_br':
        return 'br/min';
      case 'mean_temp':
        return '°C';
      case 'mean_motion':
        return 'g';
      default:
        return '';
    }
  }

  int get _historyAxisDecimals {
    if (_historyMetric == 'mean_motion') return 3;
    if (_historyMetric == 'mean_temp' || _historyMetric == 'mean_br') return 1;
    return 0;
  }

  double get _historyMinimumPadding {
    switch (_historyMetric) {
      case 'mean_hr':
        return 5;
      case 'mean_br':
        return 1;
      case 'mean_temp':
        return 0.1;
      case 'mean_motion':
        return 0.001;
      default:
        return 0;
    }
  }

  double get _historyMinimumInterval {
    switch (_historyMetric) {
      case 'mean_hr':
        return 1;
      case 'mean_br':
        return 0.5;
      case 'mean_temp':
        return 0.1;
      case 'mean_motion':
        return 0.001;
      default:
        return 20;
    }
  }

  Widget _buildHistoryCard() {
    if (_historyStatus == 'loading') {
      return const SizedBox(
        height: 190,
        child: Center(child: CircularProgressIndicator()),
      );
    }
    if (_historyStatus == 'empty') return _buildNoHistoryPlaceholder();
    if (_historyStatus == 'error') {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Could not load your history.'),
                  if (_historyMessage.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(
                      _historyMessage,
                      style: GoogleFonts.poppins(
                        fontSize: 10.5,
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            IconButton(
              onPressed: _fetchHistory,
              icon: const Icon(Icons.refresh_rounded),
              tooltip: 'Retry',
            ),
          ],
        ),
      );
    }

    final metricRows = _historyData
        .where((row) => row[_historyMetric] is num)
        .toList();
    if (metricRows.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          'No $_historyMetricLabel history is available yet.',
          style: GoogleFonts.poppins(
            fontSize: 12,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );
    }

    final spots = List<FlSpot>.generate(metricRows.length, (index) {
      final sensorValue = (metricRows[index][_historyMetric] as num).toDouble();
      final value = _historyMetric == 'mean_temp'
          ? sensorValue - 3
          : sensorValue;
      return FlSpot(index.toDouble(), value);
    });
    final values = spots.map((spot) => spot.y).toList();
    final fixedRiskAxis = _historyMetric == 'risk_index';
    final minValue = values.reduce(min);
    final maxValue = values.reduce(max);
    final padding = max((maxValue - minValue) * 0.18, _historyMinimumPadding);
    final minY = fixedRiskAxis ? 0.0 : max(0.0, minValue - padding);
    final maxY = fixedRiskAxis
        ? 100.0
        : (maxValue + padding <= minY ? minY + 1.0 : maxValue + padding);
    final axisInterval = fixedRiskAxis
        ? 20.0
        : max((maxY - minY) / 4, _historyMinimumInterval);

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Daily average · $_historyMetricLabel',
                  style: GoogleFonts.poppins(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
              ),
              DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _historyMetric,
                  isDense: true,
                  style: GoogleFonts.poppins(
                    fontSize: 11,
                    color: AppTheme.kPrimaryDeep,
                  ),
                  items: const [
                    DropdownMenuItem(
                      value: 'risk_index',
                      child: Text('Anxiety score'),
                    ),
                    DropdownMenuItem(
                      value: 'mean_hr',
                      child: Text('Heart rate'),
                    ),
                    DropdownMenuItem(
                      value: 'mean_br',
                      child: Text('Breathing'),
                    ),
                    DropdownMenuItem(
                      value: 'mean_temp',
                      child: Text('Temperature'),
                    ),
                    DropdownMenuItem(
                      value: 'mean_motion',
                      child: Text('Motion'),
                    ),
                  ],
                  onChanged: (value) {
                    if (value != null) setState(() => _historyMetric = value);
                  },
                ),
              ),
              IconButton(
                onPressed: _fetchHistory,
                icon: const Icon(Icons.refresh_rounded, size: 19),
                tooltip: 'Refresh',
              ),
            ],
          ),
          const SizedBox(height: 14),
          SizedBox(
            height: 190,
            child: LineChart(
              LineChartData(
                minX: 0,
                maxX: max(1, metricRows.length - 1).toDouble(),
                minY: minY,
                maxY: maxY,
                borderData: FlBorderData(show: false),
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  getDrawingHorizontalLine: (_) => FlLine(
                    color: Colors.grey.withValues(alpha: 0.10),
                    strokeWidth: 1,
                  ),
                ),
                titlesData: FlTitlesData(
                  topTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  rightTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 46,
                      interval: axisInterval,
                      getTitlesWidget: (value, meta) => Text(
                        value.toStringAsFixed(_historyAxisDecimals),
                        style: GoogleFonts.poppins(
                          fontSize: 8.5,
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 24,
                      getTitlesWidget: (value, meta) {
                        final index = value.round();
                        if ((value - index).abs() > 0.01) {
                          return const SizedBox.shrink();
                        }
                        if (index < 0 || index >= metricRows.length) {
                          return const SizedBox.shrink();
                        }
                        final every = max(1, (metricRows.length / 5).ceil());
                        if (index % every != 0 &&
                            index != metricRows.length - 1) {
                          return const SizedBox.shrink();
                        }
                        final date = metricRows[index]['date'] as String? ?? '';
                        final parsedDate = DateTime.tryParse(date);
                        return Text(
                          parsedDate == null
                              ? date
                              : DateFormat('d MMM').format(parsedDate),
                          style: GoogleFonts.poppins(
                            fontSize: 8,
                            color: Theme.of(
                              context,
                            ).colorScheme.onSurfaceVariant,
                          ),
                        );
                      },
                    ),
                  ),
                ),
                lineBarsData: [
                  LineChartBarData(
                    spots: spots,
                    isCurved: true,
                    color: AppTheme.kPrimaryDeep,
                    barWidth: 3,
                    dotData: const FlDotData(show: true),
                    belowBarData: BarAreaData(
                      show: true,
                      color: AppTheme.kPrimaryDeep.withValues(alpha: 0.10),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${metricRows.length} day${metricRows.length == 1 ? '' : 's'} of data'
            '${_historyMetricUnit.isEmpty ? '' : ' · $_historyMetricUnit'}',
            style: GoogleFonts.poppins(
              fontSize: 10,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoHistoryPlaceholder() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Icon(
            Icons.show_chart_rounded,
            size: 48,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
          const SizedBox(height: 12),
          Text(
            'Historical trends will appear here',
            style: GoogleFonts.poppins(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Keep using Aura with your chest strap to build your history.',
            style: GoogleFonts.poppins(
              fontSize: 12,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KPI Card
// ═══════════════════════════════════════════════════════════════════════════════

class _KpiCard extends StatelessWidget {
  final String label;
  final String value;
  final String unit;
  final IconData icon;
  final String status;
  final Color statusColor;
  final List<Color> gradient;

  const _KpiCard({
    required this.label,
    required this.value,
    required this.unit,
    required this.icon,
    required this.status,
    required this.statusColor,
    required this.gradient,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: gradient.first.withValues(alpha: 0.12),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Icon + status badge
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  gradient: LinearGradient(colors: gradient),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: Colors.white, size: 22),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  status,
                  style: GoogleFonts.poppins(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: statusColor,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),

          // Value
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 400),
                transitionBuilder: (child, anim) => FadeTransition(
                  opacity: anim,
                  child: SlideTransition(
                    position: Tween<Offset>(
                      begin: const Offset(0, 0.3),
                      end: Offset.zero,
                    ).animate(anim),
                    child: child,
                  ),
                ),
                child: Text(
                  value,
                  key: ValueKey(value),
                  style: GoogleFonts.poppins(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    color: Theme.of(context).colorScheme.onSurface,
                    height: 1.0,
                  ),
                ),
              ),
              const SizedBox(width: 4),
              Padding(
                padding: const EdgeInsets.only(bottom: 3),
                child: Text(
                  unit,
                  style: GoogleFonts.poppins(
                    fontSize: 12,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),

          // Label
          Text(
            label,
            style: GoogleFonts.poppins(
              fontSize: 12,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
