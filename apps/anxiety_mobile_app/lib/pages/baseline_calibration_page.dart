import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/api_service.dart';
import '../services/chest_strap_service.dart';
import 'main_navigation_page.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Baseline Calibration Page
//
// Collects three minutes of one-feature-packet-per-second physiological data,
// then computes baseline statistics for the exact 10-feature server contract.
//
// Feature order:
// [meanHR, meanRR, SDNN, RMSSD, meanBR, stdBR,
//  meanTemp, stdTemp, meanAccMag, stdAccMag]
// ─────────────────────────────────────────────────────────────────────────────

class BaselineCalibrationPage extends StatefulWidget {
  final String userId;
  const BaselineCalibrationPage({super.key, required this.userId});

  @override
  State<BaselineCalibrationPage> createState() =>
      _BaselineCalibrationPageState();
}

class _BaselineCalibrationPageState extends State<BaselineCalibrationPage>
    with TickerProviderStateMixin {
  // ── Constants ──────────────────────────────────────────────────
  static const int _totalSeconds = 180; // 3 minutes of resting data
  static const int _minimumReadings = 120;
  // Three one-minute averages can otherwise produce near-zero variation. A
  // tiny divisor turns ordinary calm movement into an extreme model input.
  // These floors are deliberately small relative to real stress changes.
  static const List<double> _minimumFeatureStds = [
    1.0, // heart rate (bpm)
    15.0, // beat interval (ms)
    2.0, // SDNN (ms)
    2.0, // RMSSD (ms)
    0.5, // breathing rate (breaths/min)
    0.10, // breathing variation
    0.05, // temperature (°C)
    0.01, // temperature variation
    0.01, // movement level (g)
    0.005, // movement variation (g)
  ];

  // ── State ──────────────────────────────────────────────────────
  _Phase _phase = _Phase.instructions;
  int _elapsedSeconds = 0;
  bool _uploading = false;
  String? _errorMessage;

  Timer? _collectionTimer;
  StreamSubscription<ChestStrapReading>? _collectionSubscription;
  int? _lastCollectedTimestamp;
  final List<ChestStrapReading> _collectedReadings = [];

  // Calibrated physiological features
  double _restingHR = 70.0;
  double _restingBR = 16.0;
  double _restingTemp = 36.6;

  // ── Animation ──────────────────────────────────────────────────
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;
  late AnimationController _waveController;
  late Animation<double> _waveAnimation;
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.92, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();
    _waveAnimation = CurvedAnimation(
      parent: _waveController,
      curve: Curves.linear,
    );

    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );
    _fadeController.forward();
  }

  @override
  void dispose() {
    _collectionTimer?.cancel();
    _collectionSubscription?.cancel();
    _pulseController.dispose();
    _waveController.dispose();
    _fadeController.dispose();
    super.dispose();
  }

  // ═══════════════════════════════════════════════════════════════
  // Data Collection
  // ═══════════════════════════════════════════════════════════════

  void _startCalibration() {
    final lastReading = ChestStrapService().lastReading;
    if (!ChestStrapService().isConnected) {
      setState(() {
        _errorMessage =
            'No chest strap readings are available. Connect the chest strap.';
        _phase = _Phase.error;
      });
      return;
    }

    if (lastReading == null || !lastReading.isWorn) {
      setState(() {
        _errorMessage =
            'The chest strap is connected but not being worn. Put it on properly before starting.';
        _phase = _Phase.error;
      });
      return;
    }

    _collectedReadings.clear();
    _lastCollectedTimestamp = null;

    // Collect the packets that actually arrive instead of re-adding a stale
    // lastReading once per second when BLE has stopped delivering data.
    _collectionSubscription?.cancel();
    _collectionSubscription = ChestStrapService().readingsStream.listen((
      reading,
    ) {
      if (_phase != _Phase.collecting || !reading.isWorn) return;
      if (_lastCollectedTimestamp == reading.timestamp) return;
      _lastCollectedTimestamp = reading.timestamp;
      _collectedReadings.add(reading);
    });

    setState(() {
      _phase = _Phase.collecting;
      _elapsedSeconds = 0;
      _errorMessage = null;
    });

    _collectionTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      if (!ChestStrapService().isConnected) {
        timer.cancel();
        _collectionSubscription?.cancel();
        _collectionSubscription = null;
        setState(() {
          _errorMessage = 'The chest strap disconnected during setup.';
          _phase = _Phase.error;
        });
        return;
      }

      setState(() => _elapsedSeconds++);

      if (_elapsedSeconds >= _totalSeconds) {
        timer.cancel();
        _finishCollection();
      }
    });
  }

  void _finishCollection() {
    _collectionSubscription?.cancel();
    _collectionSubscription = null;
    setState(() => _phase = _Phase.calculating);
    // Give the UI one frame to update before the heavy computation
    Future.delayed(const Duration(milliseconds: 300), _computeAndUpload);
  }

  // ═══════════════════════════════════════════════════════════════
  // Statistics & Upload
  // ═══════════════════════════════════════════════════════════════

  Future<void> _computeAndUpload() async {
    if (_collectedReadings.length < _minimumReadings) {
      setState(() {
        _errorMessage =
            'Aura received only ${_collectedReadings.length} usable readings. '
            'At least $_minimumReadings are needed. Check the connection and try again.';
        _phase = _Phase.error;
      });
      return;
    }

    // We have up to 180 readings. Let's slice into 3 windows of up to 60 readings.
    final int windowLength = 60;
    final List<List<double>> featuresPerWindow = [];

    for (int w = 0; w < 3; w++) {
      final int start = w * windowLength;
      final int end = start + windowLength;

      if (start >= _collectedReadings.length) break;

      final windowReadings = _collectedReadings.sublist(
        start,
        end > _collectedReadings.length ? _collectedReadings.length : end,
      );

      // Average the 10 features for this window
      if (windowReadings.isNotEmpty) {
        double meanHR =
            windowReadings.map((r) => r.meanHR).reduce((a, b) => a + b) /
            windowReadings.length;
        double meanRR =
            windowReadings.map((r) => r.meanRR).reduce((a, b) => a + b) /
            windowReadings.length;
        double sdnn =
            windowReadings.map((r) => r.sdnn).reduce((a, b) => a + b) /
            windowReadings.length;
        double rmssd =
            windowReadings.map((r) => r.rmssd).reduce((a, b) => a + b) /
            windowReadings.length;
        double meanBR =
            windowReadings.map((r) => r.meanBR).reduce((a, b) => a + b) /
            windowReadings.length;
        double stdBR =
            windowReadings.map((r) => r.stdBR).reduce((a, b) => a + b) /
            windowReadings.length;
        double meanTemp =
            windowReadings.map((r) => r.meanTemp).reduce((a, b) => a + b) /
            windowReadings.length;
        double stdTemp =
            windowReadings.map((r) => r.stdTemp).reduce((a, b) => a + b) /
            windowReadings.length;
        double meanAccMag =
            windowReadings.map((r) => r.meanAccMag).reduce((a, b) => a + b) /
            windowReadings.length;
        double stdAccMag =
            windowReadings.map((r) => r.stdAccMag).reduce((a, b) => a + b) /
            windowReadings.length;

        featuresPerWindow.add([
          meanHR,
          meanRR,
          sdnn,
          rmssd,
          meanBR,
          stdBR,
          meanTemp,
          stdTemp,
          meanAccMag,
          stdAccMag,
        ]);
      }
    }

    if (featuresPerWindow.isEmpty) {
      setState(() {
        _errorMessage = 'Aura did not receive enough readings. Please try again.';
        _phase = _Phase.error;
      });
      return;
    }

    // We now have N windows (e.g. 3) of 10 features each.
    // Let's compute mean and std dev across these windows for each of the 10 features.
    final List<double> finalMeans = List.filled(10, 0.0);
    final List<double> finalStds = List.filled(10, 0.0);

    for (int f = 0; f < 10; f++) {
      final List<double> vals = featuresPerWindow.map((win) => win[f]).toList();
      final double meanVal = vals.reduce((a, b) => a + b) / vals.length;
      finalMeans[f] = meanVal;

      if (vals.length < 2) {
        finalStds[f] = _minimumFeatureStds[f];
      } else {
        final double varSum = vals
            .map((x) => (x - meanVal) * (x - meanVal))
            .reduce((a, b) => a + b);
        double stdVal = sqrt(varSum / vals.length);
        finalStds[f] = max(stdVal, _minimumFeatureStds[f]);
      }
    }

    setState(() {
      _uploading = true;
      _restingHR = finalMeans[0];
      _restingBR = finalMeans[4];
      _restingTemp = finalMeans[6];
    });

    // ── Step 3: Upload to normalization endpoint ─────────────────
    bool ok = false;
    try {
      ok = await ApiService.setNormalizationParams(
        userId: widget.userId,
        bMean: finalMeans,
        bStd: finalStds,
        baselineWindows: featuresPerWindow,
      );
    } catch (e) {
      ok = false;
      debugPrint('Calibration upload error: $e');
    }

    // ── Step 4: Persist the calibration flag ────────────────────
    if (ok) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('calibration_complete', true);
    }

    if (!mounted) return;
    setState(() {
      _uploading = false;
      _phase = ok ? _Phase.done : _Phase.error;
      if (!ok) {
        _errorMessage =
            'Could not save your setup. Check your internet connection and try again.';
      }
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // Navigation
  // ═══════════════════════════════════════════════════════════════

  void _proceed() {
    Navigator.pushReplacement(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) =>
            MainNavigationPage(userId: widget.userId),
        transitionsBuilder: (context, anim, secondaryAnimation, child) =>
            FadeTransition(opacity: anim, child: child),
        transitionDuration: const Duration(milliseconds: 700),
      ),
    );
  }

  /// Skip the setup screen for now. Forecasting remains unavailable until
  /// normalization parameters are supplied by a later calibration.
  void _skip() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('calibration_complete', true);
    if (mounted) _proceed();
  }

  // ═══════════════════════════════════════════════════════════════
  // BUILD
  // ═══════════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF667eea), Color(0xFF764ba2)],
          ),
        ),
        child: SafeArea(
          child: FadeTransition(opacity: _fadeAnimation, child: _buildBody()),
        ),
      ),
    );
  }

  Widget _buildBody() {
    switch (_phase) {
      case _Phase.instructions:
        return _buildInstructionsView();
      case _Phase.collecting:
        return _buildCollectingView();
      case _Phase.calculating:
        return _buildCalculatingView();
      case _Phase.done:
        return _buildDoneView();
      case _Phase.error:
        return _buildErrorView();
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Phase 1 — Instructions
  // ─────────────────────────────────────────────────────────────

  Widget _buildInstructionsView() {
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: 20),

          // Pulsing heart icon
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) => Transform.scale(
              scale: _pulseAnimation.value,
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withValues(alpha: 0.15),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.3),
                    width: 2,
                  ),
                ),
                child: const Icon(
                  Icons.monitor_heart_outlined,
                  color: Colors.white,
                  size: 52,
                ),
              ),
            ),
          ),

          const SizedBox(height: 32),

          Text(
            'Set Up Your Resting Level',
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 26,
              fontWeight: FontWeight.w700,
              color: Colors.white,
              letterSpacing: -0.3,
            ),
          ),

          const SizedBox(height: 10),

          Text(
            'Before Aura can understand changes in your anxiety level, '
            'it needs to learn what your body looks like at rest.',
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 14,
              color: Colors.white.withValues(alpha: 0.85),
              height: 1.6,
            ),
          ),

          const SizedBox(height: 36),

          // Step cards
          _StepCard(
            number: '1',
            icon: Icons.self_improvement_rounded,
            title: 'Find a comfortable position',
            description:
                'Sit or lie down in a relaxed state. Try not to move around.',
          ),
          const SizedBox(height: 14),
          _StepCard(
            number: '2',
            icon: Icons.timer_outlined,
            title: 'Wait 3 minutes',
            description:
                'Aura will learn what your readings look like while you are calm.',
          ),
          const SizedBox(height: 14),
          _StepCard(
            number: '3',
            icon: Icons.cloud_upload_outlined,
            title: 'Your results are saved securely',
            description:
                'Aura saves a summary of your resting readings so future results can be personal to you.',
          ),

          const SizedBox(height: 28),

          // Calm state warning card
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: Colors.amber.withValues(alpha: 0.3),
                width: 1.5,
              ),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(
                  Icons.info_outline_rounded,
                  color: Colors.amber,
                  size: 20,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'IMPORTANT: Start only when you feel calm and relaxed. '
                    'If you feel stressed or have been active, skip this for now and return later. '
                    'Starting while stressed can make future results less accurate.',
                    style: GoogleFonts.poppins(
                      fontSize: 11.5,
                      color: Colors.white.withValues(alpha: 0.9),
                      height: 1.5,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 32),

          _buildChestStrapControls(),

          const SizedBox(height: 20),

          // Start button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _startCalibration,
              icon: const Icon(Icons.play_circle_outline_rounded, size: 22),
              label: Text(
                'Start 3-Minute Setup',
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: const Color(0xFF667eea),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18),
                ),
                elevation: 0,
              ),
            ),
          ),

          const SizedBox(height: 14),

          // Skip link
          TextButton(
            onPressed: _skip,
            child: Text(
              'Skip setup for now',
              style: GoogleFonts.poppins(
                fontSize: 12,
                color: Colors.white.withValues(alpha: 0.65),
                decoration: TextDecoration.underline,
                decorationColor: Colors.white.withValues(alpha: 0.5),
              ),
            ),
          ),

          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _buildChestStrapControls() {
    final chestStrap = ChestStrapService();
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.25)),
      ),
      child: ValueListenableBuilder<ChestStrapState>(
        valueListenable: chestStrap.connectionState,
        builder: (context, state, _) {
          return ListTile(
            contentPadding: EdgeInsets.zero,
            leading: Icon(
              chestStrap.isConnected
                  ? Icons.bluetooth_connected_rounded
                  : Icons.bluetooth_searching_rounded,
              color: Colors.white,
            ),
            title: Text(
              chestStrap.isConnected
                  ? 'Chest strap readings connected'
                  : 'Connect real chest strap',
              style: GoogleFonts.poppins(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            trailing: chestStrap.isConnected
                ? null
                : TextButton(
                    onPressed: chestStrap.startScan,
                    child: const Text(
                      'Find strap',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
          );
        },
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Phase 2 — Collecting
  // ─────────────────────────────────────────────────────────────

  Widget _buildCollectingView() {
    final double progress = _elapsedSeconds / _totalSeconds;
    final int remaining = _totalSeconds - _elapsedSeconds;
    final int minutes = remaining ~/ 60;
    final int seconds = remaining % 60;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            'Learning Your Resting Level',
            style: GoogleFonts.poppins(
              fontSize: 24,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 8),

          Text(
            'Stay still and breathe normally.',
            style: GoogleFonts.poppins(
              fontSize: 14,
              color: Colors.white.withValues(alpha: 0.8),
            ),
          ),

          const SizedBox(height: 48),

          // Animated waveform visualizer
          _WaveformVisualizer(animation: _waveAnimation),

          const SizedBox(height: 48),

          // Circular progress countdown
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return Stack(
                alignment: Alignment.center,
                children: [
                  // Outer glow
                  Container(
                    width: 180,
                    height: 180,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withValues(
                        alpha: 0.06 * _pulseAnimation.value,
                      ),
                    ),
                  ),
                  // Progress ring
                  SizedBox(
                    width: 160,
                    height: 160,
                    child: CircularProgressIndicator(
                      value: progress,
                      strokeWidth: 8,
                      backgroundColor: Colors.white.withValues(alpha: 0.2),
                      valueColor: const AlwaysStoppedAnimation(Colors.white),
                      strokeCap: StrokeCap.round,
                    ),
                  ),
                  // Time display
                  Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '$minutes:${seconds.toString().padLeft(2, '0')}',
                        style: GoogleFonts.poppins(
                          fontSize: 40,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                          letterSpacing: -1,
                        ),
                      ),
                      Text(
                        'remaining',
                        style: GoogleFonts.poppins(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.7),
                        ),
                      ),
                    ],
                  ),
                ],
              );
            },
          ),

          const SizedBox(height: 40),

          // Live sample count
          _buildLiveStats(),

          const SizedBox(height: 32),

          // Progress bar with label
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Data collected',
                    style: GoogleFonts.poppins(
                      fontSize: 12,
                      color: Colors.white.withValues(alpha: 0.7),
                    ),
                  ),
                  Text(
                    '${(progress * 100).toStringAsFixed(0)}%',
                    style: GoogleFonts.poppins(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: LinearProgressIndicator(
                  value: progress,
                  minHeight: 8,
                  backgroundColor: Colors.white.withValues(alpha: 0.2),
                  valueColor: const AlwaysStoppedAnimation(Colors.white),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLiveStats() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _statChip(
            Icons.timeline_rounded,
            '${_collectedReadings.length}',
            'Readings',
          ),
          _vertDivider(),
          _statChip(Icons.graphic_eq_rounded, '10', 'Types'),
          _vertDivider(),
          _statChip(Icons.speed_rounded, '1/sec', 'Speed'),
        ],
      ),
    );
  }

  Widget _statChip(IconData icon, String value, String label) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: Colors.white.withValues(alpha: 0.8), size: 20),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.poppins(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.poppins(
            fontSize: 10,
            color: Colors.white.withValues(alpha: 0.65),
          ),
        ),
      ],
    );
  }

  Widget _vertDivider() {
    return Container(
      width: 1,
      height: 44,
      color: Colors.white.withValues(alpha: 0.2),
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Phase 3 — Calculating / Uploading
  // ─────────────────────────────────────────────────────────────

  Widget _buildCalculatingView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 28),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(
              width: 72,
              height: 72,
              child: CircularProgressIndicator(
                color: Colors.white,
                strokeWidth: 5,
                strokeCap: StrokeCap.round,
              ),
            ),
            const SizedBox(height: 32),
            Text(
              _uploading ? 'Saving securely...' : 'Preparing your results...',
              style: GoogleFonts.poppins(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              _uploading
                  ? 'Saving your personal resting level securely.'
                  : 'Preparing your resting-level summary.',
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: Colors.white.withValues(alpha: 0.75),
                height: 1.6,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Phase 4 — Done (success)
  // ─────────────────────────────────────────────────────────────

  Widget _buildDoneView() {
    final int totalPackets = _collectedReadings.length;
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
      child: Column(
        children: [
          // Success icon
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) => Transform.scale(
              scale: _pulseAnimation.value,
              child: Container(
                width: 110,
                height: 110,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withValues(alpha: 0.2),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.4),
                    width: 2,
                  ),
                ),
                child: const Icon(
                  Icons.check_circle_outline_rounded,
                  color: Colors.white,
                  size: 60,
                ),
              ),
            ),
          ),

          const SizedBox(height: 28),

          Text(
            'Setup Complete!',
            style: GoogleFonts.poppins(
              fontSize: 26,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 10),

          Text(
            'Your resting level was saved securely. '
            'Aura can now give results that are personal to you.',
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 14,
              color: Colors.white.withValues(alpha: 0.85),
              height: 1.6,
            ),
          ),

          const SizedBox(height: 32),

          // Summary stats card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.white.withValues(alpha: 0.25)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Setup Summary',
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 16),
                _summaryRow('Duration', '${_totalSeconds ~/ 60} minutes'),
                _summaryRow('Usable readings', '$totalPackets'),
                _summaryRow('Reading types', '10'),
                _summaryRow(
                  'Resting heart rate',
                  '${_restingHR.toStringAsFixed(1)} beats/min',
                ),
                _summaryRow(
                  'Resting breathing rate',
                  '${_restingBR.toStringAsFixed(1)} breaths/min',
                ),
                _summaryRow(
                  'Resting Temp',
                  '${_restingTemp.toStringAsFixed(2)} °C',
                ),
                _summaryRow('Save status', 'Saved ✓'),
              ],
            ),
          ),

          const SizedBox(height: 32),

          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _proceed,
              icon: const Icon(Icons.dashboard_outlined),
              label: Text(
                'Go to Dashboard',
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: const Color(0xFF667eea),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18),
                ),
                elevation: 0,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _summaryRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: GoogleFonts.poppins(
              fontSize: 13,
              color: Colors.white.withValues(alpha: 0.75),
            ),
          ),
          Text(
            value,
            style: GoogleFonts.poppins(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Phase 5 — Error
  // ─────────────────────────────────────────────────────────────

  Widget _buildErrorView() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withValues(alpha: 0.15),
              border: Border.all(color: Colors.white.withValues(alpha: 0.3)),
            ),
            child: const Icon(
              Icons.cloud_off_rounded,
              color: Colors.white,
              size: 50,
            ),
          ),

          const SizedBox(height: 28),

          Text(
            'Setup Failed',
            style: GoogleFonts.poppins(
              fontSize: 24,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 10),

          Text(
            _errorMessage ?? 'An unexpected error occurred.',
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 14,
              color: Colors.white.withValues(alpha: 0.8),
              height: 1.6,
            ),
          ),

          const SizedBox(height: 36),

          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _collectedReadings.length >= _minimumReadings
                  ? _computeAndUpload
                  : () {
                      setState(() {
                        _phase = _Phase.instructions;
                        _errorMessage = null;
                      });
                    },
              icon: const Icon(Icons.refresh_rounded),
              label: Text(
                _collectedReadings.length >= _minimumReadings
                    ? 'Try Saving Again'
                    : 'Back to Setup',
                style: GoogleFonts.poppins(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: const Color(0xFF667eea),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18),
                ),
                elevation: 0,
              ),
            ),
          ),

          const SizedBox(height: 14),

          TextButton(
            onPressed: _skip,
            child: Text(
              'Skip and continue without setup',
              style: GoogleFonts.poppins(
                fontSize: 12,
                color: Colors.white.withValues(alpha: 0.65),
                decoration: TextDecoration.underline,
                decorationColor: Colors.white.withValues(alpha: 0.5),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase Enum
// ─────────────────────────────────────────────────────────────────────────────

enum _Phase { instructions, collecting, calculating, done, error }

// ─────────────────────────────────────────────────────────────────────────────
// Step Card widget
// ─────────────────────────────────────────────────────────────────────────────

class _StepCard extends StatelessWidget {
  final String number;
  final IconData icon;
  final String title;
  final String description;

  const _StepCard({
    required this.number,
    required this.icon,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Step number bubble
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.25),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                number,
                style: GoogleFonts.poppins(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      icon,
                      color: Colors.white.withValues(alpha: 0.8),
                      size: 18,
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        title,
                        style: GoogleFonts.poppins(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  description,
                  style: GoogleFonts.poppins(
                    fontSize: 12,
                    color: Colors.white.withValues(alpha: 0.75),
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Animated ECG-style waveform visualizer
// ─────────────────────────────────────────────────────────────────────────────

class _WaveformVisualizer extends StatelessWidget {
  final Animation<double> animation;
  const _WaveformVisualizer({required this.animation});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 80,
      child: AnimatedBuilder(
        animation: animation,
        builder: (context, child) => CustomPaint(
          painter: _WaveformPainter(phase: animation.value),
          size: const Size(double.infinity, 80),
        ),
      ),
    );
  }
}

class _WaveformPainter extends CustomPainter {
  final double phase;
  _WaveformPainter({required this.phase});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.85)
      ..strokeWidth = 2.2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final path = Path();
    final double w = size.width;
    final double h = size.height;
    final double cx = h / 2;

    const int segments = 200;
    for (int i = 0; i <= segments; i++) {
      final double x = (i / segments) * w;
      final double t = (i / segments + phase) * 2 * pi;
      final double ecgPhase = (i / segments + phase) % 1.0;

      double y;
      // Synthesise a simplified ECG shape within each beat period (~0.125 of width)
      final double beatPhase = (ecgPhase * 8) % 1.0;
      if (beatPhase < 0.05) {
        // P wave
        y = cx - 8 * sin(beatPhase / 0.05 * pi);
      } else if (beatPhase < 0.1) {
        y = cx; // PR segment
      } else if (beatPhase < 0.12) {
        // Q dip
        y = cx + 6 * sin((beatPhase - 0.1) / 0.02 * pi);
      } else if (beatPhase < 0.15) {
        // R peak
        y = cx - 28 * sin((beatPhase - 0.12) / 0.03 * pi);
      } else if (beatPhase < 0.18) {
        // S wave
        y = cx + 10 * sin((beatPhase - 0.15) / 0.03 * pi);
      } else if (beatPhase < 0.35) {
        // ST segment + T wave
        y = cx - 6 * sin((beatPhase - 0.18) / 0.17 * pi);
      } else {
        y = cx + 0.5 * sin(t * 2); // baseline wander
      }

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_WaveformPainter old) => old.phase != phase;
}
