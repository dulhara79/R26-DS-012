import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/fusion_risk_service.dart';
import '../services/chest_strap_service.dart';
import '../services/api_service.dart';
import '../services/anxiety_feedback_service.dart';
import '../services/anxiety_level_update_throttle.dart';

/// Home Page — the first tab the user sees.
///
/// Displays:
///   • Aura branding & subtitle
///   • Meditation hero image (from assets)
///   • Overall anxiety status card (combined physiological + phenotyping risk)
///   • Notification bell for anxiety escalation alerts
class HomePage extends StatefulWidget {
  final String? userId;
  const HomePage({super.key, this.userId});

  @override
  State<HomePage> createState() => HomePageState();
}

class HomePageState extends State<HomePage> with TickerProviderStateMixin {
  // ── Chest Strap Service ──────────────────────────────────────
  final ChestStrapService _chestStrap = ChestStrapService();
  ChestStrapReading? _lastReading;
  StreamSubscription<ChestStrapReading>? _readingSubscription;

  Map<String, dynamic>? _weeklySummary;
  bool _weeklyLoading = false;
  String? _weeklyError;

  // ── Notification state ─────────────────────────────────────
  final List<String> _notifications = [];
  final AnxietyLevelUpdateThrottle _notificationThrottle =
      AnxietyLevelUpdateThrottle();
  Timer? _notificationThrottleTimer;
  bool _hasUnread = false;

  // ── Animation ──────────────────────────────────────────────
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    FusionRiskService.instance.startPolling();
    FusionRiskService.instance.latest.addListener(_onFusionRiskChanged);
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );
    _fadeController.forward();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.85, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    // Listen to real chest strap data
    _lastReading = _chestStrap.hasLiveWornReading
        ? _chestStrap.lastReading
        : null;
    _notificationThrottle.seed(_labelForScore(_overallRisk));
    _readingSubscription = _chestStrap.readingsStream.listen((reading) {
      if (mounted) {
        setState(() => _lastReading = reading);
        _observeOverallLevel();
      }
    });

    // Listen for connection state changes so the UI updates when the
    // chest strap connects or disconnects.
    _chestStrap.connectionState.addListener(_onConnectionChanged);
    _chestStrap.liveReadingAvailable.addListener(_onLiveAvailabilityChanged);
    AnxietyFeedbackService().combinedRisk.addListener(_onCombinedRiskChanged);
    _loadWeeklySummary();
  }

  void _onConnectionChanged() {
    if (mounted) {
      setState(() {
        if (!_chestStrap.isConnected) _lastReading = null;
      });
      _observeOverallLevel();
    }
  }

  void _onLiveAvailabilityChanged() {
    if (mounted && !_chestStrap.hasLiveWornReading) {
      setState(() => _lastReading = null);
      _observeOverallLevel();
    }
  }

  void _onCombinedRiskChanged() {
    if (mounted) {
      setState(() {});
      _observeOverallLevel();
    }
  }

  Future<void> _loadWeeklySummary() async {
    final userId = widget.userId;
    if (userId == null || userId.isEmpty) {
      if (mounted) {
        setState(
          () => _weeklyError = 'Your weekly summary is not available yet.',
        );
      }
      return;
    }
    if (mounted) {
      setState(() {
        _weeklyLoading = true;
        _weeklyError = null;
      });
    }
    try {
      final onlineSummary = await ApiService.getWeeklyFeedbackSummary(userId);
      final summary = onlineSummary['status'] == 'success'
          ? onlineSummary
          : await AnxietyFeedbackService.getLocalWeeklySummary(userId);
      if (mounted) {
        setState(() {
          _weeklySummary = summary;
          _weeklyLoading = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _weeklyLoading = false;
          _weeklyError =
              'Could not load your weekly summary. Please try again.';
        });
      }
    }
  }

  @override
  void dispose() {
    _chestStrap.connectionState.removeListener(_onConnectionChanged);
    _chestStrap.liveReadingAvailable.removeListener(_onLiveAvailabilityChanged);
    AnxietyFeedbackService().combinedRisk.removeListener(
      _onCombinedRiskChanged,
    );
    _readingSubscription?.cancel();
    _notificationThrottleTimer?.cancel();
    _fadeController.dispose();
    _pulseController.dispose();
    FusionRiskService.instance.latest.removeListener(_onFusionRiskChanged);
    super.dispose();
  }

void _onFusionRiskChanged() {
    if (mounted) setState(() {});
}
  // ── Combined Risk Logic ─────────────────────────────────────
  bool get _hasLiveReading =>
      _chestStrap.hasLiveWornReading && (_lastReading?.isWorn ?? false);

  /// Uses the combined score when it is fresh. If only the chest strap is
  /// available, that reading becomes the current overall score.
  double? get _overallRisk {
    final backendRisk = FusionRiskService.instance.latest.value;
    if (backendRisk != null && backendRisk.hasScore) {
      return backendRisk.scoreOutOf100;
    }
    final combinedRisk = AnxietyFeedbackService().latestFusionRisk;
    if (combinedRisk != null) return combinedRisk;
    return _hasLiveReading ? _lastReading!.riskScore : null;
}

  bool get _hasOverallRisk => _overallRisk != null;

  String _labelForScore(double? score) {
    if (score == null) return 'Unavailable';
    if (score <= 20) return 'Low';
    if (score <= 45) return 'Moderate';
    if (score <= 70) return 'Elevated';
    return 'High';
  }

  Color _overallColor(double score) {
    if (score <= 20) return const Color(0xFF4CAF50);
    if (score <= 45) return const Color(0xFFFFA726);
    if (score <= 70) return const Color(0xFFFF7043);
    return const Color(0xFFEF5350);
  }

  IconData _overallIcon(double score) {
    if (!_hasOverallRisk) return Icons.sensors_off_rounded;
    if (score <= 20) return Icons.sentiment_very_satisfied_rounded;
    if (score <= 45) return Icons.sentiment_satisfied_rounded;
    if (score <= 70) return Icons.sentiment_neutral_rounded;
    return Icons.sentiment_very_dissatisfied_rounded;
  }

  String _overallMessage(double score) {
    if (!_hasOverallRisk) {
      return 'Connect and wear the chest strap to see your current readings.';
    }
    if (score <= 20) {
      return 'Your recent readings look settled. Keep doing what helps you feel comfortable.';
    } else if (score <= 45) {
      return 'Your recent readings have shifted a little. A slow breath or short pause may feel helpful.';
    } else if (score <= 70) {
      return 'A gentle pause may help. Try a calming activity if that feels right for you.';
    } else {
      return 'Take a moment to check in with yourself. Breathe slowly, and contact someone you trust if you would like support.';
    }
  }

  void _addNotification(String msg) {
    if (!mounted) return;
    setState(() {
      _notifications.insert(0, msg);
      _hasUnread = true;
    });
  }

  void _observeOverallLevel() {
    final now = DateTime.now();
    final update = _notificationThrottle.observe(
      _labelForScore(_overallRisk),
      now,
    );
    if (update != null) _addNotification(update.message);
    _schedulePendingNotification(now);
  }

  void _schedulePendingNotification(DateTime now) {
    _notificationThrottleTimer?.cancel();
    final delay = _notificationThrottle.delayUntilFlush(now);
    if (delay == null) return;
    _notificationThrottleTimer = Timer(delay, () {
      if (!mounted) return;
      final update = _notificationThrottle.flush(DateTime.now());
      if (update != null) _addNotification(update.message);
      _schedulePendingNotification(DateTime.now());
    });
  }

  void _clearNotifications() {
    if (!mounted) return;
    setState(() {
      _notifications.clear();
      _hasUnread = false;
    });
  }

  void _showNotifications() {
    setState(() => _hasUnread = false);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _NotificationsSheet(
        notifications: List.unmodifiable(_notifications),
        onClear: _clearNotifications,
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  // BUILD
  // ═══════════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    final risk = _overallRisk ?? 0.0;
    final riskCol = _hasOverallRisk ? _overallColor(risk) : Colors.grey;
    final label = _labelForScore(_overallRisk);

    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: Colors.transparent,
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
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 16),

                  // ── Header Row with Notification Bell ──
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Aura',
                            style: GoogleFonts.poppins(
                              fontSize: 30,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                              letterSpacing: -0.5,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Track your inner world and find balance.',
                            style: GoogleFonts.poppins(
                              fontSize: 14,
                              color: Colors.white.withValues(alpha: 0.8),
                            ),
                          ),
                        ],
                      ),
                      // Notification Bell
                      Stack(
                        children: [
                          Container(
                            decoration: BoxDecoration(
                              color: Colors.white.withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(14),
                              border: Border.all(
                                color: Colors.white.withValues(alpha: 0.2),
                              ),
                            ),
                            child: IconButton(
                              icon: const Icon(
                                Icons.notifications_outlined,
                                color: Colors.white,
                                size: 26,
                              ),
                              onPressed: _showNotifications,
                              tooltip: 'Anxiety Alerts',
                            ),
                          ),
                          if (_hasUnread)
                            Positioned(
                              right: 6,
                              top: 6,
                              child: Container(
                                width: 10,
                                height: 10,
                                decoration: BoxDecoration(
                                  color: const Color(0xFFFF4444),
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: const Color(0xFF667eea),
                                    width: 1.5,
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),

                  // ── Connection Status ──
                  if (!_chestStrap.isConnected ||
                      !(_lastReading?.isWorn ?? false))
                    Container(
                      margin: const EdgeInsets.only(top: 12),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: Colors.white.withValues(alpha: 0.2),
                        ),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            !_chestStrap.isConnected
                                ? Icons.bluetooth_disabled_rounded
                                : Icons.warning_amber_rounded,
                            color: Colors.white.withValues(alpha: 0.8),
                            size: 16,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              !_chestStrap.isConnected
                                  ? 'Chest strap not connected. Your current readings are unavailable.'
                                  : 'Chest strap connected. Please wear it on your chest to start the readings.',
                              style: GoogleFonts.poppins(
                                fontSize: 11,
                                color: Colors.white.withValues(alpha: 0.9),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                  const SizedBox(height: 28),

                  // ── Meditation Hero Image ──
                  Center(
                    child: Container(
                      constraints: const BoxConstraints(maxWidth: 260),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(24),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.2),
                            blurRadius: 30,
                            offset: const Offset(0, 12),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(24),
                        child: Image.asset(
                          'assets/welcome_illustration.png',
                          fit: BoxFit.contain,
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 28),

                  // ── Overall Anxiety Score Card ──
                  AnimatedBuilder(
                    animation: _pulseController,
                    builder: (context, child) {
                      return Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(22),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.2),
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: riskCol.withValues(
                                alpha: 0.15 * _pulseAnimation.value,
                              ),
                              blurRadius: 20,
                              offset: const Offset(0, 8),
                            ),
                          ],
                        ),
                        child: Column(
                          children: [
                            // Risk indicator row
                            Row(
                              children: [
                                Container(
                                  width: 56,
                                  height: 56,
                                  decoration: BoxDecoration(
                                    color: riskCol.withValues(alpha: 0.2),
                                    borderRadius: BorderRadius.circular(18),
                                    border: Border.all(
                                      color: riskCol.withValues(alpha: 0.4),
                                      width: 2,
                                    ),
                                  ),
                                  child: Icon(
                                    _overallIcon(risk),
                                    color: Colors.white,
                                    size: 30,
                                  ),
                                ),
                                const SizedBox(width: 16),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        'Overall Anxiety Level',
                                        style: GoogleFonts.poppins(
                                          fontSize: 12,
                                          color: Colors.white.withValues(
                                            alpha: 0.7,
                                          ),
                                        ),
                                      ),
                                      const SizedBox(height: 2),
                                      Text(
                                        label,
                                        style: GoogleFonts.poppins(
                                          fontSize: 22,
                                          fontWeight: FontWeight.w700,
                                          color: Colors.white,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                // Score circle
                                Container(
                                  width: 54,
                                  height: 54,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: Colors.white.withValues(alpha: 0.15),
                                    border: Border.all(
                                      color: Colors.white.withValues(
                                        alpha: 0.4,
                                      ),
                                      width: 2,
                                    ),
                                  ),
                                  child: Center(
                                    child: Text(
                                      _hasOverallRisk
                                          ? risk.toStringAsFixed(0)
                                          : '--',
                                      style: GoogleFonts.poppins(
                                        fontSize: 18,
                                        fontWeight: FontWeight.w800,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 16),

                            // Risk progress bar
                            ClipRRect(
                              borderRadius: BorderRadius.circular(6),
                              child: Stack(
                                children: [
                                  Container(
                                    height: 6,
                                    decoration: BoxDecoration(
                                      color: Colors.white.withValues(
                                        alpha: 0.15,
                                      ),
                                      borderRadius: BorderRadius.circular(6),
                                    ),
                                  ),
                                  AnimatedFractionallySizedBox(
                                    duration: const Duration(milliseconds: 600),
                                    widthFactor: _hasOverallRisk
                                        ? (risk / 100)
                                              .clamp(0.02, 1.0)
                                              .toDouble()
                                        : 0.0,
                                    child: Container(
                                      height: 6,
                                      decoration: BoxDecoration(
                                        gradient: LinearGradient(
                                          colors: [
                                            Colors.white.withValues(alpha: 0.9),
                                            riskCol.withValues(alpha: 0.8),
                                          ],
                                        ),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 8),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                _riskLabel(
                                  'Low',
                                  _hasOverallRisk && risk <= 20,
                                ),
                                _riskLabel(
                                  'Moderate',
                                  _hasOverallRisk && risk > 20 && risk <= 45,
                                ),
                                _riskLabel(
                                  'Elevated',
                                  _hasOverallRisk && risk > 45 && risk <= 70,
                                ),
                                _riskLabel(
                                  'High',
                                  _hasOverallRisk && risk > 70,
                                ),
                              ],
                            ),
                            const SizedBox(height: 16),

                            // Message
                            Container(
                              width: double.infinity,
                              padding: const EdgeInsets.all(14),
                              decoration: BoxDecoration(
                                color: Colors.white.withValues(alpha: 0.1),
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(
                                  color: Colors.white.withValues(alpha: 0.15),
                                ),
                              ),
                              child: Text(
                                _overallMessage(risk),
                                style: GoogleFonts.poppins(
                                  fontSize: 13,
                                  color: Colors.white.withValues(alpha: 0.9),
                                  height: 1.5,
                                ),
                                textAlign: TextAlign.center,
                              ),
                            ),

                            const SizedBox(height: 12),
                          ],
                        ),
                      );
                    },
                  ),

                  const SizedBox(height: 24),

                  _buildWeeklyInsightsCard(),

                  const SizedBox(height: 24),

                  // ── Quick Tips Card ──
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 14,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.15),
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.lock_outline_rounded,
                          color: Colors.white.withValues(alpha: 0.7),
                          size: 18,
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            'Your data is protected and stored without your name for research only.',
                            style: GoogleFonts.poppins(
                              fontSize: 11,
                              color: Colors.white.withValues(alpha: 0.7),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ── Helper widgets ─────────────────────────────────────────────

  Widget _riskLabel(String text, bool active) {
    return Text(
      text,
      style: GoogleFonts.poppins(
        fontSize: 10,
        color: Colors.white.withValues(alpha: active ? 1.0 : 0.5),
        fontWeight: active ? FontWeight.w700 : FontWeight.w400,
      ),
    );
  }

  Widget _buildWeeklyInsightsCard() {
    final summary = _weeklySummary;
    final alerts = (summary?['alerts'] as num?)?.toInt() ?? 0;
    final answered = (summary?['answered_alerts'] as num?)?.toInt() ?? 0;
    final commonActivity = summary?['common_activity'] as String?;
    final effectiveAction = summary?['most_effective_action'] as String?;
    final confirmationRate = (summary?['confirmation_rate'] as num?)
        ?.toDouble();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.13),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.insights_rounded, color: Colors.white),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Your week at a glance',
                  style: GoogleFonts.poppins(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              IconButton(
                onPressed: _weeklyLoading ? null : _loadWeeklySummary,
                icon: _weeklyLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.refresh_rounded, color: Colors.white),
                tooltip: 'Refresh weekly summary',
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            _weeklyError != null
                ? _weeklyError!
                : summary == null
                ? 'Loading your weekly summary...'
                : alerts == 0
                ? 'No anxiety alerts were recorded in the last 7 days.'
                : '$alerts alert${alerts == 1 ? '' : 's'}, $answered answered'
                      '${confirmationRate == null ? '' : ' · ${(confirmationRate * 100).round()}% felt anxious'}',
            style: GoogleFonts.poppins(
              color: Colors.white.withValues(alpha: 0.88),
              fontSize: 12,
              height: 1.45,
            ),
          ),
          if (commonActivity != null) ...[
            const SizedBox(height: 6),
            Text(
              'Most common situation: $commonActivity',
              style: GoogleFonts.poppins(
                color: Colors.white.withValues(alpha: 0.82),
                fontSize: 11,
              ),
            ),
          ],
          if (effectiveAction != null) ...[
            const SizedBox(height: 4),
            Text(
              'You often felt better after: $effectiveAction',
              style: GoogleFonts.poppins(
                color: Colors.white.withValues(alpha: 0.82),
                fontSize: 11,
              ),
            ),
          ],
          const SizedBox(height: 8),
          Text(
            'These are patterns from your check-ins, not medical advice.',
            style: GoogleFonts.poppins(
              color: Colors.white.withValues(alpha: 0.62),
              fontSize: 10,
            ),
          ),
          if (_weeklyError != null) ...[
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: _weeklyLoading ? null : _loadWeeklySummary,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Try again'),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.white,
                side: const BorderSide(color: Colors.white70),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ═════════════════════════════════════════════════════════════════════
// Notifications Sheet
// ═════════════════════════════════════════════════════════════════════

class _NotificationsSheet extends StatelessWidget {
  final List<String> notifications;
  final VoidCallback onClear;
  const _NotificationsSheet({
    required this.notifications,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.6,
      ),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(height: 12),
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 16),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: const Color(0xFF667eea).withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.notifications_rounded,
                    color: Color(0xFF667eea),
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  'Anxiety Alerts',
                  style: GoogleFonts.poppins(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
                const Spacer(),
                if (notifications.isNotEmpty)
                  TextButton.icon(
                    onPressed: () {
                      onClear();
                      Navigator.of(context).pop();
                    },
                    icon: const Icon(Icons.delete_sweep_outlined, size: 18),
                    label: const Text('Clear all'),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Flexible(
            child: notifications.isEmpty
                ? Padding(
                    padding: const EdgeInsets.all(40),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.notifications_none_rounded,
                          size: 48,
                          color: Colors.grey.shade300,
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'No alerts yet',
                          style: GoogleFonts.poppins(
                            fontSize: 14,
                            color: Colors.grey.shade500,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Gentle check-ins based on your recent readings will appear here.',
                          textAlign: TextAlign.center,
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: Colors.grey.shade400,
                          ),
                        ),
                      ],
                    ),
                  )
                : ListView.separated(
                    shrinkWrap: true,
                    padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                    itemCount: notifications.length,
                    separatorBuilder: (_, _) => const SizedBox(height: 8),
                    itemBuilder: (_, i) {
                      return Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: Theme.of(
                            context,
                          ).colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: Theme.of(
                              context,
                            ).colorScheme.tertiary.withValues(alpha: 0.5),
                          ),
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(
                              Icons.warning_amber_rounded,
                              color: Theme.of(context).colorScheme.tertiary,
                              size: 20,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                notifications[i],
                                style: GoogleFonts.poppins(
                                  fontSize: 12,
                                  color: Theme.of(
                                    context,
                                  ).colorScheme.onSurface,
                                  height: 1.4,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
