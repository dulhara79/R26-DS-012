// ─────────────────────────────────────────────────────────────────────────────
//  Component 2 — Sensing & data details
//
//  Opened from the Activity tab ("Behavioural Context"). The Activity tab owns
//  the participant-facing summary (baseline, observations, change detection);
//  this page only shows how collection is working:
//   1. Raw weekly passive metrics (home/away time, places, sleep and movement
//      proxies), never compared with anyone else
//   2. Data-quality coverage for the last 14 completed days
//   3. Collection status with a one-tap fix when collection has stopped
//   4. Today's on-device measurements
//   5. A plain-text summary the participant can choose to give a clinician
//   6. A static, always-visible crisis-resource banner
//
//  Nothing here is simulated or scored. Every value is either a real pipeline
//  output or an explicit "not available" state.
// ─────────────────────────────────────────────────────────────────────────────

import 'dart:async';
import 'dart:convert';
import 'package:app_settings/app_settings.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:battery_plus/battery_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:call_log/call_log.dart';
import 'package:usage_stats/usage_stats.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/background/background_service.dart' as bg;
import '../services/background_service_helper.dart';
import 'clinician_summary_page.dart';
import '../services/research_permission_service.dart';
import '../theme/c2_palette.dart';
import '../widgets/c2/crisis_banner.dart';

// Colour tokens are shared with the Activity tab so both pages match.
typedef _C = C2Palette;

// ─────────────────────────────────────────────
// NEW MODEL (1) — Passive metrics already computed by RAPIDS, shown raw,
// no scoring, independent of whether the 28-day baseline is ready.
// ─────────────────────────────────────────────
class PassiveMetrics {
  final double? homeHours; // hours spent at inferred "home" cluster
  final double? awayHours; // hours spent away from home
  final int? significantPlaces; // count of distinct location clusters visited
  final String?
  sleepProxyWindow; // e.g. "11:42 PM \u2013 7:05 AM" (screen-off span)
  final double? overnightScreenOffHours;
  final double?
  activityProxyScore; // raw accelerometer-derived movement index, unitless
  final bool activityDataAvailable;

  const PassiveMetrics({
    this.homeHours,
    this.awayHours,
    this.significantPlaces,
    this.sleepProxyWindow,
    this.overnightScreenOffHours,
    this.activityProxyScore,
    this.activityDataAvailable = false,
  });

  factory PassiveMetrics.fromJson(Map<String, dynamic>? j) {
    if (j == null) return const PassiveMetrics();
    return PassiveMetrics(
      homeHours: (j['home_hours'] as num?)?.toDouble(),
      awayHours: (j['away_hours'] as num?)?.toDouble(),
      significantPlaces: (j['significant_places'] as num?)?.toInt(),
      sleepProxyWindow: j['sleep_proxy_window'] as String?,
      overnightScreenOffHours: (j['overnight_screen_off_hours'] as num?)
          ?.toDouble(),
      activityProxyScore: (j['activity_proxy_score'] as num?)?.toDouble(),
      activityDataAvailable: j['activity_data_available'] as bool? ?? false,
    );
  }
}

// ─────────────────────────────────────────────
// NEW MODEL (2) — Data-quality / coverage, purely descriptive.
// Your ablation found missingness carries no signal (AUROC 0.5172, chance
// level), so this is framed strictly as a trust indicator, never as
// something that means anything about the person.
// ─────────────────────────────────────────────
class DayCoverage {
  final DateTime date;
  final bool usable; // true if the day met minimum sensor coverage thresholds

  const DayCoverage({required this.date, required this.usable});

  factory DayCoverage.fromJson(Map<String, dynamic> j) => DayCoverage(
    date: DateTime.tryParse(j['date'] as String? ?? '') ?? DateTime.now(),
    usable: j['usable'] as bool? ?? false,
  );
}

// ─────────────────────────────────────────────
// PAGE
// ─────────────────────────────────────────────
class DigitalPhenotypingPage extends StatefulWidget {
  final String? userId;
  const DigitalPhenotypingPage({super.key, this.userId});

  @override
  State<DigitalPhenotypingPage> createState() => _DigitalPhenotypingPageState();
}

class _DigitalPhenotypingPageState extends State<DigitalPhenotypingPage> {
  bool _loading = true;

  // Real device measurements (today)
  int _callCount = 0;
  int _smsCount = 0;
  double _screenHours = 0.0;
  String _locationStatus = 'Checking\u2026';
  double? _locationAccuracy;
  String _batteryStatus = 'Checking\u2026';
  int _queueSize = 0;
  bool _serviceRunning = false;
  int _daysEnrolled = 0;

  PassiveMetrics _passive = const PassiveMetrics();
  List<DayCoverage> _coverage = [];

  bool _fixing = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    await Future.wait([_fetchDeviceMetrics(), _fetchServiceStatus()]);
    await Future.wait([_fetchPassiveMetrics(), _fetchCoverage()]);
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _fetchServiceStatus() async {
    try {
      _queueSize = await BackgroundServiceHelper.getOfflineQueueSize();
      _serviceRunning = await BackgroundServiceHelper.isServiceRunning();
      final prefs = await SharedPreferences.getInstance();
      final enrolled = prefs.getString('enrolled_date');
      if (enrolled != null) {
        final d = DateTime.tryParse(enrolled);
        if (d != null) _daysEnrolled = DateTime.now().difference(d).inDays;
      }
    } catch (e) {
      debugPrint('Service status error: $e');
    }
  }

  /// (1) Passive metrics — raw numbers only, no scoring, shown regardless of
  /// baseline status since they don't rely on a 28-day comparison window.
  Future<void> _fetchPassiveMetrics() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final cached = prefs.getString('c2_passive_metrics');
      if (cached != null && cached.isNotEmpty) {
        _passive = PassiveMetrics.fromJson(
          jsonDecode(cached) as Map<String, dynamic>,
        );
      }
    } catch (e) {
      debugPrint('Passive metrics parse error: $e');
    }
  }

  /// (2) Per-day usable-data coverage for the last 14 days.
  Future<void> _fetchCoverage() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final cached = prefs.getString('c2_day_coverage');
      if (cached != null && cached.isNotEmpty) {
        final list = jsonDecode(cached) as List;
        _coverage = list
            .map((e) => DayCoverage.fromJson(e as Map<String, dynamic>))
            .toList();
      }
    } catch (e) {
      debugPrint('Coverage parse error: $e');
    }
    // Fallback: build a 14-day scaffold marked "no data" so the panel never
    // silently shows nothing.
    if (_coverage.isEmpty) {
      final today = DateTime.now();
      _coverage = List.generate(14, (i) {
        final d = today.subtract(Duration(days: 13 - i));
        return DayCoverage(
          date: DateTime(d.year, d.month, d.day),
          usable: false,
        );
      });
    }
  }

  Future<void> _fetchDeviceMetrics() async {
    try {
      final now = DateTime.now().millisecondsSinceEpoch;
      final entries = await CallLog.query(
        dateFrom: now - 86400000,
      ).timeout(const Duration(seconds: 5), onTimeout: () => []);
      _callCount = entries.length;

      final smsQuery = SmsQuery();
      final inbox = await smsQuery
          .querySms(kinds: [SmsQueryKind.inbox])
          .timeout(const Duration(seconds: 5), onTimeout: () => []);
      final sent = await smsQuery
          .querySms(kinds: [SmsQueryKind.sent])
          .timeout(const Duration(seconds: 5), onTimeout: () => []);
      bool isToday(DateTime? d) {
        if (d == null) return false;
        final n = DateTime.now();
        return d.year == n.year && d.month == n.month && d.day == n.day;
      }

      _smsCount =
          inbox.where((m) => isToday(m.date)).length +
          sent.where((m) => isToday(m.date)).length;

      final end = DateTime.now();
      final start = DateTime(end.year, end.month, end.day);
      final usage = await UsageStats.queryUsageStats(
        start,
        end,
      ).timeout(const Duration(seconds: 5), onTimeout: () => []);
      double secs = 0;
      for (final u in usage) {
        secs += (int.tryParse(u.totalTimeInForeground ?? '0') ?? 0) / 1000;
      }
      _screenHours = secs / 3600;

      final battery = Battery();
      final level = await battery.batteryLevel.timeout(
        const Duration(seconds: 3),
        onTimeout: () => 0,
      );
      final state = await battery.batteryState.timeout(
        const Duration(seconds: 3),
        onTimeout: () => BatteryState.unknown,
      );
      _batteryStatus = '$level% \u00b7 ${state.name}';

      try {
        // NOTE: high accuracy, matching background_service.dart. The pipeline
        // requires <100 m fixes; LocationAccuracy.low returns 100-300 m.
        final pos = await Geolocator.getCurrentPosition(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.high,
          ),
        ).timeout(const Duration(seconds: 8));
        _locationAccuracy = pos.accuracy;
        _locationStatus = pos.accuracy <= 100
            ? 'Active \u00b7 \u00b1${pos.accuracy.toStringAsFixed(0)} m'
            : 'Low precision \u00b7 \u00b1${pos.accuracy.toStringAsFixed(0)} m';
      } catch (_) {
        _locationStatus = 'Unavailable \u2014 check permissions';
      }
    } catch (e) {
      debugPrint('Device metrics error: $e');
    }
  }

  // ─── HELPERS ─────────────────────────────────

  // ─────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _C.scaffold,
      appBar: AppBar(
        backgroundColor: _C.scaffold,
        elevation: 0,
        leading: Navigator.canPop(context)
            ? IconButton(
                icon: Icon(
                  Icons.arrow_back_ios_new_rounded,
                  color: _C.textPrimary,
                  size: 20,
                ),
                onPressed: () => Navigator.pop(context),
              )
            : null,
        title: Text(
          'Sensing & data details',
          style: GoogleFonts.poppins(
            color: _C.textPrimary,
            fontWeight: FontWeight.w600,
            fontSize: 17,
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.ios_share_rounded, color: _C.textMuted, size: 19),
            tooltip: 'Prepare for my appointment',
            onPressed: _exportForClinician,
          ),
          IconButton(
            icon: Icon(Icons.refresh_rounded, color: _C.textMuted, size: 20),
            onPressed: () {
              setState(() => _loading = true);
              _load();
            },
          ),
        ],
      ),
      body: SafeArea(
        child: _loading
            ? Center(child: CircularProgressIndicator(color: _C.primary))
            : RefreshIndicator(
                onRefresh: _load,
                color: _C.primary,
                child: SingleChildScrollView(
                  physics: const AlwaysScrollableScrollPhysics(
                    parent: BouncingScrollPhysics(),
                  ),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 8,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Static crisis-resource banner — always visible,
                      // independent of any model output.
                      const C2CrisisBanner(),
                      const SizedBox(height: 16),
                      _header(),
                      const SizedBox(height: 18),
                      _sectionTitle('Your week'),
                      const SizedBox(height: 4),
                      Text(
                        'Raw numbers, not compared with anyone else',
                        style: GoogleFonts.poppins(
                          fontSize: 11.5,
                          color: _C.textMuted,
                        ),
                      ),
                      const SizedBox(height: 10),
                      _passiveMetricsCard(),
                      const SizedBox(height: 18),
                      _sectionTitle('Data quality'),
                      const SizedBox(height: 10),
                      _dataQualityCard(),
                      const SizedBox(height: 18),
                      _sectionTitle('Collection status'),
                      const SizedBox(height: 10),
                      _collectionStatusCard(),
                      const SizedBox(height: 18),
                      _sectionTitle('Today\u2019s measurements'),
                      const SizedBox(height: 10),
                      _metric(
                        Icons.location_on_rounded,
                        'Location',
                        _locationStatus,
                        'GPS fix every 15 minutes',
                        warn:
                            _locationAccuracy != null &&
                            _locationAccuracy! > 100,
                      ),
                      const SizedBox(height: 10),
                      _metric(
                        Icons.screen_lock_portrait_rounded,
                        'Screen time',
                        '${_screenHours.toStringAsFixed(1)} hrs',
                        'Foreground app usage since midnight',
                      ),
                      const SizedBox(height: 10),
                      _metric(
                        Icons.record_voice_over_rounded,
                        'Communication',
                        '$_callCount calls \u00b7 $_smsCount SMS',
                        'Counts only \u2014 no content is collected',
                      ),
                      const SizedBox(height: 10),
                      _metric(
                        Icons.battery_charging_full_rounded,
                        'Battery',
                        _batteryStatus,
                        'Affects collection reliability',
                      ),
                      const SizedBox(height: 18),
                      _clinicianExportCard(),
                      const SizedBox(height: 18),
                      _disclaimerCard(),
                      const SizedBox(height: 30),
                    ],
                  ),
                ),
              ),
      ),
    );
  }

  // ─── HEADER ──────────────────────────────────

  Widget _header() => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        'How your data is collected',
        style: GoogleFonts.poppins(
          fontSize: 23,
          fontWeight: FontWeight.w700,
          color: _C.textPrimary,
          letterSpacing: -0.5,
        ),
      ),
      const SizedBox(height: 3),
      Text(
        'Your pattern comparisons are on the Activity tab. This page shows '
        'the raw numbers behind them and whether collection is working.',
        style: GoogleFonts.poppins(
          fontSize: 12.5,
          color: _C.textMuted,
          height: 1.45,
        ),
      ),
    ],
  );

  Widget _sectionTitle(String t) => Text(
    t,
    style: GoogleFonts.poppins(
      fontSize: 16,
      fontWeight: FontWeight.w700,
      color: _C.textPrimary,
      letterSpacing: -0.3,
    ),
  );

  // ─── (1) PASSIVE METRICS CARD ────────────────
  // Same raw-numbers, no-scoring treatment as v1's "Today's Measurements",
  // just extended to the additional RAPIDS features. Shown even before the
  // baseline is ready, since these are not baseline-relative comparisons.

  Widget _passiveMetricsCard() {
    final p = _passive;
    return Column(
      children: [
        _metric(
          Icons.home_outlined,
          'Time at home vs. away',
          p.homeHours != null && p.awayHours != null
              ? '${p.homeHours!.toStringAsFixed(1)}h home \u00b7 ${p.awayHours!.toStringAsFixed(1)}h away'
              : 'Not available yet',
          'Average per day over your last 7 usable days',
        ),
        const SizedBox(height: 10),
        _metric(
          Icons.place_outlined,
          'Significant places visited',
          p.significantPlaces != null
              ? '${p.significantPlaces} places'
              : 'Not available yet',
          'Different places you spent time at, on average per day',
        ),
        const SizedBox(height: 10),
        _metric(
          Icons.bedtime_outlined,
          'Sleep proxy',
          p.sleepProxyWindow ??
              (p.overnightScreenOffHours != null
                  ? '${p.overnightScreenOffHours!.toStringAsFixed(1)}h screen-off'
                  : 'Not available yet'),
          'Estimated from overnight screen activity, not a sleep sensor',
        ),
        const SizedBox(height: 10),
        _metric(
          Icons.directions_walk_rounded,
          'Movement proxy',
          p.activityDataAvailable && p.activityProxyScore != null
              ? '${(p.activityProxyScore! * 100).toStringAsFixed(1)}% high-motion readings'
              : 'No movement data yet',
          'From the phone\u2019s motion sensor \u2014 not a step or exercise count',
        ),
      ],
    );
  }

  // ─── (2) DATA QUALITY CARD ────────────────────
  // Framed strictly as a trust/quality indicator. The missingness ablation
  // found no signal here (0.5172, chance level) — this panel exists so
  // participants can see coverage, not so it implies anything.

  Widget _dataQualityCard() {
    final usableDays = _coverage.where((d) => d.usable).length;
    final totalDays = _coverage.length;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _C.cardBase,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _C.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'You had usable data on $usableDays of the last $totalDays days',
            style: GoogleFonts.poppins(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: _C.textPrimary,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: _coverage
                .map(
                  (d) => Expanded(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 2),
                      child: Container(
                        height: 22,
                        decoration: BoxDecoration(
                          color: d.usable ? _C.teal : _C.p100,
                          borderRadius: BorderRadius.circular(5),
                        ),
                      ),
                    ),
                  ),
                )
                .toList(),
          ),
          const SizedBox(height: 10),
          Text(
            'This shows how much sensing data reached Aura. Missing days '
            'happen (phone off, battery saver, permissions) and say nothing '
            'about your wellbeing.',
            style: GoogleFonts.poppins(
              fontSize: 11,
              color: _C.textMuted,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  // ─── COLLECTION STATUS ───────────────────────

  Widget _collectionStatusCard() {
    final ok = _serviceRunning;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _C.cardBase,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _C.border),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  color: ok ? _C.teal : _C.rose,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 10),
              Text(
                ok ? 'Collection active' : 'Collection stopped',
                style: GoogleFonts.poppins(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: _C.textPrimary,
                ),
              ),
              const Spacer(),
              Text(
                '$_daysEnrolled days enrolled',
                style: GoogleFonts.poppins(fontSize: 11, color: _C.textMuted),
              ),
            ],
          ),
          if (!ok) ...[
            const SizedBox(height: 10),
            Text(
              'Aura cannot collect new data right now. This usually means a '
              'permission was turned off or battery saver stopped the app.',
              style: GoogleFonts.poppins(
                fontSize: 11.5,
                color: _C.textSecondary,
                height: 1.45,
              ),
            ),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _fixing ? null : _fixCollection,
                icon: _fixing
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.build_circle_outlined, size: 18),
                label: Text(
                  _fixing ? 'Checking\u2026' : 'Fix collection',
                  style: GoogleFonts.poppins(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                style: FilledButton.styleFrom(
                  backgroundColor: _C.primary,
                  foregroundColor: _C.cardBase,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
          ],
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _statTile(
                  'Pending upload',
                  '$_queueSize',
                  warn: _queueSize > 500,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _statTile(
                  'Usable days (last 14)',
                  '${_coverage.where((d) => d.usable).length}/${_coverage.length}',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  /// Re-runs permission onboarding and restarts the collector. If it still
  /// cannot start, opens the app's system settings so the participant can
  /// re-enable a permission or turn off battery optimisation.
  Future<void> _fixCollection() async {
    if (kIsWeb) return;
    setState(() => _fixing = true);
    var running = false;
    try {
      await ResearchPermissionService.requestMissingPermissions(force: true);
      running = await bg.startBackgroundServiceIfPermitted();
    } catch (e) {
      debugPrint('Fix collection error: $e');
    }
    await _fetchServiceStatus();
    if (!mounted) return;
    setState(() => _fixing = false);
    if (running || _serviceRunning) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Collection is running again.')),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Opening settings. Allow location and turn off battery '
            'optimisation for Aura.',
          ),
        ),
      );
      await AppSettings.openAppSettings();
    }
  }

  Widget _statTile(String label, String value, {bool warn = false}) =>
      Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
        decoration: BoxDecoration(
          color: warn ? _C.amberBg : _C.p100,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              value,
              style: GoogleFonts.poppins(
                fontSize: 17,
                fontWeight: FontWeight.w700,
                color: warn ? _C.amber : _C.p500,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: GoogleFonts.poppins(fontSize: 11, color: _C.textMuted),
            ),
          ],
        ),
      );

  // ─── METRIC ROW ──────────────────────────────

  Widget _metric(
    IconData icon,
    String title,
    String value,
    String subtitle, {
    bool warn = false,
  }) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.cardBase,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(
        color: warn ? _C.amber.withValues(alpha: 0.5) : _C.border,
      ),
    ),
    child: Row(
      children: [
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: warn ? _C.amberBg : _C.chip,
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: warn ? _C.amber : _C.primary, size: 20),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: GoogleFonts.poppins(fontSize: 12, color: _C.textMuted),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: _C.textPrimary,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                subtitle,
                style: GoogleFonts.poppins(fontSize: 11, color: _C.textMuted),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  // ─── CLINICIAN EXPORT ─────────────────────

  Widget _clinicianExportCard() => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: _C.cardBase,
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: _C.border),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.description_outlined, size: 18, color: _C.primary),
            const SizedBox(width: 8),
            Text(
              'Prepare for my appointment',
              style: GoogleFonts.poppins(
                fontSize: 13.5,
                fontWeight: FontWeight.w700,
                color: _C.textPrimary,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          'Preview a one-page summary of your check-ins, what helped and '
          'your behavioural patterns, choose what to include, then share it '
          'as a PDF. Nothing is sent automatically.',
          style: GoogleFonts.poppins(
            fontSize: 11.5,
            color: _C.textSecondary,
            height: 1.45,
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: _exportForClinician,
            icon: const Icon(Icons.ios_share_rounded, size: 16),
            label: Text(
              'Open summary',
              style: GoogleFonts.poppins(
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
              ),
            ),
            style: OutlinedButton.styleFrom(
              foregroundColor: _C.primary,
              side: BorderSide(color: _C.p200),
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ),
      ],
    ),
  );

  /// Opens the participant-controlled summary screen, where the participant
  /// previews, chooses sections and shares a PDF. Nothing is sent
  /// automatically.
  void _exportForClinician() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ClinicianSummaryPage(userId: widget.userId),
      ),
    );
  }

  // ─── DISCLAIMER ──────────────────────────────

  Widget _disclaimerCard() => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.p100,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: _C.p200),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.shield_outlined, size: 17, color: _C.p500),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            'These are descriptive observations of your own behaviour over '
            'time. They are not a diagnosis, a risk score, or a prediction. '
            'Discuss any concerns with your clinician.',
            style: GoogleFonts.poppins(
              fontSize: 11,
              color: _C.textSecondary,
              height: 1.5,
            ),
          ),
        ),
      ],
    ),
  );
}
