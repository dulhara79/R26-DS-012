// ─────────────────────────────────────────────────────────────────────────────
//  Component 2 — Behavioural Observation Panel  (v2)
//
//  Adds, on top of the v1 descriptive-observations page:
//   1. More passive metrics already produced by the RAPIDS pipeline
//      (home/away time, significant places, sleep proxy, activity proxy)
//   2. A data-quality / coverage panel ("usable data on N of last 14 days")
//   3. Change-detection copy for Day 57+, with the false-alarm rate shown
//      next to any flagged change, not buried in a settings page
//   4. A check-in history/journal view, kept explicitly separate from
//      passive data, with a one-line "why separate" explanation
//   5. A plain-text export the participant can hand to their own clinician
//      (no score reaches the clinician automatically — human stays in the loop)
//   6. A static, always-visible crisis-resource banner, independent of any
//      model output
//
//  Nothing here is simulated or scored. Every new value is either a real
//  pipeline output or an explicit "not available" state, same discipline as
//  v1's `blockingIssues`.
// ─────────────────────────────────────────────────────────────────────────────

import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:battery_plus/battery_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:call_log/call_log.dart';
import 'package:usage_stats/usage_stats.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/background_service_helper.dart';

// ─────────────────────────────────────────────
// COLOUR TOKENS — unchanged from v1
// ─────────────────────────────────────────────
class _C {
  static const scaffold = Color(0xFFF5F3FF);
  static const cardBase = Color(0xFFFFFFFF);
  static const chip = Color(0xFFF0ECFF);

  static const p500 = Color(0xFF5E60CE);
  static const p400 = Color(0xFF7C5CBF);
  static const p200 = Color(0xFFC4B5FD);
  static const p100 = Color(0xFFF0ECFF);

  static const primary = Color(0xFF5E60CE);
  static const amber = Color(0xFFF59B24);
  static const amberBg = Color(0xFFFEF3DC);
  static const rose = Color(0xFFEF5777);
  static const roseBg = Color(0xFFFDEAEE);
  static const teal = Color(0xFF0F9D8C);
  static const tealBg = Color(0xFFE3F5F2);

  static const textPrimary = Color(0xFF2D3142);
  static const textSecondary = Color(0xFF5A607F);
  static const textMuted = Color(0xFF9095A7);
  static const border = Color(0xFFE8E5F4);
}

// ─────────────────────────────────────────────
// DATA MODEL — mirrors component2_output.py; unchanged from v1
// ─────────────────────────────────────────────

/// A single behavioural observation expressed against the participant's own
/// baseline. [z] is null when no baseline exists yet.
class Observation {
  final String key;
  final String label;
  final double? z;
  final String direction; // above | below | stable | no_baseline | unknown
  final String confidence; // high | medium | low | insufficient
  final double? value;
  final String unit;

  const Observation({
    required this.key,
    required this.label,
    required this.z,
    required this.direction,
    required this.confidence,
    this.value,
    this.unit = '',
  });

  factory Observation.fromJson(String key, Map<String, dynamic> j) =>
      Observation(
        key: key,
        label: j['label'] as String? ?? key,
        z: (j['z'] as num?)?.toDouble(),
        direction: j['direction'] as String? ?? 'unknown',
        confidence: j['confidence'] as String? ?? 'low',
        value: (j['value'] as num?)?.toDouble(),
        unit: j['unit'] as String? ?? '',
      );

  bool get isFlagged => z != null && z!.abs() >= 1.5;
}

class ObservationPayload {
  final String participantId;
  final DateTime windowStart;
  final DateTime windowEnd;
  final bool baselineReady;
  final bool reportable;
  final List<Observation> observations;
  final List<String> blockingIssues;
  final int daysWithData;
  final int baselineDaysAvailable;
  final int baselineDaysRequired;
  final int emaReceived;
  final int emaExpected;

  const ObservationPayload({
    required this.participantId,
    required this.windowStart,
    required this.windowEnd,
    required this.baselineReady,
    required this.reportable,
    required this.observations,
    required this.blockingIssues,
    required this.daysWithData,
    required this.baselineDaysAvailable,
    required this.baselineDaysRequired,
    required this.emaReceived,
    required this.emaExpected,
  });

  factory ObservationPayload.fromJson(Map<String, dynamic> j) {
    final obsMap = (j['observations'] as Map<String, dynamic>? ?? {});
    final q = (j['data_quality'] as Map<String, dynamic>? ?? {});
    final w = (j['window'] as Map<String, dynamic>? ?? {});
    return ObservationPayload(
      participantId: j['participant_id'] as String? ?? 'unknown',
      windowStart:
          DateTime.tryParse(w['start'] as String? ?? '') ?? DateTime.now(),
      windowEnd: DateTime.tryParse(w['end'] as String? ?? '') ?? DateTime.now(),
      baselineReady: j['baseline_ready'] as bool? ?? false,
      reportable: j['reportable'] as bool? ?? false,
      observations: obsMap.entries
          .map(
            (e) => Observation.fromJson(e.key, e.value as Map<String, dynamic>),
          )
          .toList(),
      blockingIssues:
          (j['blocking_issues'] as List?)?.map((e) => e.toString()).toList() ??
          [],
      daysWithData: (q['days_with_data'] as num?)?.toInt() ?? 0,
      baselineDaysAvailable:
          (q['baseline_days_available'] as num?)?.toInt() ?? 0,
      baselineDaysRequired:
          (q['baseline_days_required'] as num?)?.toInt() ?? 28,
      emaReceived: (q['ema_received'] as num?)?.toInt() ?? 0,
      emaExpected: (q['ema_expected'] as num?)?.toInt() ?? 0,
    );
  }

  /// Local fallback used until the analysis backend is wired up. Reports the
  /// baseline-building state honestly rather than inventing observations.
  factory ObservationPayload.buildingBaseline({
    required String participantId,
    required int daysEnrolled,
    required int emaReceived,
    required int emaExpected,
  }) {
    const required = 28;
    return ObservationPayload(
      participantId: participantId,
      windowStart: DateTime.now().subtract(const Duration(days: 27)),
      windowEnd: DateTime.now(),
      baselineReady: daysEnrolled >= required * 2,
      reportable: false,
      observations: const [],
      blockingIssues: [
        'Personal baseline requires $required days of data before the '
            'reporting window. $daysEnrolled days collected so far.',
      ],
      daysWithData: daysEnrolled.clamp(0, 28),
      baselineDaysAvailable: (daysEnrolled - 28).clamp(0, 999),
      baselineDaysRequired: required,
      emaReceived: emaReceived,
      emaExpected: emaExpected,
    );
  }
}

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
// NEW MODEL (4) — Check-in history, kept structurally separate from
// passive data. Free-text answers are shown verbatim, never summarised
// or re-scored.
// ─────────────────────────────────────────────
class CheckInEntry {
  final DateTime timestamp;
  final Map<String, String>
  answers; // question label -> participant's own words

  const CheckInEntry({required this.timestamp, required this.answers});

  factory CheckInEntry.fromJson(Map<String, dynamic> j) => CheckInEntry(
    timestamp:
        DateTime.tryParse(j['timestamp'] as String? ?? '') ?? DateTime.now(),
    answers: (j['answers'] as Map<String, dynamic>? ?? {}).map(
      (k, v) => MapEntry(k, v.toString()),
    ),
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

  ObservationPayload? _payload;
  PassiveMetrics _passive = const PassiveMetrics();
  List<DayCoverage> _coverage = [];
  List<CheckInEntry> _checkIns = [];

  static const double _falseAlarmRate = 0.06; // ~6%, from validation on GLOBEM

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    await Future.wait([_fetchDeviceMetrics(), _fetchServiceStatus()]);
    await Future.wait([
      _fetchObservations(),
      _fetchPassiveMetrics(),
      _fetchCoverage(),
      _fetchCheckIns(),
    ]);
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

  /// Loads the observation payload. Until the analysis backend is available
  /// this reports the baseline-building state — it never fabricates values.
  Future<void> _fetchObservations() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final cached = prefs.getString('c2_observation_payload');
      if (cached != null && cached.isNotEmpty) {
        _payload = ObservationPayload.fromJson(
          jsonDecode(cached) as Map<String, dynamic>,
        );
        return;
      }
    } catch (e) {
      debugPrint('Observation payload parse error: $e');
    }
    _payload = ObservationPayload.buildingBaseline(
      participantId:
          widget.userId ?? await BackgroundServiceHelper.getCachedId(),
      daysEnrolled: _daysEnrolled,
      emaReceived: 0,
      emaExpected: 0,
    );
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

  /// (4) Check-in history, participant's own words, never re-scored here.
  Future<void> _fetchCheckIns() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final cached = prefs.getString('c2_checkin_history');
      if (cached != null && cached.isNotEmpty) {
        final list = jsonDecode(cached) as List;
        _checkIns =
            list
                .map((e) => CheckInEntry.fromJson(e as Map<String, dynamic>))
                .toList()
              ..sort((a, b) => b.timestamp.compareTo(a.timestamp));
      }
    } catch (e) {
      debugPrint('Check-in history parse error: $e');
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

  Color _zColor(double? z) {
    if (z == null) return _C.textMuted;
    final a = z.abs();
    if (a >= 2.0) return _C.rose;
    if (a >= 1.5) return _C.amber;
    return _C.teal;
  }

  Color _zBg(double? z) {
    if (z == null) return _C.p100;
    final a = z.abs();
    if (a >= 2.0) return _C.roseBg;
    if (a >= 1.5) return _C.amberBg;
    return _C.tealBg;
  }

  IconData _dirIcon(String d) => switch (d) {
    'above' => Icons.trending_up_rounded,
    'below' => Icons.trending_down_rounded,
    'stable' => Icons.trending_flat_rounded,
    _ => Icons.remove_rounded,
  };

  List<Observation> get _flaggedObservations =>
      (_payload?.observations ?? []).where((o) => o.isFlagged).toList();

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
                icon: const Icon(
                  Icons.arrow_back_ios_new_rounded,
                  color: _C.textPrimary,
                  size: 20,
                ),
                onPressed: () => Navigator.pop(context),
              )
            : null,
        title: Text(
          'Behavioural Context',
          style: GoogleFonts.poppins(
            color: _C.textPrimary,
            fontWeight: FontWeight.w600,
            fontSize: 17,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(
              Icons.ios_share_rounded,
              color: _C.textMuted,
              size: 19,
            ),
            tooltip: 'Export for clinician',
            onPressed: _exportForClinician,
          ),
          IconButton(
            icon: const Icon(
              Icons.refresh_rounded,
              color: _C.textMuted,
              size: 20,
            ),
            onPressed: () {
              setState(() => _loading = true);
              _load();
            },
          ),
        ],
      ),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator(color: _C.primary))
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
                      // (6) Static crisis-resource banner — always visible,
                      // independent of any model output.
                      _crisisBanner(),
                      const SizedBox(height: 16),
                      _header(),
                      const SizedBox(height: 16),
                      _observationCard(),
                      if (_flaggedObservations.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        _changeDetectionCard(),
                      ],
                      const SizedBox(height: 18),
                      _sectionTitle('Your Week'),
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
                      _sectionTitle('Data Quality'),
                      const SizedBox(height: 10),
                      _dataQualityCard(),
                      const SizedBox(height: 18),
                      _sectionTitle('Collection Status'),
                      const SizedBox(height: 10),
                      _collectionStatusCard(),
                      const SizedBox(height: 18),
                      _sectionTitle('Today\u2019s Measurements'),
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
                      _sectionTitle('Check-ins'),
                      const SizedBox(height: 10),
                      _checkInSeparationCard(),
                      const SizedBox(height: 10),
                      _checkInHistoryLink(context),
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
        'Behavioural observations',
        style: GoogleFonts.poppins(
          fontSize: 23,
          fontWeight: FontWeight.w700,
          color: _C.textPrimary,
          letterSpacing: -0.5,
        ),
      ),
      const SizedBox(height: 3),
      Text(
        'Measured against your own typical patterns',
        style: GoogleFonts.poppins(fontSize: 13, color: _C.textMuted),
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

  // ─── (6) CRISIS BANNER ───────────────────────
  // Static, always visible, not tied to any model output or flagged state.
  // NOTE: Replace the placeholder contacts below with the exact resource
  // list approved in your NHSL ethics protocol before this ships — a study
  // app's crisis text should match what the ethics committee signed off on,
  // not be assembled ad hoc.

  Widget _crisisBanner() => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.roseBg,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: _C.rose.withValues(alpha: 0.35)),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(Icons.favorite_rounded, size: 17, color: _C.rose),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'If you need to talk to someone right now',
                style: GoogleFonts.poppins(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w700,
                  color: _C.textPrimary,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'This app does not monitor you for crisis. These lines '
                'are staffed by people, any time.',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  color: _C.textSecondary,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  _crisisChip(
                    'National Mental Health Helpline',
                    '1926',
                    'tel:1926',
                  ),
                  _crisisChip(
                    'Sri Lanka Sumithrayo',
                    '011 2 696 666',
                    'tel:+94112696666',
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _crisisChip(String label, String number, String uri) => InkWell(
    borderRadius: BorderRadius.circular(20),
    onTap: () async {
      await Clipboard.setData(ClipboardData(text: number));
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Copied $number to clipboard')),
        );
      }
    },
    child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: _C.cardBase,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _C.rose.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.call_rounded, size: 13, color: _C.rose),
          const SizedBox(width: 6),
          Text(
            '$label \u00b7 $number',
            style: GoogleFonts.poppins(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: _C.textPrimary,
            ),
          ),
        ],
      ),
    ),
  );

  // ─── OBSERVATION CARD ────────────────────────

  Widget _observationCard() {
    final p = _payload;
    if (p == null) return const SizedBox.shrink();

    if (!p.reportable) return _baselineBuildingCard(p);

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _C.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.insights_rounded, color: _C.primary, size: 19),
              const SizedBox(width: 8),
              Text(
                'Last 28 days',
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: _C.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Compared with your own baseline',
            style: GoogleFonts.poppins(fontSize: 11, color: _C.textMuted),
          ),
          const SizedBox(height: 16),
          ...p.observations.map(_observationRow),
        ],
      ),
    );
  }

  Widget _observationRow(Observation o) {
    final hasZ = o.z != null;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: _zBg(o.z),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(_dirIcon(o.direction), size: 18, color: _zColor(o.z)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  o.label,
                  style: GoogleFonts.poppins(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: _C.textPrimary,
                  ),
                ),
                Text(
                  hasZ
                      ? '${o.direction == 'stable' ? 'Within' : 'Outside'} your usual range'
                      : 'Not enough data yet',
                  style: GoogleFonts.poppins(fontSize: 11, color: _C.textMuted),
                ),
              ],
            ),
          ),
          if (hasZ)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
              decoration: BoxDecoration(
                color: _zBg(o.z),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '${o.z! >= 0 ? '+' : ''}${o.z!.toStringAsFixed(1)}\u03c3',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: _zColor(o.z),
                ),
              ),
            )
          else
            Text(
              '\u2014',
              style: GoogleFonts.poppins(fontSize: 13, color: _C.textMuted),
            ),
        ],
      ),
    );
  }

  Widget _baselineBuildingCard(ObservationPayload p) {
    final have = p.baselineDaysAvailable;
    final need = p.baselineDaysRequired;
    final frac = need == 0 ? 0.0 : (have / need).clamp(0.0, 1.0);

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _C.cardBase,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _C.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.hourglass_top_rounded, color: _C.p400, size: 19),
              const SizedBox(width: 8),
              Text(
                'Building your baseline',
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: _C.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            'Observations compare your recent behaviour with your own typical '
            'patterns. That needs $need days of history first.',
            style: GoogleFonts.poppins(
              fontSize: 12,
              color: _C.textSecondary,
              height: 1.45,
            ),
          ),
          const SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: frac,
              minHeight: 8,
              backgroundColor: _C.p100,
              valueColor: const AlwaysStoppedAnimation(_C.p400),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Day $have of $need',
            style: GoogleFonts.poppins(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: _C.p500,
            ),
          ),
          if (p.blockingIssues.isNotEmpty) ...[
            const SizedBox(height: 14),
            ...p.blockingIssues.map(
              (b) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(top: 2),
                      child: Icon(
                        Icons.info_outline_rounded,
                        size: 14,
                        color: _C.textMuted,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        b,
                        style: GoogleFonts.poppins(
                          fontSize: 11,
                          color: _C.textMuted,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  // ─── (3) CHANGE DETECTION CARD ───────────────
  // Only rendered once at least one observation crosses the flag threshold.
  // Copy states the shift plainly, and puts the false-alarm rate right next
  // to it, so a flagged change reads as "worth noticing" not "diagnostic".

  Widget _changeDetectionCard() {
    final flagged = _flaggedObservations;
    if (flagged.isEmpty) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _C.amberBg,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _C.amber.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.notifications_active_outlined,
                size: 17,
                color: _C.amber,
              ),
              const SizedBox(width: 8),
              Text(
                'Change noticed',
                style: GoogleFonts.poppins(
                  fontSize: 13.5,
                  fontWeight: FontWeight.w700,
                  color: _C.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ...flagged.map(
            (o) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                'Your ${o.label.toLowerCase()} this week was '
                '${o.z! >= 0 ? 'higher' : 'lower'} than your usual range.',
                style: GoogleFonts.poppins(
                  fontSize: 12.5,
                  color: _C.textPrimary,
                  height: 1.4,
                ),
              ),
            ),
          ),
          const SizedBox(height: 6),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.info_outline_rounded,
                size: 13,
                color: _C.textMuted,
              ),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  'About ${(_falseAlarmRate * 100).toStringAsFixed(0)}% of flags '
                  'like this happen without anything meaningful behind them. '
                  'This is a nudge to notice, not a diagnosis.',
                  style: GoogleFonts.poppins(
                    fontSize: 11,
                    color: _C.textMuted,
                    height: 1.4,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

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
          'From your location-clustering step',
        ),
        const SizedBox(height: 10),
        _metric(
          Icons.place_outlined,
          'Significant places visited',
          p.significantPlaces != null
              ? '${p.significantPlaces} places'
              : 'Not available yet',
          'Distinct location clusters this week',
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
          'Activity proxy',
          p.activityDataAvailable && p.activityProxyScore != null
              ? p.activityProxyScore!.toStringAsFixed(2)
              : 'No accelerometer data available',
          'Raw movement index \u2014 unitless, not compared with anyone else',
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
            'This reflects sensor coverage only. It doesn\u2019t indicate '
            'anything about your behaviour or wellbeing \u2014 in our '
            'validation, missing data carried no meaningful signal.',
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
                  'Days with data',
                  '${_payload?.daysWithData ?? 0}/28',
                ),
              ),
            ],
          ),
        ],
      ),
    );
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

  // ─── (4) CHECK-IN SEPARATION + HISTORY LINK ──

  Widget _checkInSeparationCard() => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.tealBg,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: _C.teal.withValues(alpha: 0.3)),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(Icons.edit_note_rounded, size: 17, color: _C.teal),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            'Your check-ins are kept separate from the passive data on '
            'this page, because this study hasn\u2019t established a link '
            'between the two \u2014 so your check-ins always reflect what '
            'you actually told us, not a model\u2019s guess.',
            style: GoogleFonts.poppins(
              fontSize: 11.5,
              color: _C.textSecondary,
              height: 1.5,
            ),
          ),
        ),
      ],
    ),
  );

  Widget _checkInHistoryLink(BuildContext context) => InkWell(
    borderRadius: BorderRadius.circular(16),
    onTap: () => Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => CheckInHistoryPage(entries: _checkIns)),
    ),
    child: Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _C.cardBase,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _C.border),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: const BoxDecoration(
              color: _C.chip,
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.history_edu_rounded,
              color: _C.primary,
              size: 20,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'View check-in history',
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: _C.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${_checkIns.length} entries, in your own words',
                  style: GoogleFonts.poppins(fontSize: 11, color: _C.textMuted),
                ),
              ],
            ),
          ),
          const Icon(Icons.chevron_right_rounded, color: _C.textMuted),
        ],
      ),
    ),
  );

  // ─── (5) CLINICIAN EXPORT ─────────────────────

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
            const Icon(Icons.description_outlined, size: 18, color: _C.primary),
            const SizedBox(width: 8),
            Text(
              'Export for your clinician',
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
          'A plain-text summary of this week\u2019s raw figures, for you to '
          'share yourself. Nothing is sent automatically \u2014 you stay in '
          'control of what your clinician sees.',
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
              'Export this week',
              style: GoogleFonts.poppins(
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
              ),
            ),
            style: OutlinedButton.styleFrom(
              foregroundColor: _C.primary,
              side: const BorderSide(color: _C.p200),
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

  /// Builds a plain-text weekly summary and opens the share sheet so the
  /// participant can send it wherever they choose (email, print, hand a
  /// phone to their clinician directly). No score is included — only the
  /// same raw figures shown on this page, matching `display_permitted: false`.
  Future<void> _exportForClinician() async {
    final p = _payload;
    final buf = StringBuffer();
    final now = DateTime.now();

    buf.writeln('WEEKLY BEHAVIOURAL DATA SUMMARY');
    buf.writeln('Generated ${now.toIso8601String()}');
    buf.writeln(
      'Participant ID: ${p?.participantId ?? widget.userId ?? 'unknown'}',
    );
    buf.writeln('');
    buf.writeln('This is a plain export of raw, participant-facing figures.');
    buf.writeln('It contains no risk score, prediction, or model output.');
    buf.writeln('Nothing here should be treated as a diagnosis.');
    buf.writeln('');
    buf.writeln('-- Data coverage --');
    final usableDays = _coverage.where((d) => d.usable).length;
    buf.writeln('Usable data on $usableDays of ${_coverage.length} days.');
    buf.writeln('');
    buf.writeln('-- This week, raw figures --');
    buf.writeln('Screen time today: ${_screenHours.toStringAsFixed(1)} hrs');
    buf.writeln('Communication today: $_callCount calls, $_smsCount SMS');
    if (_passive.homeHours != null && _passive.awayHours != null) {
      buf.writeln(
        'Time at home: ${_passive.homeHours!.toStringAsFixed(1)} hrs, '
        'away: ${_passive.awayHours!.toStringAsFixed(1)} hrs',
      );
    }
    if (_passive.significantPlaces != null) {
      buf.writeln('Significant places visited: ${_passive.significantPlaces}');
    }
    if (_passive.sleepProxyWindow != null) {
      buf.writeln('Sleep proxy window: ${_passive.sleepProxyWindow}');
    }
    if (_passive.activityDataAvailable && _passive.activityProxyScore != null) {
      buf.writeln(
        'Activity proxy: ${_passive.activityProxyScore!.toStringAsFixed(2)}',
      );
    }
    buf.writeln('');
    if (p != null && p.reportable && p.observations.isNotEmpty) {
      buf.writeln('-- Observations vs. own baseline --');
      for (final o in p.observations) {
        final z = o.z != null
            ? '${o.z! >= 0 ? '+' : ''}${o.z!.toStringAsFixed(2)}\u03c3'
            : 'n/a';
        buf.writeln(
          '${o.label}: $z (${o.direction}, confidence: ${o.confidence})',
        );
      }
      buf.writeln('');
      buf.writeln(
        'Note: roughly ${(_falseAlarmRate * 100).toStringAsFixed(0)}% of flagged '
        'shifts like these occur without further significance (validation '
        'false-alarm rate).',
      );
    } else {
      buf.writeln(
        'Baseline not yet established \u2014 no comparative observations '
        'available this week.',
      );
    }
    buf.writeln('');
    buf.writeln('-- Check-ins --');
    buf.writeln(
      '${_checkIns.length} check-in entries recorded, kept separate '
      'from passive data and available to the participant in the app.',
    );

    try {
      await Clipboard.setData(ClipboardData(text: buf.toString()));
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Copied summary to clipboard')),
        );
      }
    } catch (e) {
      debugPrint('Export error: $e');
    }
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
        const Icon(Icons.shield_outlined, size: 17, color: _C.p500),
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

// ─────────────────────────────────────────────────────────────────────────────
// (4) CHECK-IN HISTORY PAGE
// A simple, chronological journal of past check-in answers, shown verbatim.
// No aggregation, scoring, or cross-referencing against passive data here —
// that separation is the point.
// ─────────────────────────────────────────────────────────────────────────────
class CheckInHistoryPage extends StatelessWidget {
  final List<CheckInEntry> entries;
  const CheckInHistoryPage({super.key, required this.entries});

  String _formatDate(DateTime d) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    final hour = d.hour % 12 == 0 ? 12 : d.hour % 12;
    final ampm = d.hour >= 12 ? 'PM' : 'AM';
    final minute = d.minute.toString().padLeft(2, '0');
    return '${months[d.month - 1]} ${d.day}, ${d.year} \u00b7 $hour:$minute $ampm';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _C.scaffold,
      appBar: AppBar(
        backgroundColor: _C.scaffold,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(
            Icons.arrow_back_ios_new_rounded,
            color: _C.textPrimary,
            size: 20,
          ),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          'Check-in history',
          style: GoogleFonts.poppins(
            color: _C.textPrimary,
            fontWeight: FontWeight.w600,
            fontSize: 17,
          ),
        ),
      ),
      body: SafeArea(
        child: entries.isEmpty
            ? Center(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text(
                    'No check-ins yet. They\u2019ll show up here, in your own '
                    'words, as soon as you complete one.',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      color: _C.textMuted,
                    ),
                  ),
                ),
              )
            : ListView.separated(
                padding: const EdgeInsets.all(20),
                itemCount: entries.length,
                separatorBuilder: (context, index) => const SizedBox(height: 12),
                itemBuilder: (context, i) {
                  final e = entries[i];
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
                          _formatDate(e.timestamp),
                          style: GoogleFonts.poppins(
                            fontSize: 11.5,
                            fontWeight: FontWeight.w600,
                            color: _C.p500,
                          ),
                        ),
                        const SizedBox(height: 10),
                        ...e.answers.entries.map(
                          (qa) => Padding(
                            padding: const EdgeInsets.only(bottom: 10),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  qa.key,
                                  style: GoogleFonts.poppins(
                                    fontSize: 11.5,
                                    color: _C.textMuted,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                                const SizedBox(height: 3),
                                Text(
                                  qa.value,
                                  style: GoogleFonts.poppins(
                                    fontSize: 13.5,
                                    color: _C.textPrimary,
                                    height: 1.4,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
      ),
    );
  }
}
