import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/background_service_helper.dart';
import '../services/c2_view_logic.dart';
import '../services/clinician_insight_service.dart';
import '../services/component2_data_service.dart';
import '../services/self_report_history_service.dart';
import '../theme/c2_palette.dart';
import '../widgets/c2/crisis_banner.dart';
import 'clinician_summary_page.dart';
import 'digital_phenotyping_page.dart';

typedef _C = C2Palette;

class ParticipantBehaviorPage extends StatefulWidget {
  final String? userId;

  /// Status of the sync that ran just before this page opened
  /// (see [Component2SyncResult.status]). Null when no sync was attempted.
  final String? syncStatus;

  const ParticipantBehaviorPage({super.key, this.userId, this.syncStatus});

  @override
  State<ParticipantBehaviorPage> createState() =>
      _ParticipantBehaviorPageState();
}

class _ParticipantBehaviorPageState extends State<ParticipantBehaviorPage> {
  bool _loading = true;
  bool _showCollectionDetails = false;

  String _participantId = '';
  String? _syncStatus;
  DateTime? _lastSync;
  bool _synthetic = false;

  int _daysEnrolled = 0;
  int _daysWithData = 0;
  int _baselineCalendarDaysElapsed = 0;
  int _baselineDaysAvailable = 0;
  int _baselineUsableDays = 0;
  int _baselineDaysRequired = kC2BaselineDays;
  int _baselineMinUsableDays = 14;
  int _recentUsableDays = 0;
  bool _baselineReadyFromBackend = false;
  bool _reportable = false;
  int _emaReceived = 0;
  int _emaExpected = 0;
  int _pendingUploads = 0;
  bool _serviceRunning = false;

  List<_PatternItem> _patterns = const [];
  _ChangeDetection? _changeDetection;
  List<_CoverageDay> _coverage = const [];
  int _selfReports30d = 0;
  int _alertCheckIns30d = 0;

  @override
  void initState() {
    super.initState();
    _syncStatus = widget.syncStatus;
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final id = widget.userId ?? await BackgroundServiceHelper.getCachedId();
    final enrolledRaw = prefs.getString('enrolled_date');
    final enrolled = enrolledRaw == null
        ? null
        : DateTime.tryParse(enrolledRaw);

    final daysEnrolled = enrolled == null
        ? 0
        : DateTime.now().difference(enrolled).inDays.clamp(0, 9999);

    Map<String, dynamic>? payload;
    final payloadRaw = prefs.getString('c2_observation_payload');
    if (payloadRaw != null && payloadRaw.isNotEmpty) {
      try {
        payload = jsonDecode(payloadRaw) as Map<String, dynamic>;
      } catch (_) {}
    }

    final quality = payload?['data_quality'] is Map<String, dynamic>
        ? payload!['data_quality'] as Map<String, dynamic>
        : <String, dynamic>{};

    final observations = payload?['observations'] is Map<String, dynamic>
        ? payload!['observations'] as Map<String, dynamic>
        : <String, dynamic>{};

    final patterns = _orderedPatterns(observations);

    _ChangeDetection? change;
    final changeRaw = payload?['change_detection'];
    if (changeRaw is Map<String, dynamic>) {
      change = _ChangeDetection.fromJson(changeRaw);
    }

    List<_CoverageDay> coverage = [];
    final coverageRaw = prefs.getString('c2_day_coverage');
    if (coverageRaw != null && coverageRaw.isNotEmpty) {
      try {
        final decoded = jsonDecode(coverageRaw) as List;
        coverage = decoded
            .whereType<Map<String, dynamic>>()
            .map(_CoverageDay.fromJson)
            .toList();
      } catch (_) {}
    }

    // Check-ins come from the participant's own local history. The backend's
    // `checkin_history` field is always empty, so it is not used here.
    final selfReports = await SelfReportHistoryService.loadRecords(
      id,
      days: 30,
    );
    final alertCheckIns = await ClinicianInsightService.loadCheckInRecords(
      id,
      days: 30,
    );

    final queueSize = await BackgroundServiceHelper.getOfflineQueueSize();
    final running = await BackgroundServiceHelper.isServiceRunning();

    if (!mounted) return;
    setState(() {
      _participantId = id;
      _lastSync = DateTime.tryParse(
        prefs.getString('c2_last_sync_utc') ?? '',
      )?.toLocal();
      _synthetic = payload?['synthetic'] == true;
      // Prefer backend study-enrollment age when available. Reinstalling the
      // app can reset local SharedPreferences while backend history survives.
      _daysEnrolled =
          (quality['days_enrolled'] as num?)?.toInt() ?? daysEnrolled;
      _daysWithData = (quality['days_with_data'] as num?)?.toInt() ?? 0;
      _baselineCalendarDaysElapsed =
          (quality['baseline_calendar_days_elapsed'] as num?)?.toInt() ??
          daysEnrolled.clamp(0, kC2BaselineDays);
      _baselineDaysAvailable =
          (quality['baseline_days_with_features'] as num?)?.toInt() ??
          (quality['baseline_days_available'] as num?)?.toInt() ??
          0;
      _baselineUsableDays =
          (quality['baseline_usable_days'] as num?)?.toInt() ?? 0;
      _baselineDaysRequired =
          (quality['baseline_days_required'] as num?)?.toInt() ??
          kC2BaselineDays;
      _baselineMinUsableDays =
          (quality['baseline_min_usable_days'] as num?)?.toInt() ?? 14;
      _recentUsableDays = (quality['recent_usable_days'] as num?)?.toInt() ?? 0;
      _baselineReadyFromBackend = payload?['baseline_ready'] == true;
      _reportable = payload?['reportable'] == true;
      _emaReceived = (quality['ema_received'] as num?)?.toInt() ?? 0;
      _emaExpected = (quality['ema_expected'] as num?)?.toInt() ?? 0;
      _patterns = patterns;
      _changeDetection = change;
      _coverage = coverage;
      _selfReports30d = selfReports.length;
      _alertCheckIns30d = alertCheckIns.length;
      _pendingUploads = queueSize;
      _serviceRunning = running;
      _loading = false;
    });
  }

  /// The four thesis observations first, in a fixed order, followed by any
  /// additional observation the backend sends. Observations the backend did
  /// not send are shown as "not enough information yet".
  static List<_PatternItem> _orderedPatterns(Map<String, dynamic> raw) {
    _PatternItem fromEntry(String key, String fallbackLabel) {
      final value = raw[key] is Map<String, dynamic>
          ? raw[key] as Map<String, dynamic>
          : <String, dynamic>{};
      return _PatternItem(
        label: (value['label'] ?? fallbackLabel).toString(),
        direction: (value['direction'] ?? 'unknown').toString(),
        z: (value['z'] as num?)?.toDouble(),
        value: (value['value'] as num?)?.toDouble(),
        unit: (value['unit'] ?? '').toString(),
      );
    }

    final known = kC2Observations.map((spec) => spec.key).toSet();
    return [
      for (final spec in kC2Observations) fromEntry(spec.key, spec.label),
      for (final key in raw.keys)
        if (!known.contains(key)) fromEntry(key, _friendlyLabel(key)),
    ];
  }

  Future<void> _refreshFromBackend() async {
    final id = widget.userId ?? await BackgroundServiceHelper.getCachedId();
    if (id.isNotEmpty && id != 'No_User_ID') {
      final result = await Component2DataService.sync(id);
      _syncStatus = result.status;
    }
    await _load();
  }

  static String _friendlyLabel(String key) {
    final spaced = key.replaceAll('_', ' ');
    if (spaced.isEmpty) return key;
    return '${spaced[0].toUpperCase()}${spaced.substring(1)}';
  }

  int get _usableCoverageDays => _coverage.where((d) => d.usable).length;

  bool get _baselineReady => _baselineReadyFromBackend;

  void _openDetails() {
    Navigator.of(context)
        .push(
          MaterialPageRoute(
            builder: (_) => DigitalPhenotypingPage(userId: widget.userId),
          ),
        )
        .then((_) => _load());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _C.scaffold,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: _C.scaffold,
        title: Text(
          'Behavioural Context',
          style: GoogleFonts.poppins(
            fontSize: 17,
            fontWeight: FontWeight.w600,
            color: _C.textPrimary,
          ),
        ),
        actions: [
          IconButton(
            tooltip: 'Refresh',
            icon: Icon(Icons.refresh_rounded, color: _C.textMuted, size: 20),
            onPressed: () {
              setState(() => _loading = true);
              _refreshFromBackend();
            },
          ),
        ],
      ),
      body: _loading
          ? Center(child: CircularProgressIndicator(color: _C.primary))
          : RefreshIndicator(
              color: _C.primary,
              onRefresh: _refreshFromBackend,
              child: ListView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.fromLTRB(18, 8, 18, 28),
                children: [
                  if (_synthetic) ...[
                    _previewRibbon(),
                    const SizedBox(height: 12),
                  ],
                  _introCard(),
                  const SizedBox(height: 8),
                  _syncLine(),
                  if (!_serviceRunning && !kIsWeb) ...[
                    const SizedBox(height: 12),
                    _collectionStoppedCard(),
                  ],
                  const SizedBox(height: 14),
                  _baselineCard(),
                  const SizedBox(height: 18),
                  _sectionTitle('This week'),
                  const SizedBox(height: 4),
                  _sectionSubtitle(
                    'Your last 7 usable days compared with your own baseline',
                  ),
                  const SizedBox(height: 8),
                  _patternsCard(),
                  if (_shouldShowChangeDetection()) ...[
                    const SizedBox(height: 12),
                    _changeCard(),
                  ],
                  const SizedBox(height: 18),
                  _sectionTitle('Data quality'),
                  const SizedBox(height: 8),
                  _dataQualityCard(),
                  const SizedBox(height: 18),
                  _sectionTitle('Check-ins'),
                  const SizedBox(height: 8),
                  _checkInCard(),
                  const SizedBox(height: 12),
                  _appointmentCard(),
                  const SizedBox(height: 18),
                  _collectionDetails(),
                  const SizedBox(height: 18),
                  const C2CrisisBanner(compact: true),
                  const SizedBox(height: 12),
                  _disclaimer(),
                ],
              ),
            ),
    );
  }

  Widget _previewRibbon() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
    decoration: BoxDecoration(
      color: _C.amberBg,
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: _C.amber.withValues(alpha: 0.45)),
    ),
    child: Row(
      children: [
        Icon(Icons.science_outlined, size: 17, color: _C.amber),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            'Preview data — synthetic values for demonstration only, not a '
            'real participant.',
            style: GoogleFonts.poppins(
              fontSize: 11.5,
              fontWeight: FontWeight.w600,
              color: _C.textPrimary,
            ),
          ),
        ),
      ],
    ),
  );

  Widget _introCard() => _card(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Your recent patterns',
          style: GoogleFonts.poppins(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            color: _C.textPrimary,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          'This page compares your recent behaviour with your own usual '
          'patterns. It does not estimate or diagnose anxiety.',
          style: GoogleFonts.poppins(
            fontSize: 12.5,
            height: 1.5,
            color: _C.textSecondary,
          ),
        ),
      ],
    ),
  );

  Widget _syncLine() {
    final freshness = c2SyncFreshness(_lastSync, DateTime.now());
    final failed =
        _syncStatus != null &&
        _syncStatus != 'ok' &&
        _syncStatus != 'demo_data';

    final String text;
    final IconData icon;
    Color color = _C.textMuted;
    if (freshness == C2SyncFreshness.never) {
      text = failed
          ? 'Could not reach Aura’s server yet. Pull down to try again.'
          : 'Not updated yet. Your first summary appears after a full day.';
      icon = Icons.cloud_off_outlined;
    } else {
      final when = c2RelativeTime(_lastSync!, DateTime.now());
      if (failed) {
        text = 'Showing saved data from $when · offline';
        icon = Icons.cloud_off_outlined;
        color = _C.amber;
      } else if (freshness == C2SyncFreshness.stale) {
        text = 'Last updated $when · may be out of date';
        icon = Icons.history_rounded;
        color = _C.amber;
      } else {
        text = 'Updated $when';
        icon = Icons.cloud_done_outlined;
      }
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Row(
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              text,
              style: GoogleFonts.poppins(fontSize: 11, color: color),
            ),
          ),
        ],
      ),
    );
  }

  Widget _collectionStoppedCard() => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.amberBg,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: _C.amber.withValues(alpha: 0.4)),
    ),
    child: Row(
      children: [
        Icon(Icons.sensors_off_rounded, color: _C.amber, size: 20),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Data collection has stopped',
                style: GoogleFonts.poppins(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: _C.textPrimary,
                ),
              ),
              Text(
                _pendingUploads > 0
                    ? '$_pendingUploads readings are waiting to upload.'
                    : 'New days will not count towards your baseline.',
                style: GoogleFonts.poppins(
                  fontSize: 11.5,
                  color: _C.textSecondary,
                ),
              ),
            ],
          ),
        ),
        TextButton(
          onPressed: _openDetails,
          style: TextButton.styleFrom(foregroundColor: _C.primary),
          child: Text(
            'Fix',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w700),
          ),
        ),
      ],
    ),
  );

  Widget _baselineCard() {
    final need = _baselineDaysRequired <= 0
        ? kC2BaselineDays
        : _baselineDaysRequired;
    final elapsed = _baselineCalendarDaysElapsed.clamp(0, need);
    final fraction = (elapsed / need).clamp(0.0, 1.0);

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.auto_graph_rounded, color: _C.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  _baselineReady
                      ? 'Your personal baseline is ready'
                      : 'Building your personal baseline',
                  style: GoogleFonts.poppins(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: _C.textPrimary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            _baselineReady
                ? 'Aura can now compare your recent behaviour with your own '
                      'usual patterns.'
                : 'Aura learns what is usual for you over your first $need '
                      'days. It needs at least $_baselineMinUsableDays of those '
                      'days to have enough sensing data.',
            style: GoogleFonts.poppins(
              fontSize: 12,
              height: 1.45,
              color: _C.textSecondary,
            ),
          ),
          if (!_baselineReady) ...[
            const SizedBox(height: 14),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: fraction,
                minHeight: 9,
                backgroundColor: _C.p100,
                valueColor: AlwaysStoppedAnimation(_C.primary),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '$elapsed of $need days completed',
              style: GoogleFonts.poppins(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: _C.primary,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '$_baselineUsableDays of $_baselineMinUsableDays usable days '
              'collected',
              style: GoogleFonts.poppins(
                fontSize: 11.5,
                color: _C.textSecondary,
              ),
            ),
            if (_baselineDaysAvailable > _baselineUsableDays) ...[
              const SizedBox(height: 3),
              Text(
                '${_baselineDaysAvailable - _baselineUsableDays} more '
                'day${_baselineDaysAvailable - _baselineUsableDays == 1 ? '' : 's'} '
                'had some data, but not enough to count.',
                style: GoogleFonts.poppins(fontSize: 10.5, color: _C.textMuted),
              ),
            ],
          ],
          const SizedBox(height: 16),
          _timeline(),
        ],
      ),
    );
  }

  /// Days 1–28 baseline · Day 29+ comparisons · Day 57+ change detection
  /// (thesis Figure 4.3).
  Widget _timeline() {
    final stage = c2StageFor(_daysEnrolled);
    Widget step(
      C2Stage s,
      String days,
      String label,
      IconData icon, {
      required bool last,
    }) {
      final reached = stage.index >= s.index;
      final current = stage == s;
      final color = reached ? _C.primary : _C.textMuted;
      return Expanded(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 26,
                  height: 26,
                  decoration: BoxDecoration(
                    color: current
                        ? _C.primary
                        : reached
                        ? _C.p100
                        : _C.chip,
                    shape: BoxShape.circle,
                    border: Border.all(color: reached ? _C.primary : _C.border),
                  ),
                  child: Icon(
                    reached && !current ? Icons.check_rounded : icon,
                    size: 14,
                    color: current ? _C.cardBase : color,
                  ),
                ),
                if (!last)
                  Expanded(
                    child: Container(
                      height: 2,
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      color: stage.index > s.index ? _C.primary : _C.border,
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              days,
              style: GoogleFonts.poppins(
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(right: 6),
              child: Text(
                label,
                style: GoogleFonts.poppins(
                  fontSize: 10,
                  height: 1.3,
                  color: current ? _C.textPrimary : _C.textMuted,
                ),
              ),
            ),
          ],
        ),
      );
    }

    return Semantics(
      label: 'Study timeline. You are on day $_daysEnrolled.',
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          step(
            C2Stage.baseline,
            'Days 1–28',
            'Learning your usual patterns',
            Icons.hourglass_top_rounded,
            last: false,
          ),
          step(
            C2Stage.observations,
            'Day 29+',
            'Weekly comparisons',
            Icons.insights_rounded,
            last: false,
          ),
          step(
            C2Stage.changeDetection,
            'Day 57+',
            'Notices lasting changes',
            Icons.notifications_none_rounded,
            last: true,
          ),
        ],
      ),
    );
  }

  Widget _patternsCard() {
    final String? note;
    if (!_baselineReady) {
      note = 'Comparisons start once your baseline is ready.';
    } else if (!_reportable) {
      note =
          'Not enough data this week yet. Aura needs at least 3 usable days '
          '($_recentUsableDays so far).';
    } else {
      note = null;
    }

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (note != null) ...[
            Row(
              children: [
                Icon(Icons.info_outline_rounded, size: 15, color: _C.textMuted),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    note,
                    style: GoogleFonts.poppins(
                      fontSize: 11.5,
                      color: _C.textSecondary,
                    ),
                  ),
                ),
              ],
            ),
            Divider(height: 22, color: _C.border),
          ],
          for (int i = 0; i < _patterns.length; i++) ...[
            _patternRow(_patterns[i]),
            if (i != _patterns.length - 1)
              Divider(height: 22, color: _C.border),
          ],
        ],
      ),
    );
  }

  Widget _patternRow(_PatternItem item) {
    final available = _reportable && item.z != null;
    String message;
    IconData icon;
    Color color;

    if (!available) {
      message = 'Not enough information yet';
      icon = Icons.hourglass_empty_rounded;
      color = _C.textMuted;
    } else if (item.direction == 'above') {
      message = 'Higher than your usual pattern';
      icon = Icons.trending_up_rounded;
      color = _C.primary;
    } else if (item.direction == 'below') {
      message = 'Lower than your usual pattern';
      icon = Icons.trending_down_rounded;
      color = _C.primary;
    } else {
      message = 'Similar to your usual pattern';
      icon = Icons.trending_flat_rounded;
      color = _C.teal;
    }

    final valueText = available ? c2FormatValue(item.value, item.unit) : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.label,
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: _C.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    message,
                    style: GoogleFonts.poppins(
                      fontSize: 11.5,
                      color: _C.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
            if (valueText != null)
              Text(
                valueText,
                style: GoogleFonts.poppins(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w700,
                  color: _C.textPrimary,
                ),
              ),
          ],
        ),
        if (available) ...[
          const SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.only(left: 50),
            child: _RangeMarker(z: item.z!, color: color),
          ),
        ],
      ],
    );
  }

  bool _shouldShowChangeDetection() {
    // The backend already enforces CHANGE_DETECTION_START_DAY = 57.
    // Re-checking local install age can hide a valid result after reinstall.
    return _changeDetection?.detected ?? false;
  }

  Widget _changeCard() => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: _C.amberBg,
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: _C.amber.withValues(alpha: 0.35)),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.notifications_none_rounded, color: _C.amber),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Lasting change noticed',
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: _C.textPrimary,
                ),
              ),
              const SizedBox(height: 5),
              Text(
                _changeDetection?.message ??
                    'A sustained change in one of your recent behavioural '
                        'patterns was detected.',
                style: GoogleFonts.poppins(
                  fontSize: 12,
                  height: 1.45,
                  color: _C.textPrimary,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'This is a pattern change, not an anxiety diagnosis or risk '
                'prediction. If it matches how you have been feeling, you '
                'may want to mention it to your clinician.',
                style: GoogleFonts.poppins(
                  fontSize: 11.5,
                  height: 1.45,
                  color: _C.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _dataQualityCard() {
    final total = _coverage.isEmpty ? 14 : _coverage.length;
    final usable = _coverage.isEmpty
        ? _daysWithData.clamp(0, total)
        : _usableCoverageDays;

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$usable of the last $total completed days had enough sensing data',
            style: GoogleFonts.poppins(
              fontSize: 13.5,
              fontWeight: FontWeight.w600,
              color: _C.textPrimary,
            ),
          ),
          if (_coverage.isNotEmpty) ...[
            const SizedBox(height: 12),
            Semantics(
              label: '$usable of $total days usable',
              child: Row(
                children: [
                  for (final day in _coverage)
                    Expanded(
                      child: Tooltip(
                        message:
                            '${day.date.day}/${day.date.month}: '
                            '${day.usable ? 'usable' : 'not enough data'}',
                        child: Container(
                          height: 22,
                          margin: const EdgeInsets.symmetric(horizontal: 2),
                          decoration: BoxDecoration(
                            color: day.usable ? _C.teal : _C.p100,
                            borderRadius: BorderRadius.circular(5),
                            border: day.usable
                                ? null
                                : Border.all(color: _C.border),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 6),
            Row(
              children: [
                Text(
                  '${_coverage.first.date.day}/${_coverage.first.date.month}',
                  style: GoogleFonts.poppins(fontSize: 10, color: _C.textMuted),
                ),
                const Spacer(),
                _legendDot(_C.teal, 'Usable'),
                const SizedBox(width: 10),
                _legendDot(_C.p100, 'Not enough', outlined: true),
                const Spacer(),
                Text(
                  '${_coverage.last.date.day}/${_coverage.last.date.month}',
                  style: GoogleFonts.poppins(fontSize: 10, color: _C.textMuted),
                ),
              ],
            ),
          ],
          const SizedBox(height: 10),
          Text(
            'Low coverage makes comparisons less reliable. Missing days '
            'happen (phone off, battery saver) and say nothing about your '
            'wellbeing.',
            style: GoogleFonts.poppins(
              fontSize: 11.5,
              height: 1.45,
              color: _C.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _legendDot(Color color, String label, {bool outlined = false}) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Container(
        width: 9,
        height: 9,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(3),
          border: outlined ? Border.all(color: _C.border) : null,
        ),
      ),
      const SizedBox(width: 4),
      Text(
        label,
        style: GoogleFonts.poppins(fontSize: 10, color: _C.textMuted),
      ),
    ],
  );

  Widget _checkInCard() => _card(
    child: Row(
      children: [
        Icon(Icons.edit_note_rounded, color: _C.primary),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _selfReports30d + _alertCheckIns30d == 0
                    ? 'No check-ins in the last 30 days'
                    : '$_selfReports30d questionnaire'
                          '${_selfReports30d == 1 ? '' : 's'} · '
                          '$_alertCheckIns30d alert check-in'
                          '${_alertCheckIns30d == 1 ? '' : 's'} '
                          'in the last 30 days',
                style: GoogleFonts.poppins(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: _C.textPrimary,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                'What you tell Aura is kept separate from the patterns above, '
                'which come only from your phone’s sensors.',
                style: GoogleFonts.poppins(
                  fontSize: 11.5,
                  height: 1.4,
                  color: _C.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _appointmentCard() => _card(
    child: InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: () => Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ClinicianSummaryPage(userId: widget.userId),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: _C.p100,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(Icons.description_outlined, color: _C.primary),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Prepare for my appointment',
                  style: GoogleFonts.poppins(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: _C.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'A one-page summary you choose to share as a PDF.',
                  style: GoogleFonts.poppins(
                    fontSize: 11.5,
                    color: _C.textSecondary,
                  ),
                ),
              ],
            ),
          ),
          Icon(Icons.chevron_right_rounded, color: _C.textMuted),
        ],
      ),
    ),
  );

  Widget _collectionDetails() => _card(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        InkWell(
          onTap: () =>
              setState(() => _showCollectionDetails = !_showCollectionDetails),
          child: Row(
            children: [
              Icon(Icons.settings_input_antenna_rounded, color: _C.primary),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Data collection details',
                  style: GoogleFonts.poppins(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: _C.textPrimary,
                  ),
                ),
              ),
              Icon(
                _showCollectionDetails
                    ? Icons.keyboard_arrow_up_rounded
                    : Icons.keyboard_arrow_down_rounded,
                color: _C.textMuted,
              ),
            ],
          ),
        ),
        if (_showCollectionDetails) ...[
          Divider(height: 24, color: _C.border),
          _detailRow(
            'Participant',
            _participantId.isEmpty ? 'Unknown' : _participantId,
          ),
          _detailRow('Days enrolled', '$_daysEnrolled'),
          _detailRow(
            'Collection service',
            _serviceRunning ? 'Running' : 'Stopped',
          ),
          _detailRow('Pending uploads', '$_pendingUploads'),
          if (_emaExpected > 0)
            _detailRow('Check-in coverage', '$_emaReceived / $_emaExpected'),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: _openDetails,
              style: TextButton.styleFrom(foregroundColor: _C.primary),
              icon: const Icon(Icons.sensors_rounded, size: 17),
              label: const Text('Open sensing & data details'),
            ),
          ),
        ],
      ],
    ),
  );

  Widget _detailRow(String label, String value) => Padding(
    padding: const EdgeInsets.only(bottom: 7),
    child: Row(
      children: [
        Expanded(
          child: Text(
            label,
            style: GoogleFonts.poppins(fontSize: 11.5, color: _C.textSecondary),
          ),
        ),
        Text(
          value,
          style: GoogleFonts.poppins(
            fontSize: 11.5,
            fontWeight: FontWeight.w600,
            color: _C.textPrimary,
          ),
        ),
      ],
    ),
  );

  Widget _disclaimer() => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.p100,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: _C.p200),
    ),
    child: Text(
      'These are descriptive observations of your own behavioural patterns. '
      'They are not a diagnosis, anxiety risk score, or clinical prediction. '
      'Discuss any concerns with a qualified clinician.',
      style: GoogleFonts.poppins(
        fontSize: 11,
        height: 1.45,
        color: _C.textSecondary,
      ),
    ),
  );

  Widget _sectionTitle(String title) => Text(
    title,
    style: GoogleFonts.poppins(
      fontSize: 16,
      fontWeight: FontWeight.w700,
      color: _C.textPrimary,
    ),
  );

  Widget _sectionSubtitle(String text) => Text(
    text,
    style: GoogleFonts.poppins(fontSize: 11.5, color: _C.textMuted),
  );

  Widget _card({required Widget child}) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: _C.cardBase,
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: _C.border),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withValues(alpha: 0.04),
          blurRadius: 10,
          offset: const Offset(0, 4),
        ),
      ],
    ),
    child: child,
  );
}

/// A small bar showing where this week sits relative to the participant's
/// own usual range (shaded band = ±1σ of their baseline).
class _RangeMarker extends StatelessWidget {
  final double z;
  final Color color;

  const _RangeMarker({required this.z, required this.color});

  @override
  Widget build(BuildContext context) {
    final position = c2RangePosition(z);
    return Semantics(
      label: z.abs() < 1
          ? 'Inside your usual range'
          : 'Outside your usual range',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            height: 14,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final width = constraints.maxWidth;
                return Stack(
                  clipBehavior: Clip.none,
                  alignment: Alignment.centerLeft,
                  children: [
                    Container(
                      height: 6,
                      decoration: BoxDecoration(
                        color: _C.chip,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                    Positioned(
                      left: width * kC2UsualBandStart,
                      width: width * (kC2UsualBandEnd - kC2UsualBandStart),
                      child: Container(
                        height: 6,
                        decoration: BoxDecoration(
                          color: _C.p200,
                          borderRadius: BorderRadius.circular(3),
                        ),
                      ),
                    ),
                    Positioned(
                      left: (width * position - 7).clamp(0.0, width - 14),
                      child: Container(
                        width: 14,
                        height: 14,
                        decoration: BoxDecoration(
                          color: color,
                          shape: BoxShape.circle,
                          border: Border.all(color: _C.cardBase, width: 2),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Text(
                'Lower',
                style: GoogleFonts.poppins(fontSize: 9.5, color: _C.textMuted),
              ),
              const Spacer(),
              Text(
                'Your usual range',
                style: GoogleFonts.poppins(fontSize: 9.5, color: _C.textMuted),
              ),
              const Spacer(),
              Text(
                'Higher',
                style: GoogleFonts.poppins(fontSize: 9.5, color: _C.textMuted),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _PatternItem {
  final String label;
  final String direction;
  final double? z;
  final double? value;
  final String unit;

  const _PatternItem({
    required this.label,
    required this.direction,
    this.z,
    this.value,
    this.unit = '',
  });
}

class _ChangeDetection {
  final bool detected;
  final String message;

  const _ChangeDetection({required this.detected, required this.message});

  factory _ChangeDetection.fromJson(Map<String, dynamic> json) {
    return _ChangeDetection(
      detected: json['detected'] as bool? ?? false,
      message: json['message']?.toString() ?? '',
    );
  }
}

class _CoverageDay {
  final DateTime date;
  final bool usable;

  const _CoverageDay({required this.date, required this.usable});

  factory _CoverageDay.fromJson(Map<String, dynamic> json) {
    return _CoverageDay(
      date: DateTime.tryParse(json['date']?.toString() ?? '') ?? DateTime.now(),
      usable: json['usable'] as bool? ?? false,
    );
  }
}
