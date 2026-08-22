import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/background_service_helper.dart';
import 'digital_phenotyping_page.dart';

class ParticipantBehaviorPage extends StatefulWidget {
  final String? userId;

  const ParticipantBehaviorPage({super.key, this.userId});

  @override
  State<ParticipantBehaviorPage> createState() => _ParticipantBehaviorPageState();
}

class _ParticipantBehaviorPageState extends State<ParticipantBehaviorPage> {
  bool _loading = true;
  bool _showCollectionDetails = false;

  String _participantId = '';
  int _daysEnrolled = 0;
  int _daysWithData = 0;
  int _baselineDaysAvailable = 0;
  int _baselineDaysRequired = 28;
  int _emaReceived = 0;
  int _emaExpected = 0;
  int _pendingUploads = 0;
  bool _serviceRunning = false;

  List<_PatternItem> _patterns = const [];
  _ChangeDetection? _changeDetection;
  List<_CoverageDay> _coverage = const [];
  List<Map<String, dynamic>> _checkIns = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final id = widget.userId ?? await BackgroundServiceHelper.getCachedId();
    final enrolledRaw = prefs.getString('enrolled_date');
    final enrolled = enrolledRaw == null ? null : DateTime.tryParse(enrolledRaw);

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

    final patterns = observations.entries.map((entry) {
      final value = entry.value is Map<String, dynamic>
          ? entry.value as Map<String, dynamic>
          : <String, dynamic>{};
      return _PatternItem(
        label: (value['label'] ?? _friendlyLabel(entry.key)).toString(),
        direction: (value['direction'] ?? 'unknown').toString(),
        z: (value['z'] as num?)?.toDouble(),
      );
    }).toList();

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

    List<Map<String, dynamic>> checkIns = [];
    final checkInRaw = prefs.getString('c2_checkin_history');
    if (checkInRaw != null && checkInRaw.isNotEmpty) {
      try {
        checkIns = (jsonDecode(checkInRaw) as List)
            .whereType<Map<String, dynamic>>()
            .toList();
      } catch (_) {}
    }

    final queueSize = await BackgroundServiceHelper.getOfflineQueueSize();
    final running = await BackgroundServiceHelper.isServiceRunning();

    if (!mounted) return;
    setState(() {
      _participantId = id;
      _daysEnrolled = daysEnrolled;
      _daysWithData = (quality['days_with_data'] as num?)?.toInt() ?? 0;
      _baselineDaysAvailable =
          (quality['baseline_days_available'] as num?)?.toInt() ??
              daysEnrolled.clamp(0, 28);
      _baselineDaysRequired =
          (quality['baseline_days_required'] as num?)?.toInt() ?? 28;
      _emaReceived = (quality['ema_received'] as num?)?.toInt() ?? 0;
      _emaExpected = (quality['ema_expected'] as num?)?.toInt() ?? 0;
      _patterns = patterns;
      _changeDetection = change;
      _coverage = coverage;
      _checkIns = checkIns;
      _pendingUploads = queueSize;
      _serviceRunning = running;
      _loading = false;
    });
  }

  static String _friendlyLabel(String key) {
    final spaced = key.replaceAll('_', ' ');
    if (spaced.isEmpty) return key;
    return '${spaced[0].toUpperCase()}${spaced.substring(1)}';
  }

  int get _usableCoverageDays => _coverage.where((d) => d.usable).length;

  bool get _baselineReady => _baselineDaysAvailable >= _baselineDaysRequired;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF7F5FF),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFFF7F5FF),
        title: Text(
          'Behavioural Context',
          style: GoogleFonts.poppins(
            fontSize: 17,
            fontWeight: FontWeight.w600,
            color: const Color(0xFF2D3142),
          ),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.fromLTRB(18, 8, 18, 28),
                children: [
                  _introCard(),
                  const SizedBox(height: 14),
                  _baselineCard(),
                  const SizedBox(height: 18),
                  _sectionTitle('This week'),
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
                  const SizedBox(height: 18),
                  _collectionDetails(),
                  const SizedBox(height: 18),
                  _disclaimer(),
                ],
              ),
            ),
    );
  }

  Widget _introCard() => _card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Your recent patterns',
              style: GoogleFonts.poppins(
                fontSize: 22,
                fontWeight: FontWeight.w700,
                color: const Color(0xFF2D3142),
              ),
            ),
            const SizedBox(height: 6),
            Text(
              'This page compares your recent behaviour with your own usual patterns. It does not estimate or diagnose anxiety.',
              style: GoogleFonts.poppins(
                fontSize: 12.5,
                height: 1.5,
                color: const Color(0xFF676B80),
              ),
            ),
          ],
        ),
      );

  Widget _baselineCard() {
    final need = _baselineDaysRequired <= 0 ? 28 : _baselineDaysRequired;
    final have = _baselineDaysAvailable.clamp(0, need);
    final fraction = (have / need).clamp(0.0, 1.0);

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.auto_graph_rounded, color: Color(0xFF6D5BD0)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  _baselineReady
                      ? 'Your personal baseline is ready'
                      : 'Building your personal baseline',
                  style: GoogleFonts.poppins(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: const Color(0xFF2D3142),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            _baselineReady
                ? 'We can now compare your recent behaviour with your own usual patterns.'
                : 'We need about 28 days of information before showing personalised comparisons.',
            style: GoogleFonts.poppins(
              fontSize: 12,
              height: 1.45,
              color: const Color(0xFF676B80),
            ),
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: fraction,
              minHeight: 9,
              backgroundColor: const Color(0xFFE9E4FF),
              valueColor: const AlwaysStoppedAnimation(Color(0xFF6D5BD0)),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '$have of $need days collected',
            style: GoogleFonts.poppins(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: const Color(0xFF6D5BD0),
            ),
          ),
        ],
      ),
    );
  }

  Widget _patternsCard() {
    final visible = _patterns.isNotEmpty
        ? _patterns
        : const [
            _PatternItem(label: 'Screen activity', direction: 'unknown'),
            _PatternItem(label: 'Mobility', direction: 'unknown'),
            _PatternItem(label: 'Physical activity', direction: 'unknown'),
            _PatternItem(label: 'Routine regularity', direction: 'unknown'),
          ];

    return _card(
      child: Column(
        children: [
          for (int i = 0; i < visible.length; i++) ...[
            _patternRow(visible[i]),
            if (i != visible.length - 1)
              const Divider(height: 22, color: Color(0xFFE9E7F2)),
          ],
        ],
      ),
    );
  }

  Widget _patternRow(_PatternItem item) {
    String message;
    IconData icon;
    Color color;

    if (!_baselineReady || item.z == null) {
      message = 'Not enough information yet';
      icon = Icons.hourglass_empty_rounded;
      color = const Color(0xFF8B8FA3);
    } else if (item.direction == 'above') {
      message = 'Higher than your usual pattern';
      icon = Icons.trending_up_rounded;
      color = const Color(0xFF7D68D6);
    } else if (item.direction == 'below') {
      message = 'Lower than your usual pattern';
      icon = Icons.trending_down_rounded;
      color = const Color(0xFF7D68D6);
    } else {
      message = 'Similar to your usual pattern';
      icon = Icons.trending_flat_rounded;
      color = const Color(0xFF2D9C79);
    }

    return Row(
      children: [
        Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.10),
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
                  color: const Color(0xFF2D3142),
                ),
              ),
              const SizedBox(height: 2),
              Text(
                message,
                style: GoogleFonts.poppins(
                  fontSize: 11.5,
                  color: const Color(0xFF75798C),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  bool _shouldShowChangeDetection() {
    return _daysEnrolled >= 57 && (_changeDetection?.detected ?? false);
  }

  Widget _changeCard() => _card(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.notifications_none_rounded, color: Color(0xFFB7791F)),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Recent change noticed',
                    style: GoogleFonts.poppins(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF2D3142),
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    _changeDetection?.message ??
                        'A noticeable change in one of your recent behavioural patterns was detected.',
                    style: GoogleFonts.poppins(
                      fontSize: 12,
                      height: 1.45,
                      color: const Color(0xFF676B80),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'This is a pattern change, not an anxiety diagnosis or risk prediction.',
                    style: GoogleFonts.poppins(
                      fontSize: 11.5,
                      height: 1.45,
                      color: const Color(0xFF8A6A2F),
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
    final usable = _coverage.isEmpty ? _daysWithData.clamp(0, total) : _usableCoverageDays;

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$usable of the last $total days had enough data',
            style: GoogleFonts.poppins(
              fontSize: 13.5,
              fontWeight: FontWeight.w600,
              color: const Color(0xFF2D3142),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Low coverage can make behavioural comparisons less reliable. Missing data does not mean anything about your wellbeing.',
            style: GoogleFonts.poppins(
              fontSize: 11.5,
              height: 1.45,
              color: const Color(0xFF75798C),
            ),
          ),
        ],
      ),
    );
  }

  Widget _checkInCard() => _card(
        child: Row(
          children: [
            const Icon(Icons.edit_note_rounded, color: Color(0xFF6D5BD0)),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${_checkIns.length} check-ins recorded',
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: const Color(0xFF2D3142),
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    'Your check-ins stay separate from passive behavioural observations.',
                    style: GoogleFonts.poppins(
                      fontSize: 11.5,
                      height: 1.4,
                      color: const Color(0xFF75798C),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      );

  Widget _collectionDetails() => _card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            InkWell(
              onTap: () => setState(() => _showCollectionDetails = !_showCollectionDetails),
              child: Row(
                children: [
                  const Icon(Icons.settings_input_antenna_rounded, color: Color(0xFF6D5BD0)),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Data collection details',
                      style: GoogleFonts.poppins(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: const Color(0xFF2D3142),
                      ),
                    ),
                  ),
                  Icon(
                    _showCollectionDetails
                        ? Icons.keyboard_arrow_up_rounded
                        : Icons.keyboard_arrow_down_rounded,
                    color: const Color(0xFF8B8FA3),
                  ),
                ],
              ),
            ),
            if (_showCollectionDetails) ...[
              const Divider(height: 24, color: Color(0xFFE9E7F2)),
              _detailRow('Participant', _participantId.isEmpty ? 'Unknown' : _participantId),
              _detailRow('Days enrolled', '$_daysEnrolled'),
              _detailRow('Collection service', _serviceRunning ? 'Running' : 'Stopped'),
              _detailRow('Pending uploads', '$_pendingUploads'),
              if (_emaExpected > 0)
                _detailRow('Check-in coverage', '$_emaReceived / $_emaExpected'),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => DigitalPhenotypingPage(userId: widget.userId),
                      ),
                    );
                  },
                  icon: const Icon(Icons.science_outlined, size: 17),
                  label: const Text('Open research/debug view'),
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
                style: GoogleFonts.poppins(
                  fontSize: 11.5,
                  color: const Color(0xFF75798C),
                ),
              ),
            ),
            Text(
              value,
              style: GoogleFonts.poppins(
                fontSize: 11.5,
                fontWeight: FontWeight.w600,
                color: const Color(0xFF2D3142),
              ),
            ),
          ],
        ),
      );

  Widget _disclaimer() => Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: const Color(0xFFF0EDFF),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(
          'These are descriptive observations of your own behavioural patterns. They are not a diagnosis, anxiety risk score, or clinical prediction. Discuss any concerns with a qualified clinician.',
          style: GoogleFonts.poppins(
            fontSize: 11,
            height: 1.45,
            color: const Color(0xFF625B82),
          ),
        ),
      );

  Widget _sectionTitle(String title) => Text(
        title,
        style: GoogleFonts.poppins(
          fontSize: 16,
          fontWeight: FontWeight.w700,
          color: const Color(0xFF2D3142),
        ),
      );

  Widget _card({required Widget child}) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xFFE9E7F2)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.035),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: child,
      );
}

class _PatternItem {
  final String label;
  final String direction;
  final double? z;

  const _PatternItem({
    required this.label,
    required this.direction,
    this.z,
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
