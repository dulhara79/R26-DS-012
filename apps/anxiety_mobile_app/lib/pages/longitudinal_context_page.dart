import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../services/clinician_insight_service.dart';
import '../services/clinician_longitudinal_context_service.dart';
import '../services/self_report_history_service.dart';

class LongitudinalContextPage extends StatefulWidget {
  final String userId;

  const LongitudinalContextPage({super.key, required this.userId});

  @override
  State<LongitudinalContextPage> createState() => _LongitudinalContextPageState();
}

class _LongitudinalContextPageState extends State<LongitudinalContextPage> {
  bool _loading = true;
  Map<String, dynamic> _context = <String, dynamic>{};
  List<CheckInRecord> _eventCheckIns = const [];
  List<SelfReportRecord> _selfReports = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final context =
        await ClinicianLongitudinalContextService.buildAndCache(widget.userId);
    final events = await ClinicianInsightService.loadCheckInRecords(
      widget.userId,
      days: 30,
    );
    final selfReports = await SelfReportHistoryService.loadRecords(
      widget.userId,
      days: 30,
      limit: 30,
    );
    if (!mounted) return;
    setState(() {
      _context = context;
      _eventCheckIns = events;
      _selfReports = selfReports;
      _loading = false;
    });
  }

  Map<String, dynamic> _map(dynamic value) =>
      value is Map ? Map<String, dynamic>.from(value) : <String, dynamic>{};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF7F5FF),
      appBar: AppBar(
        backgroundColor: const Color(0xFFF7F5FF),
        elevation: 0,
        title: Text(
          'Check-in history',
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
                padding: const EdgeInsets.fromLTRB(18, 8, 18, 30),
                children: [
                  _introCard(),
                  const SizedBox(height: 14),
                  _selfReportTrendCard(),
                  const SizedBox(height: 14),
                  _physiologicalConfirmationsCard(),
                  const SizedBox(height: 14),
                  _interventionResponseCard(),
                  const SizedBox(height: 14),
                  _behavioralChangesCard(),
                  const SizedBox(height: 20),
                  _sectionTitle('Recent self-reports'),
                  const SizedBox(height: 8),
                  if (_selfReports.isEmpty)
                    _emptyCard('No locally retained self-report history yet.')
                  else
                    for (final record in _selfReports.take(8)) ...[
                      _selfReportCard(record),
                      const SizedBox(height: 8),
                    ],
                  const SizedBox(height: 14),
                  _sectionTitle('Physiological event check-ins'),
                  const SizedBox(height: 8),
                  if (_eventCheckIns.isEmpty)
                    _emptyCard('No physiological event check-ins in the last 30 days.')
                  else
                    for (final record in _eventCheckIns.take(8)) ...[
                      _eventRecordCard(record),
                      const SizedBox(height: 8),
                    ],
                  const SizedBox(height: 12),
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
              'Your longitudinal context',
              style: GoogleFonts.poppins(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: const Color(0xFF2D3142),
              ),
            ),
            const SizedBox(height: 6),
            Text(
              'This combines your self-reports, responses to physiological alerts, intervention follow-ups and behavioural changes. These streams are shown separately so one signal is not mistaken for another.',
              style: GoogleFonts.poppins(
                fontSize: 11.5,
                height: 1.5,
                color: const Color(0xFF676B80),
              ),
            ),
          ],
        ),
      );

  Widget _selfReportTrendCard() {
    final selfReport = _map(_context['self_report_trend']);
    final seven = _map(selfReport['seven_day']);
    final ema = _map(seven['ema']);
    final gad = _map(selfReport['gad7']);
    final pss = _map(selfReport['pss10']);

    return _contextCard(
      icon: Icons.edit_note_rounded,
      title: 'Self-report trend',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _insightRow('EMA check-ins (7 days)', '${ema['count'] ?? 0}'),
          if (ema['mean_anxiety'] != null)
            _insightRow('Average EMA anxiety', '${ema['mean_anxiety']} / 5'),
          if (ema['mean_stress'] != null)
            _insightRow('Average EMA stress', '${ema['mean_stress']} / 4'),
          if (ema['common_context'] != null)
            _insightRow('Most common EMA context', ema['common_context'].toString()),
          _scoreTrendRow('GAD-7', gad, 21),
          _scoreTrendRow('PSS-10', pss, 40),
          const SizedBox(height: 7),
          _note(
            'Self-report values describe what you reported. Score changes are not interpreted as a diagnosis by the app.',
          ),
        ],
      ),
    );
  }

  Widget _scoreTrendRow(String label, Map<String, dynamic> trend, int maxScore) {
    if (trend['available'] != true || trend['latest_score'] == null) {
      return _insightRow(label, 'No local trend recorded yet');
    }
    final latest = trend['latest_score'];
    final delta = trend['delta'] as num?;
    final change = delta == null
        ? 'first locally retained result'
        : delta > 0
            ? '+${delta.toStringAsFixed(delta % 1 == 0 ? 0 : 1)} from previous'
            : delta < 0
                ? '${delta.toStringAsFixed(delta % 1 == 0 ? 0 : 1)} from previous'
                : 'unchanged from previous';
    return _insightRow(label, '$latest / $maxScore · $change');
  }

  Widget _physiologicalConfirmationsCard() {
    final physiological = _map(_context['physiological_event_confirmations']);
    final thirty = _map(physiological['thirty_day']);
    final answered = (thirty['answered'] as num?)?.toInt() ?? 0;
    final confirmed = (thirty['confirmed_anxiety'] as num?)?.toInt() ?? 0;
    final rate = (thirty['confirmation_rate'] as num?)?.toDouble();

    return _contextCard(
      icon: Icons.monitor_heart_outlined,
      title: 'Physiological event confirmations',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _insightRow('Answered alert check-ins', '$answered in 30 days'),
          _insightRow('Participant confirmed anxiety', '$confirmed'),
          if (rate != null)
            _insightRow('Confirmation rate', '${(rate * 100).round()}%'),
          if (thirty['common_context'] != null)
            _insightRow('Common situation', thirty['common_context'].toString()),
          const SizedBox(height: 7),
          _note(
            'A confirmation means the participant reported anxiety at that check-in. It does not prove every physiological alert was a clinical anxiety episode.',
          ),
        ],
      ),
    );
  }

  Widget _interventionResponseCard() {
    final interventions = _map(_context['intervention_response']);
    final thirty = _map(interventions['thirty_day']);
    final attempts = (thirty['intervention_attempts'] as num?)?.toInt() ?? 0;
    final followups = (thirty['followups_answered'] as num?)?.toInt() ?? 0;
    final better = (thirty['felt_better_count'] as num?)?.toInt() ?? 0;
    final rate = (thirty['felt_better_rate'] as num?)?.toDouble();

    return _contextCard(
      icon: Icons.self_improvement_rounded,
      title: 'Intervention response',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _insightRow('Actions attempted', '$attempts'),
          _insightRow('Follow-ups answered', '$followups'),
          _insightRow('Reported feeling better', '$better'),
          if (rate != null)
            _insightRow('Reported improvement rate', '${(rate * 100).round()}%'),
          if (thirty['most_helpful_action'] != null)
            _insightRow('Frequently helpful action', thirty['most_helpful_action'].toString()),
          const SizedBox(height: 7),
          _note(
            'These are observational follow-up reports and should not be interpreted as proof that an intervention caused improvement.',
          ),
        ],
      ),
    );
  }

  Widget _behavioralChangesCard() {
    final c2 = _map(_context['c2_behavioral_changes']);
    final patterns = c2['patterns'] is List ? c2['patterns'] as List : const [];
    final change = _map(c2['change_detection']);
    final quality = _map(c2['data_quality']);

    return _contextCard(
      icon: Icons.psychology_alt_outlined,
      title: 'C2 behavioural changes',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _insightRow('Personal baseline ready', c2['baseline_ready'] == true ? 'Yes' : 'No'),
          if (quality['recent_usable_days'] != null)
            _insightRow('Recent usable sensing days', quality['recent_usable_days'].toString()),
          for (final raw in patterns.take(4))
            if (raw is Map)
              _insightRow(
                raw['label']?.toString() ?? 'Behaviour',
                _directionLabel(raw['direction']?.toString()),
              ),
          if (change['detected'] == true)
            _insightRow(
              'Sustained change detected',
              '${change['feature'] ?? 'Behaviour'} · ${_directionLabel(change['direction']?.toString())}',
            ),
          const SizedBox(height: 7),
          _note(
            'C2 status: not validated. It contributes no numerical score to fusion; these are within-person behavioural observations only.',
          ),
        ],
      ),
    );
  }

  Widget _selfReportCard(SelfReportRecord record) {
    String detail;
    if (record.instrument == 'EMA') {
      final anxiety = record.metrics['anxiety'];
      final stress = record.metrics['stress'];
      detail = 'Anxiety ${anxiety ?? '—'} / 5 · Stress ${stress ?? '—'} / 4';
      if (record.context != null) detail += ' · ${record.context}';
    } else {
      final max = record.instrument == 'GAD-7' ? 21 : 40;
      detail = '${record.totalScore?.toStringAsFixed(0) ?? '—'} / $max';
      if (record.severity != null) detail += ' · ${record.severity}';
    }

    return _card(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.assignment_outlined, color: Color(0xFF6D5BD0)),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(record.instrument,
                    style: GoogleFonts.poppins(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w600,
                        color: const Color(0xFF2D3142))),
                const SizedBox(height: 3),
                Text(detail,
                    style: GoogleFonts.poppins(
                        fontSize: 11, color: const Color(0xFF676B80))),
                const SizedBox(height: 3),
                Text(_formatDateTime(record.recordedAt),
                    style: GoogleFonts.poppins(
                        fontSize: 10, color: const Color(0xFF8B8FA3))),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _eventRecordCard(CheckInRecord record) {
    final response = record.confirmedAnxious == null
        ? 'Not answered'
        : record.confirmedAnxious == true
            ? 'Reported anxiety'
            : 'Did not report anxiety';
    return _card(
      child: ExpansionTile(
        tilePadding: EdgeInsets.zero,
        childrenPadding: EdgeInsets.zero,
        title: Text(response,
            style: GoogleFonts.poppins(
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
                color: const Color(0xFF2D3142))),
        subtitle: Text(_formatDateTime(record.detectedAt),
            style: GoogleFonts.poppins(
                fontSize: 10, color: const Color(0xFF8B8FA3))),
        children: [
          if (record.activity != null) _detailRow('Context', record.activity!),
          if (record.actionTaken != null) _detailRow('Action', record.actionTaken!),
          if (record.feltBetter != null)
            _detailRow('Follow-up', record.feltBetter! ? 'Reported feeling better' : 'Did not report feeling better'),
          _detailRow('Source', _sourceLabel(record.riskSource)),
        ],
      ),
    );
  }

  Widget _contextCard({
    required IconData icon,
    required String title,
    required Widget child,
  }) =>
      _card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: const Color(0xFF2D9C79)),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(title,
                      style: GoogleFonts.poppins(
                          fontSize: 14.5,
                          fontWeight: FontWeight.w700,
                          color: const Color(0xFF2D3142))),
                ),
              ],
            ),
            const SizedBox(height: 10),
            child,
          ],
        ),
      );

  Widget _insightRow(String label, String value) => Padding(
        padding: const EdgeInsets.only(bottom: 7),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.only(top: 5),
              child: Icon(Icons.circle, size: 5, color: Color(0xFF2D9C79)),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: RichText(
                text: TextSpan(
                  style: GoogleFonts.poppins(
                      fontSize: 11,
                      height: 1.4,
                      color: const Color(0xFF676B80)),
                  children: [
                    TextSpan(text: '$label: ', style: const TextStyle(fontWeight: FontWeight.w600)),
                    TextSpan(text: value),
                  ],
                ),
              ),
            ),
          ],
        ),
      );

  Widget _detailRow(String label, String value) => Padding(
        padding: const EdgeInsets.only(bottom: 7),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 90,
              child: Text(label,
                  style: GoogleFonts.poppins(
                      fontSize: 10.5, color: const Color(0xFF8B8FA3))),
            ),
            Expanded(
              child: Text(value,
                  style: GoogleFonts.poppins(
                      fontSize: 10.8,
                      fontWeight: FontWeight.w500,
                      color: const Color(0xFF4F5368))),
            ),
          ],
        ),
      );

  Widget _note(String text) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0xFFF0EDFF),
          borderRadius: BorderRadius.circular(11),
        ),
        child: Text(text,
            style: GoogleFonts.poppins(
                fontSize: 10.2,
                height: 1.45,
                color: const Color(0xFF625B82))),
      );

  Widget _emptyCard(String text) => _card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Text(text,
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                  fontSize: 11, color: const Color(0xFF75798C))),
        ),
      );

  Widget _disclaimer() => Container(
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: const Color(0xFFFFF8E8),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(
          'Clinician longitudinal context should support, not replace, clinical assessment. Self-report, physiological confirmations, intervention follow-ups and behavioural changes have different meanings and are intentionally kept separate.',
          style: GoogleFonts.poppins(
              fontSize: 10.5,
              height: 1.45,
              color: const Color(0xFF795B26)),
        ),
      );

  Widget _sectionTitle(String text) => Text(text,
      style: GoogleFonts.poppins(
          fontSize: 15,
          fontWeight: FontWeight.w700,
          color: const Color(0xFF2D3142)));

  Widget _card({required Widget child}) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(15),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xFFE9E7F2)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.03),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: child,
      );

  String _directionLabel(String? direction) {
    switch (direction) {
      case 'above':
        return 'Higher than personal baseline';
      case 'below':
        return 'Lower than personal baseline';
      case 'stable':
      case 'similar':
        return 'Similar to personal baseline';
      default:
        return 'Not enough information';
    }
  }

  String _sourceLabel(String source) {
    switch (source) {
      case 'physiological_forecast':
        return 'Physiological forecast check-in';
      case 'physiological':
        return 'Physiological change check-in';
      default:
        return 'Aura check-in';
    }
  }

  String _formatDateTime(DateTime value) {
    final local = value.toLocal();
    final hour = local.hour % 12 == 0 ? 12 : local.hour % 12;
    final minute = local.minute.toString().padLeft(2, '0');
    final amPm = local.hour < 12 ? 'AM' : 'PM';
    return '${local.year}-${local.month.toString().padLeft(2, '0')}-${local.day.toString().padLeft(2, '0')} · $hour:$minute $amPm';
  }
}
