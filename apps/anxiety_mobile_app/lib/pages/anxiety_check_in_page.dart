import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../services/anxiety_feedback_service.dart';
import '../theme/app_theme.dart';

class AnxietyCheckInPage extends StatefulWidget {
  final String eventId;

  const AnxietyCheckInPage({super.key, required this.eventId});

  @override
  State<AnxietyCheckInPage> createState() => _AnxietyCheckInPageState();
}

class _AnxietyCheckInPageState extends State<AnxietyCheckInPage> {
  static const _activities = [
    'Resting',
    'Studying or working',
    'Exercising',
    'Traveling',
    'Socializing',
    'Other',
  ];

  AnxietyAlertEvent? _event;
  bool _loading = true;
  bool _saving = false;
  int _breathingSeconds = 120;
  Timer? _breathingTimer;
  final TextEditingController _otherActivityController =
      TextEditingController();
  final TextEditingController _alternativeActionController =
      TextEditingController();
  String? _selectedActivity;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  @override
  void dispose() {
    _breathingTimer?.cancel();
    _otherActivityController.dispose();
    _alternativeActionController.dispose();
    super.dispose();
  }

  Future<void> _reload() async {
    final event = await AnxietyFeedbackService.getEvent(widget.eventId);
    if (!mounted) return;
    setState(() {
      _event = event;
      _selectedActivity = event?.activity;
      _loading = false;
    });
  }

  Future<void> _saveConfirmation(bool confirmed) async {
    setState(() => _saving = true);
    await AnxietyFeedbackService.recordConfirmation(widget.eventId, confirmed);
    await _reload();
    if (mounted) setState(() => _saving = false);
  }

  Future<void> _saveActivity() async {
    var activity = _selectedActivity;
    if (activity == 'Other') {
      final custom = _otherActivityController.text.trim();
      if (custom.isEmpty) return;
      activity = custom;
    }
    if (activity == null) return;
    setState(() => _saving = true);
    await AnxietyFeedbackService.recordContext(widget.eventId, activity);
    await _reload();
    if (mounted) setState(() => _saving = false);
  }

  void _startBreathing() {
    _breathingTimer?.cancel();
    setState(() => _breathingSeconds = 120);
    _breathingTimer = Timer.periodic(const Duration(seconds: 1), (timer) async {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_breathingSeconds > 1) {
        setState(() => _breathingSeconds--);
        return;
      }
      timer.cancel();
      setState(() {
        _breathingSeconds = 0;
        _saving = true;
      });
      await AnxietyFeedbackService().recordIntervention(
        eventId: widget.eventId,
        completedGuidance: true,
      );
      await _reload();
      if (mounted) setState(() => _saving = false);
    });
  }

  Future<void> _saveAlternativeAction() async {
    final action = _alternativeActionController.text.trim();
    if (action.isEmpty) return;
    setState(() => _saving = true);
    await AnxietyFeedbackService().recordIntervention(
      eventId: widget.eventId,
      completedGuidance: false,
      alternativeAction: action,
    );
    await _reload();
    if (mounted) setState(() => _saving = false);
  }

  Future<void> _saveFeltBetter(bool value) async {
    setState(() => _saving = true);
    await AnxietyFeedbackService.recordFeltBetter(widget.eventId, value);
    if (!mounted) return;
    final closed = await Navigator.of(context).maybePop();
    if (!closed && mounted) {
      await _reload();
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final event = _event;
    if (event == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Aura check-in')),
        body: const Center(
          child: Text('This check-in is no longer available.'),
        ),
      );
    }

    final isFollowup = event.followupAt != null;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        title: Text(isFollowup ? 'Five-minute follow-up' : 'Quick check-in'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
        children: isFollowup
            ? [
                _followupCard(event),
                if (_saving) ...[
                  const SizedBox(height: 18),
                  const Center(child: CircularProgressIndicator()),
                ],
              ]
            : [
                _infoCard(
                  icon: Icons.monitor_heart_outlined,
                  title: event.predictedRiskScore == null
                      ? 'Aura noticed a change in your readings'
                      : 'Aura noticed a possible change in your readings',
                  body: event.predictedRiskScore == null
                      ? 'Aura uses recent body readings to offer a check-in. It cannot tell you whether you have anxiety.'
                      : 'This is a model estimate based on recent body readings, not a diagnosis. Take a moment to notice how you feel.',
                ),
                const SizedBox(height: 16),
                _questionCard(
                  title: 'Do you notice any anxiety right now?',
                  selected: event.confirmedAnxious,
                  onYes: () => _saveConfirmation(true),
                  onNo: () => _saveConfirmation(false),
                ),
                if (event.confirmedAnxious != null &&
                    event.activity == null) ...[
                  const SizedBox(height: 16),
                  _activityCard(event),
                ],
                if (event.confirmedAnxious == true &&
                    event.activity != null) ...[
                  const SizedBox(height: 16),
                  _guidanceCard(event),
                ],
                if (event.confirmedAnxious == false &&
                    event.activity != null) ...[
                  const SizedBox(height: 16),
                  _infoCard(
                    icon: Icons.check_circle_outline_rounded,
                    title: 'Thanks, that helps Aura learn',
                    body:
                        'Your answer helps Aura learn what is normal for you.',
                  ),
                ],
                if (_saving) ...[
                  const SizedBox(height: 18),
                  const Center(child: CircularProgressIndicator()),
                ],
              ],
      ),
    );
  }

  Widget _activityCard(AnxietyAlertEvent event) {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _title('What are you doing right now?'),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _activities.map((activity) {
              return ChoiceChip(
                label: Text(activity),
                selected:
                    _selectedActivity == activity ||
                    (activity == 'Other' &&
                        event.activity != null &&
                        !_activities.contains(event.activity)),
                onSelected: (_) => setState(() => _selectedActivity = activity),
              );
            }).toList(),
          ),
          if (_selectedActivity == 'Other') ...[
            const SizedBox(height: 10),
            TextField(
              controller: _otherActivityController,
              decoration: const InputDecoration(
                hintText: 'Briefly describe what you are doing',
                border: OutlineInputBorder(),
              ),
            ),
          ],
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: _saving ? null : _saveActivity,
              child: const Text('Continue'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _guidanceCard(AnxietyAlertEvent event) {
    if (event.interventionCompleted != null) {
      return _followupScheduledCard();
    }

    final breathingActive = _breathingTimer?.isActive ?? false;
    final cycleSecond = (120 - _breathingSeconds) % 10;
    final cue = cycleSecond < 4 ? 'Breathe in gently' : 'Breathe out slowly';
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _title('Try two minutes of slow breathing'),
          const SizedBox(height: 8),
          const Text(
            'If comfortable, breathe in for 4 seconds and out for 6 seconds. Stop if you feel dizzy or uncomfortable.',
          ),
          const SizedBox(height: 16),
          Center(
            child: Text(
              breathingActive
                  ? '$cue\n${_breathingSeconds ~/ 60}:${(_breathingSeconds % 60).toString().padLeft(2, '0')}'
                  : 'Ready when you are',
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: breathingActive ? 22 : 16,
                fontWeight: FontWeight.w600,
                color: AppTheme.kPrimaryDeep,
              ),
            ),
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: breathingActive ? null : _startBreathing,
              icon: const Icon(Icons.air_rounded),
              label: const Text('Start breathing exercise'),
            ),
          ),
          const Divider(height: 30),
          TextField(
            controller: _alternativeActionController,
            decoration: const InputDecoration(
              labelText: 'Did you try something else?',
              hintText: 'Example: took a walk or called someone',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 10),
          OutlinedButton(
            onPressed: _saving ? null : _saveAlternativeAction,
            child: const Text('Save what I did'),
          ),
        ],
      ),
    );
  }

  Widget _followupCard(AnxietyAlertEvent event) {
    final change = event.followupRiskScore == null
        ? null
        : event.followupRiskScore! - event.initialRiskScore;
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _title('Five-minute follow-up'),
          const SizedBox(height: 8),
          Text(
            event.predictedRiskScore != null &&
                    event.followupRiskScore != null &&
                    event.followupRiskScore! <= event.predictedRiskScore! - 10
                ? 'Your recent readings are lower than Aura\'s earlier estimate.'
                : change == null
                ? 'Aura could not get a current reading, but your answer still helps.'
                : change <= -10
                ? 'Your recent readings have moved lower.'
                : 'Your recent readings have changed since the first check-in.',
          ),
          const SizedBox(height: 12),
          _questionCard(
            title: 'Do you feel better now?',
            selected: event.feltBetter,
            onYes: () => _saveFeltBetter(true),
            onNo: () => _saveFeltBetter(false),
            nested: true,
          ),
        ],
      ),
    );
  }

  Widget _followupScheduledCard() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.schedule_rounded, color: AppTheme.kPrimaryDeep),
              const SizedBox(width: 10),
              Expanded(child: _title('Follow-up scheduled')),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'Aura will check again in 5 minutes and ask how you feel.',
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: () => Navigator.of(context).maybePop(),
              child: const Text('Done for now'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _questionCard({
    required String title,
    required bool? selected,
    required VoidCallback onYes,
    required VoidCallback onNo,
    bool nested = false,
  }) {
    final content = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _title(title),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: ChoiceChip(
                label: const Center(child: Text('Yes')),
                selected: selected == true,
                onSelected: _saving ? null : (_) => onYes(),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: ChoiceChip(
                label: const Center(child: Text('No')),
                selected: selected == false,
                onSelected: _saving ? null : (_) => onNo(),
              ),
            ),
          ],
        ),
      ],
    );
    return nested ? content : _card(child: content);
  }

  Widget _infoCard({
    required IconData icon,
    required String title,
    required String body,
  }) {
    return _card(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppTheme.kPrimaryDeep),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [_title(title), const SizedBox(height: 6), Text(body)],
            ),
          ),
        ],
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _title(String text) => Text(
    text,
    style: GoogleFonts.poppins(
      fontSize: 15,
      fontWeight: FontWeight.w600,
      color: Theme.of(context).colorScheme.onSurface,
    ),
  );
}
