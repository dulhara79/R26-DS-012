import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';

import 'background_service_helper.dart';
import 'services/notification_helper.dart';
import 'theme/app_theme.dart';

String weeklyCheckInWeekKey([DateTime? value]) {
  final date = value ?? DateTime.now();
  final localDate = DateTime(date.year, date.month, date.day);
  final monday = localDate.subtract(Duration(days: localDate.weekday - 1));
  return DateFormat('yyyy-MM-dd').format(monday);
}

bool _wasSubmittedThisWeek(String? submittedDate, DateTime now) {
  if (submittedDate == null || submittedDate.isEmpty) return false;
  final parsed = DateTime.tryParse(submittedDate);
  return parsed != null &&
      weeklyCheckInWeekKey(parsed) == weeklyCheckInWeekKey(now);
}

// ─────────────────────────────────────────────────────────
// EMA Rating Bottom Sheet (called 3x daily)
// ─────────────────────────────────────────────────────────

class EmaRatingSheet extends StatefulWidget {
  final String timePeriod; // 'morning' | 'afternoon' | 'evening'
  const EmaRatingSheet({super.key, required this.timePeriod});

  @override
  State<EmaRatingSheet> createState() => _EmaRatingSheetState();
}

class _EmaRatingSheetState extends State<EmaRatingSheet> {
  final List<int?> _ratings = List.filled(4, null);
  String? _selectedContext;
  bool _saving = false;

  static const _questions = [
    'How stressed do you feel right now?',
    'How anxious or worried do you feel right now?',
    'How mentally exhausted do you feel right now?',
    'How socially connected do you feel right now?',
  ];

  static const _distressEmojis = ['😌', '🙂', '😐', '😟', '😭'];
  static const _connectionEmojis = ['😢', '😟', '😐', '🙂', '😊'];

  static const _contexts = [
    'Studying / Working',
    'Socializing',
    'Commuting',
    'Resting / Relaxing',
    'Eating',
    'Exercising',
    'On phone / Social media',
    'Other',
  ];

  Future<void> _submit() async {
    if (_saving) return;
    if (_ratings.any((r) => r == null) || _selectedContext == null) return;
    setState(() => _saving = true);

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    final today = DateFormat('yyyy-MM-dd').format(DateTime.now());

    final submissionPeriod = widget.timePeriod;

    final data = {
      'stress': _ratings[0]! - 1,
      'anxiety': _ratings[1],
      'fatigue': _ratings[2],
      'social': _ratings[3],
      'context': _selectedContext,
      'period': submissionPeriod,
      'date': today,
    };

    await BackgroundServiceHelper.sendToSheet(
      uid,
      'EMA_Rating_$submissionPeriod',
      jsonEncode(data),
    );

    await prefs.setString('ema_submitted_${widget.timePeriod}', today);
    await NotificationHelper.cancelDailyCheckIn(widget.timePeriod);
    if (mounted) Navigator.pop(context, true);
  }

  @override
  Widget build(BuildContext context) {
    final periodTitle =
        {
          'morning': '☀️ Morning Check-in',
          'afternoon': '🌤️ Afternoon Check-in',
          'evening': '🌙 Evening Check-in',
        }[widget.timePeriod] ??
        'Check-in';

    final allAnswered =
        _ratings.every((r) => r != null) && _selectedContext != null;

    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.9,
      ),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            periodTitle,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          Flexible(
            child: ListView(
              shrinkWrap: true,
              children: [
                ...List.generate(_questions.length, (qi) => _buildQuestion(qi)),
                const SizedBox(height: 20),
                const Text(
                  'What were you doing?',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _contexts.map((ctx) {
                    final isSelected = _selectedContext == ctx;
                    return FilterChip(
                      label: Text(ctx, style: const TextStyle(fontSize: 13)),
                      selected: isSelected,
                      onSelected: (_) => setState(() => _selectedContext = ctx),
                      selectedColor: AppTheme.kPrimaryDeep.withValues(
                        alpha: 0.15,
                      ),
                      checkmarkColor: AppTheme.kPrimaryDeep,
                    );
                  }).toList(),
                ),
                const SizedBox(height: 32),
              ],
            ),
          ),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: allAnswered && !_saving ? _submit : null,
              child: _saving
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : const Text('Submit Check-in'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuestion(int index) {
    final emojis = index == 3 ? _connectionEmojis : _distressEmojis;
    final lowLabel = [
      '1 = Not at all stressed',
      '1 = Not at all anxious',
      '1 = Not at all exhausted',
      '1 = Not connected',
    ][index];
    final highLabel = [
      '5 = Very stressed',
      '5 = Very anxious',
      '5 = Very exhausted',
      '5 = Very connected',
    ][index];

    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _questions[index],
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Theme.of(context).colorScheme.onSurface,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: List.generate(5, (i) {
              final isSelected = _ratings[index] == i + 1;
              return GestureDetector(
                onTap: () => setState(() => _ratings[index] = i + 1),
                child: Column(
                  children: [
                    AnimatedScale(
                      scale: isSelected ? 1.2 : 1.0,
                      duration: const Duration(milliseconds: 150),
                      child: Text(
                        emojis[i],
                        style: const TextStyle(fontSize: 26),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${i + 1}',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: isSelected
                            ? FontWeight.bold
                            : FontWeight.normal,
                        color: isSelected ? AppTheme.kPrimaryDeep : Colors.grey,
                      ),
                    ),
                  ],
                ),
              );
            }),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  lowLabel,
                  style: TextStyle(
                    fontSize: 10,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  highLabel,
                  textAlign: TextAlign.end,
                  style: TextStyle(
                    fontSize: 10,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────
// GAD-7 Weekly Assessment Screen
// ─────────────────────────────────────────────────────────

class Gad7Screen extends StatefulWidget {
  const Gad7Screen({super.key});

  @override
  State<Gad7Screen> createState() => _Gad7ScreenState();
}

class _Gad7ScreenState extends State<Gad7Screen> {
  static const _questions = [
    'Feeling nervous, anxious, or on edge',
    'Not being able to stop or control worrying',
    'Worrying too much about different things',
    'Trouble relaxing',
    'Being so restless that it is hard to sit still',
    'Becoming easily annoyed or irritable',
    'Feeling afraid, as if something awful might happen',
  ];

  static const _options = [
    'Not at all',
    'Several days',
    'More than half the days',
    'Nearly every day',
  ];

  final List<int?> _answers = List.filled(7, null);
  bool _saving = false;

  int get _total => _answers.fold(0, (sum, v) => sum + (v ?? 0));

  String get _severity {
    if (_total <= 4) return 'Minimal anxiety';
    if (_total <= 9) return 'Mild anxiety';
    if (_total <= 14) return 'Moderate anxiety';
    return 'Severe anxiety';
  }

  Color get _severityColor {
    if (_total <= 4) return Colors.green;
    if (_total <= 9) return Colors.orange.shade600;
    if (_total <= 14) return Colors.deepOrange;
    return Colors.red.shade700;
  }

  Future<void> _submit() async {
    if (_saving) return;
    if (_answers.any((a) => a == null)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please answer all 7 questions.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    setState(() => _saving = true);

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    final today = DateFormat('yyyy-MM-dd').format(DateTime.now());

    final data = {
      'answers': _answers,
      'total_score': _total,
      'severity': _severity,
      'date': today,
    };

    await BackgroundServiceHelper.sendToSheet(
      uid,
      'GAD7_Weekly',
      jsonEncode(data),
    );

    await prefs.setString('last_gad7_submitted', today);
    await prefs.setString('last_gad7_week', weeklyCheckInWeekKey());
    await NotificationHelper.cancelWeeklyCheckIn('anxiety');

    if (mounted) {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => AlertDialog(
          title: const Text('Anxiety Check Complete'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Your score: $_total / 21',
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _severity,
                style: TextStyle(
                  fontSize: 16,
                  color: _severityColor,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Thank you for completing this week\'s check-in. Your answers help the research team understand how anxiety changes over time.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                  fontSize: 13,
                ),
              ),
            ],
          ),
          actions: [
            ElevatedButton(
              onPressed: () {
                Navigator.pop(ctx);
                Navigator.pop(context);
              },
              child: const Text('Done'),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final allAnswered = _answers.every((a) => a != null);

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Weekly Anxiety Check'),
        leading: const BackButton(),
      ),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            color: AppTheme.kPrimaryDeep.withValues(alpha: 0.08),
            child: Text(
              'Over the last 2 weeks, how often have you been bothered by the following problems?',
              style: TextStyle(
                fontSize: 14,
                color: Colors.teal.shade900,
                fontWeight: FontWeight.w500,
              ),
              textAlign: TextAlign.center,
            ),
          ),

          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _questions.length,
              itemBuilder: (context, qi) {
                return Container(
                  margin: const EdgeInsets.only(bottom: 14),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surface,
                    borderRadius: BorderRadius.circular(14),
                    border: _answers[qi] != null
                        ? Border.all(
                            color: AppTheme.kPrimaryDeep.withValues(alpha: 0.4),
                          )
                        : null,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.04),
                        blurRadius: 8,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            CircleAvatar(
                              radius: 13,
                              backgroundColor: _answers[qi] != null
                                  ? AppTheme.kPrimaryDeep
                                  : Colors.grey.shade200,
                              child: Text(
                                '${qi + 1}',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                  color: _answers[qi] != null
                                      ? Colors.white
                                      : Colors.grey,
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                _questions[qi],
                                style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  color: Theme.of(
                                    context,
                                  ).colorScheme.onSurface,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        ..._options.asMap().entries.map((entry) {
                          final score = entry.key;
                          final label = entry.value;
                          final isSelected = _answers[qi] == score;
                          return GestureDetector(
                            onTap: () => setState(() => _answers[qi] = score),
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 120),
                              margin: const EdgeInsets.only(bottom: 6),
                              padding: const EdgeInsets.symmetric(
                                horizontal: 14,
                                vertical: 10,
                              ),
                              decoration: BoxDecoration(
                                color: isSelected
                                    ? AppTheme.kPrimaryDeep.withValues(
                                        alpha: 0.1,
                                      )
                                    : Theme.of(
                                        context,
                                      ).colorScheme.surfaceContainerHighest,
                                borderRadius: BorderRadius.circular(10),
                                border: Border.all(
                                  color: isSelected
                                      ? AppTheme.kPrimaryDeep
                                      : Theme.of(
                                          context,
                                        ).colorScheme.outlineVariant,
                                  width: isSelected ? 2 : 1,
                                ),
                              ),
                              child: Row(
                                children: [
                                  Text(
                                    '$score',
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      color: isSelected
                                          ? AppTheme.kPrimaryDeep
                                          : Colors.grey.shade400,
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Text(
                                      label,
                                      style: TextStyle(
                                        fontSize: 13,
                                        color: isSelected
                                            ? Theme.of(
                                                context,
                                              ).colorScheme.onSurface
                                            : Theme.of(
                                                context,
                                              ).colorScheme.onSurfaceVariant,
                                        fontWeight: isSelected
                                            ? FontWeight.w600
                                            : FontWeight.normal,
                                      ),
                                    ),
                                  ),
                                  if (isSelected)
                                    const Icon(
                                      Icons.check_circle,
                                      color: AppTheme.kPrimaryDeep,
                                      size: 18,
                                    ),
                                ],
                              ),
                            ),
                          );
                        }),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),

          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.06),
                  blurRadius: 12,
                  offset: const Offset(0, -4),
                ),
              ],
            ),
            child: Column(
              children: [
                if (allAnswered) ...[
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Score: $_total / 21  •  ',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 15,
                        ),
                      ),
                      Text(
                        _severity,
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 15,
                          color: _severityColor,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                ],
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: allAnswered && !_saving ? _submit : null,
                    child: _saving
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : const Text('Submit Answers'),
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

// ─────────────────────────────────────────────────────────
// Helper: check if GAD-7 is due this week
// ─────────────────────────────────────────────────────────

Future<bool> isGad7DueThisWeek() async {
  final prefs = await SharedPreferences.getInstance();
  final now = DateTime.now();
  if (_wasSubmittedThisWeek(prefs.getString('last_gad7_submitted'), now)) {
    return false;
  }
  final thisWeek = weeklyCheckInWeekKey(now);
  final lastWeek = prefs.getString('last_gad7_week') ?? '';
  return lastWeek != thisWeek;
}

// ─────────────────────────────────────────────────────────
// PSS-10 Weekly Assessment Screen
// ─────────────────────────────────────────────────────────

class Pss10Screen extends StatefulWidget {
  const Pss10Screen({super.key});

  @override
  State<Pss10Screen> createState() => _Pss10ScreenState();
}

class _Pss10ScreenState extends State<Pss10Screen> {
  static const _questions = [
    'In the last week, how often have you been upset because of something that happened unexpectedly?',
    'In the last week, how often have you felt that you were unable to control the important things in your life?',
    'In the last week, how often have you felt nervous and stressed?',
    'In the last week, how often have you felt confident about your ability to handle your personal problems?',
    'In the last week, how often have you felt that things were going your way?',
    'In the last week, how often have you found that you could not cope with all the things that you had to do?',
    'In the last week, how often have you been able to control irritations in your life?',
    'In the last week, how often have you felt that you were on top of things?',
    'In the last week, how often have you been angered because of things that were outside of your control?',
    'In the last week, how often have you felt difficulties were piling up so high that you could not overcome them?',
  ];

  static const _options = [
    'Never',
    'Almost Never',
    'Sometimes',
    'Fairly Often',
    'Very Often',
  ];

  final List<int?> _answers = List.filled(10, null);
  bool _saving = false;

  int get _total {
    int sum = 0;
    for (int i = 0; i < _answers.length; i++) {
      int score = _answers[i] ?? 0;
      if (i == 3 || i == 4 || i == 6 || i == 7) {
        sum += (4 - score);
      } else {
        sum += score;
      }
    }
    return sum;
  }

  Future<void> _submit() async {
    if (_saving) return;
    if (_answers.any((a) => a == null)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please answer all 10 questions.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    setState(() => _saving = true);
    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    final today = DateFormat('yyyy-MM-dd').format(DateTime.now());

    final data = {
      'answers': _answers,
      'total_score': _total,
      'date': today,
      'type': 'weekly',
    };

    await BackgroundServiceHelper.sendToSheet(
      uid,
      'PSS10_Weekly',
      jsonEncode(data),
    );
    await prefs.setString('last_pss10_submitted', today);
    await prefs.setString('last_pss10_week', weeklyCheckInWeekKey());
    await NotificationHelper.cancelWeeklyCheckIn('stress');

    if (mounted) {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => AlertDialog(
          title: const Text('Stress Check Complete'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Your Score: $_total / 40',
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'Thank you. This weekly check helps track your stress over time.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 13),
              ),
            ],
          ),
          actions: [
            ElevatedButton(
              onPressed: () {
                Navigator.pop(ctx);
                Navigator.pop(context);
              },
              child: const Text('Done'),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final allAnswered = _answers.every((a) => a != null);

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(title: const Text('Weekly Stress Check')),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _questions.length,
              itemBuilder: (context, qi) => _buildQuestionCard(qi),
            ),
          ),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.06),
                  blurRadius: 12,
                  offset: const Offset(0, -4),
                ),
              ],
            ),
            child: Column(
              children: [
                if (allAnswered)
                  Text(
                    'Total Stress Score: $_total',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: allAnswered && !_saving ? _submit : null,
                    child: _saving
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : const Text('Submit Answers'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuestionCard(int qi) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: _answers[qi] != null
            ? Border.all(color: AppTheme.kPrimaryDeep.withValues(alpha: 0.3))
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${qi + 1}. ${_questions[qi]}',
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 16),
          ...List.generate(5, (si) {
            final isSelected = _answers[qi] == si;
            return GestureDetector(
              onTap: () => setState(() => _answers[qi] = si),
              child: Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: isSelected
                      ? AppTheme.kPrimaryDeep.withValues(alpha: 0.1)
                      : Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: isSelected
                        ? AppTheme.kPrimaryDeep
                        : Theme.of(context).colorScheme.outlineVariant,
                  ),
                ),
                child: Row(
                  children: [
                    Text(
                      '$si',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: isSelected ? AppTheme.kPrimaryDeep : Colors.grey,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      _options[si],
                      style: TextStyle(
                        fontSize: 13,
                        color: isSelected
                            ? Theme.of(context).colorScheme.onSurface
                            : Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                    const Spacer(),
                    if (isSelected)
                      const Icon(
                        Icons.check_circle,
                        color: AppTheme.kPrimaryDeep,
                        size: 16,
                      ),
                  ],
                ),
              ),
            );
          }),
        ],
      ),
    );
  }
}

Future<bool> isPss10DueThisWeek() async {
  final prefs = await SharedPreferences.getInstance();
  final now = DateTime.now();
  if (_wasSubmittedThisWeek(prefs.getString('last_pss10_submitted'), now)) {
    return false;
  }
  final thisWeek = weeklyCheckInWeekKey(now);

  final lastWeek = prefs.getString('last_pss10_week') ?? '';
  return lastWeek != thisWeek;
}
