import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
import 'background_service_helper.dart';
import 'theme/app_theme.dart';
import 'main.dart'; 

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
  int? _selectedRating;
  String? _selectedContext;
  bool _saving = false;

  static const _emojis = ['😌', '😐', '😟', '😰', '😱'];
  static const _labels = [
    'Not anxious',
    'Slightly',
    'Moderately',
    'Very anxious',
    'Extremely',
  ];
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
    if (_selectedRating == null || _selectedContext == null) return;
    setState(() => _saving = true);

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    final today = DateFormat('yyyy-MM-dd').format(DateTime.now());

    final data = {
      'rating': _selectedRating,
      'context': _selectedContext,
      'period': widget.timePeriod,
      'date': today,
    };

    await BackgroundServiceHelper.sendToSheet(
      uid,
      'EMA_Rating_${widget.timePeriod}',
      jsonEncode(data),
    );

    await prefs.setString('ema_submitted_${widget.timePeriod}', today);

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

    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          const SizedBox(height: 20),

          Text(
            periodTitle,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'How anxious do you feel right now?',
            style: TextStyle(fontSize: 14, color: Colors.grey.shade600),
          ),
          const SizedBox(height: 20),

          // Emoji rating row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: List.generate(5, (i) {
              final isSelected = _selectedRating == i + 1;
              return GestureDetector(
                onTap: () => setState(() => _selectedRating = i + 1),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 150),
                  padding: const EdgeInsets.symmetric(
                    vertical: 10,
                    horizontal: 8,
                  ),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? AppTheme.kPrimaryDeep.withValues(alpha: 0.12)
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: isSelected ? AppTheme.kPrimaryDeep : Colors.transparent,
                      width: 2,
                    ),
                  ),
                  child: Column(
                    children: [
                      Text(
                        _emojis[i],
                        style: TextStyle(fontSize: isSelected ? 34 : 28),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '${i + 1}',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: isSelected ? AppTheme.kPrimaryDeep : Colors.grey,
                        ),
                      ),
                      Text(
                        _labels[i],
                        style: TextStyle(
                          fontSize: 9,
                          color: isSelected ? AppTheme.kPrimaryDeep : Colors.grey,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              );
            }),
          ),

          const SizedBox(height: 24),
          const Text(
            'What were you doing?',
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 12),

          // Context chips
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _contexts.map((ctx) {
              final isSelected = _selectedContext == ctx;
              return FilterChip(
                label: Text(ctx, style: const TextStyle(fontSize: 13)),
                selected: isSelected,
                onSelected: (_) => setState(() => _selectedContext = ctx),
                selectedColor: AppTheme.kPrimaryDeep.withValues(alpha: 0.15),
                checkmarkColor: AppTheme.kPrimaryDeep,
                side: BorderSide(
                  color: isSelected ? AppTheme.kPrimaryDeep : Colors.grey.shade300,
                ),
              );
            }).toList(),
          ),

          const SizedBox(height: 28),

          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed:
                  (_selectedRating != null &&
                      _selectedContext != null &&
                      !_saving)
                  ? _submit
                  : null,
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
    await prefs.setString('last_gad7_week', _weekKey());

    if (mounted) {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => AlertDialog(
          title: const Text('GAD-7 Complete'),
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
                'Thank you for completing this week\'s assessment. This data helps the research team understand your anxiety patterns.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey.shade700, fontSize: 13),
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

  String _weekKey() {
    final now = DateTime.now();
    final weekNum =
        ((now.difference(DateTime(now.year, 1, 1)).inDays +
                    DateTime(now.year, 1, 1).weekday) /
                7)
            .ceil();
    return '${now.year}-W${weekNum.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final allAnswered = _answers.every((a) => a != null);

    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      appBar: AppBar(
        title: const Text('Weekly GAD-7 Assessment'),
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
                    color: Colors.white,
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
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.black87,
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
                                    ? AppTheme.kPrimaryDeep.withValues(alpha: 0.1)
                                    : Colors.grey.shade50,
                                borderRadius: BorderRadius.circular(10),
                                border: Border.all(
                                  color: isSelected
                                      ? AppTheme.kPrimaryDeep
                                      : Colors.grey.shade200,
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
                                            ? Colors.black87
                                            : Colors.grey.shade700,
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
              color: Colors.white,
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
                        : const Text('Submit Assessment'),
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
  final weekNum =
      ((now.difference(DateTime(now.year, 1, 1)).inDays +
                  DateTime(now.year, 1, 1).weekday) /
              7)
          .ceil();
  final thisWeek = '${now.year}-W${weekNum.toString().padLeft(2, '0')}';
  final lastWeek = prefs.getString('last_gad7_week') ?? '';
  return lastWeek != thisWeek;
}
