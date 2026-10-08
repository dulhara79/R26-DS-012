/// Participant-controlled clinician summary ("Prepare for my appointment").
///
/// Turns the descriptive payload produced by
/// `ClinicianLongitudinalContextService.buildAndCache` into four readable
/// sections. The on-screen preview, the copied text and the PDF are all built
/// from the same [ClinicianSummary], so they always show the same content.
///
/// Nothing here sends data anywhere: the participant decides whether to share.
library;

enum SummaryPeriod { week, month }

extension SummaryPeriodInfo on SummaryPeriod {
  /// Key used by the longitudinal-context payload.
  String get contextKey =>
      this == SummaryPeriod.week ? 'seven_day' : 'thirty_day';
  String get label =>
      this == SummaryPeriod.week ? 'Last 7 days' : 'Last 30 days';
}

enum SummarySectionId { selfReport, bodySignals, whatHelped, behaviour }

class SummaryRow {
  final String label;
  final String value;
  const SummaryRow(this.label, this.value);
}

class SummarySection {
  final SummarySectionId id;
  final String title;
  final String description;
  final List<SummaryRow> rows;

  /// Shown instead of rows when there is nothing to report.
  final String emptyText;

  /// One-line "how to read this" note for the clinician.
  final String note;

  const SummarySection({
    required this.id,
    required this.title,
    required this.description,
    required this.rows,
    required this.emptyText,
    required this.note,
  });

  bool get isEmpty => rows.isEmpty;
}

class ClinicianSummary {
  final String participantId;
  final DateTime generatedAt;
  final SummaryPeriod period;
  final List<SummarySection> sections;

  const ClinicianSummary({
    required this.participantId,
    required this.generatedAt,
    required this.period,
    required this.sections,
  });

  static const String disclaimer =
      'Participant-shared descriptive summary from the Aura research app. '
      'It is not a diagnosis, a risk score or a clinical prediction, and it '
      'does not replace clinical assessment.';

  static const String privacyNote =
      'Contains no exact location, location history, app names, call or '
      'message content, and no behavioural model score.';

  factory ClinicianSummary.fromContext(
    Map<String, dynamic> context, {
    required String participantId,
    required SummaryPeriod period,
    DateTime? generatedAt,
  }) {
    return ClinicianSummary(
      participantId: participantId,
      generatedAt: generatedAt ?? DateTime.now(),
      period: period,
      sections: [
        _selfReport(context, period),
        _bodySignals(context, period),
        _whatHelped(context, period),
        _behaviour(context),
      ],
    );
  }

  /// Plain-text version for copying, limited to the included sections.
  String toPlainText({Set<SummarySectionId>? include}) {
    final buf = StringBuffer()
      ..writeln('AURA - SUMMARY FOR MY CLINICIAN')
      ..writeln('Participant ID: $participantId')
      ..writeln('Period: ${period.label}')
      ..writeln('Prepared: ${formatDate(generatedAt)}')
      ..writeln()
      ..writeln(disclaimer)
      ..writeln();
    for (final section in sections) {
      if (include != null && !include.contains(section.id)) continue;
      buf.writeln(section.title.toUpperCase());
      if (section.isEmpty) {
        buf.writeln(section.emptyText);
      } else {
        for (final row in section.rows) {
          buf.writeln('${row.label}: ${row.value}');
        }
      }
      buf
        ..writeln('How to read this: ${section.note}')
        ..writeln();
    }
    buf.writeln(privacyNote);
    return buf.toString();
  }

  static String formatDate(DateTime d) {
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
    return '${d.day} ${months[d.month - 1]} ${d.year}';
  }

  // ─── Sections ────────────────────────────────────────────────────────────

  static SummarySection _selfReport(
    Map<String, dynamic> context,
    SummaryPeriod period,
  ) {
    final trend = _map(context['self_report_trend']);
    final ema = _map(_map(trend[period.contextKey])['ema']);
    final rows = <SummaryRow>[];

    final emaCount = _int(ema['count']);
    if (emaCount > 0) {
      rows.add(SummaryRow('Daily check-ins completed', '$emaCount'));
      void avg(String key, String label, int max) {
        final v = ema[key];
        if (v is num) rows.add(SummaryRow(label, '${_num(v)} / $max'));
      }

      avg('mean_anxiety', 'Average anxiety / worry', 5);
      avg('mean_stress', 'Average stress', 4);
      avg('mean_fatigue', 'Average tiredness', 5);
      avg('mean_social_connection', 'Average social connection', 5);
      final context0 = ema['common_context'];
      if (context0 != null) {
        rows.add(SummaryRow('Most common situation', '$context0'));
      }
    }

    void score(String key, String label, int max) {
      final s = _map(trend[key]);
      if (s['available'] != true || s['latest_score'] is! num) return;
      final latest = _num(s['latest_score'] as num);
      final band = s['latest_label'] == null ? '' : ' (${s['latest_label']})';
      final delta = s['delta'];
      final change = delta is num
          ? delta == 0
                ? ', unchanged from previous'
                : ', ${delta > 0 ? '+' : ''}${_num(delta)} from previous'
          : '';
      rows.add(SummaryRow(label, '$latest / $max$band$change'));
    }

    score('gad7', 'Latest GAD-7 (anxiety)', 21);
    score('pss10', 'Latest PSS-10 (stress)', 40);

    return SummarySection(
      id: SummarySectionId.selfReport,
      title: 'How I have been feeling',
      description: 'Daily check-ins and weekly questionnaires',
      rows: rows,
      emptyText: 'No check-ins or questionnaires recorded in this period.',
      note:
          'Self-reported. GAD-7 and PSS-10 show the most recent result, '
          'whatever the period.',
    );
  }

  static SummarySection _bodySignals(
    Map<String, dynamic> context,
    SummaryPeriod period,
  ) {
    final p = _map(
      _map(context['physiological_event_confirmations'])[period.contextKey],
    );
    final events = _int(p['events']);
    final rows = <SummaryRow>[];
    if (events > 0) {
      final answered = _int(p['answered']);
      rows
        ..add(SummaryRow('Aura check-ins from body signals', '$events'))
        ..add(SummaryRow('Answered', '$answered of $events'))
        ..add(
          SummaryRow(
            'I confirmed feeling anxious',
            '${_int(p['confirmed_anxiety'])} of $answered answered',
          ),
        );
      if (p['common_context'] != null) {
        rows.add(
          SummaryRow('Usually happened while', '${p['common_context']}'),
        );
      }
    }
    return SummarySection(
      id: SummarySectionId.bodySignals,
      title: 'Body-signal check-ins',
      description: 'When the chest strap prompted a check-in',
      rows: rows,
      emptyText: 'No body-signal check-ins in this period.',
      note:
          'Shows how often I agreed with the app\'s check-in. It does not '
          'confirm that every alert was a clinical anxiety episode.',
    );
  }

  static SummarySection _whatHelped(
    Map<String, dynamic> context,
    SummaryPeriod period,
  ) {
    final i = _map(_map(context['intervention_response'])[period.contextKey]);
    final attempts = _int(i['intervention_attempts']);
    final followups = _int(i['followups_answered']);
    final rows = <SummaryRow>[];
    if (attempts > 0 || followups > 0) {
      rows
        ..add(SummaryRow('Things I tried after a check-in', '$attempts'))
        ..add(
          SummaryRow(
            'Felt better 5 minutes later',
            '${_int(i['felt_better_count'])} of $followups follow-ups',
          ),
        );
      if (i['most_helpful_action'] != null) {
        rows.add(
          SummaryRow('What helped most often', '${i['most_helpful_action']}'),
        );
      }
    }
    return SummarySection(
      id: SummarySectionId.whatHelped,
      title: 'What helped',
      description: 'Breathing exercises and other actions I tried',
      rows: rows,
      emptyText: 'No actions or follow-ups recorded in this period.',
      note:
          'Self-reported and observational. It does not show that an action '
          'caused the improvement.',
    );
  }

  static SummarySection _behaviour(Map<String, dynamic> context) {
    final c2 = _map(context['c2_behavioral_changes']);
    final quality = _map(c2['data_quality']);
    final rows = <SummaryRow>[];

    if (c2['baseline_ready'] != true) {
      final usable = _int(quality['baseline_usable_days']);
      final required = _int(quality['baseline_days_required'], 28);
      rows.add(
        SummaryRow(
          'Personal baseline',
          'Still being built ($usable usable days so far, first $required days)',
        ),
      );
    } else {
      final patterns = c2['patterns'];
      if (patterns is List) {
        for (final raw in patterns) {
          final pattern = _map(raw);
          final label = pattern['label']?.toString();
          if (label == null) continue;
          rows.add(SummaryRow(label, _direction(pattern['direction'])));
        }
      }
      final change = _map(c2['change_detection']);
      rows.add(
        SummaryRow(
          'Lasting change (Day 57+)',
          change['detected'] == true
              ? '${_direction(change['direction'])} - '
                    '${change['feature'] ?? 'a behavioural pattern'}'
              : 'None detected',
        ),
      );
      if (quality['recent_usable_days'] is num) {
        rows.add(
          SummaryRow(
            'Usable sensing days this week',
            '${_int(quality['recent_usable_days'])} of 7',
          ),
        );
      }
    }

    return SummarySection(
      id: SummarySectionId.behaviour,
      title: 'Behavioural patterns',
      description: 'From phone sensors, compared with my own usual week',
      rows: rows,
      emptyText: 'No behavioural data yet.',
      note:
          'Compared only with my own baseline, never with other people. '
          'Not a validated anxiety measure and not part of any risk score.',
    );
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────

  static Map<String, dynamic> _map(dynamic v) =>
      v is Map ? Map<String, dynamic>.from(v) : <String, dynamic>{};

  static int _int(dynamic v, [int fallback = 0]) =>
      v is num ? v.toInt() : fallback;

  static String _num(num v) =>
      v % 1 == 0 ? v.toInt().toString() : v.toStringAsFixed(1);

  static String _direction(dynamic d) => switch (d?.toString()) {
    'above' => 'Higher than usual',
    'below' => 'Lower than usual',
    'stable' => 'Similar to usual',
    _ => 'Not enough data',
  };
}
