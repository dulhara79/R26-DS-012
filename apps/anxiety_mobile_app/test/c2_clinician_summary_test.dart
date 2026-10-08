import 'dart:convert';

import 'package:anxiety_mobile_app/services/c2_clinician_pdf.dart';
import 'package:anxiety_mobile_app/services/c2_clinician_summary.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> _context() => {
  'self_report_trend': {
    'seven_day': {
      'ema': {'count': 4, 'mean_anxiety': 2.5, 'mean_stress': 2},
    },
    'thirty_day': {
      'ema': {
        'count': 12,
        'mean_anxiety': 2.8,
        'mean_stress': 2.1,
        'common_context': 'Studying / Working',
      },
    },
    'gad7': {
      'available': true,
      'latest_score': 9,
      'latest_label': 'Mild anxiety',
      'previous_score': 7,
      'delta': 2,
    },
    'pss10': {'available': false},
  },
  'physiological_event_confirmations': {
    'seven_day': {'events': 0},
    'thirty_day': {
      'events': 8,
      'answered': 7,
      'confirmed_anxiety': 5,
      'common_context': 'Studying or working',
    },
  },
  'intervention_response': {
    'seven_day': {'intervention_attempts': 0, 'followups_answered': 0},
    'thirty_day': {
      'intervention_attempts': 4,
      'followups_answered': 3,
      'felt_better_count': 2,
      'most_helpful_action': '2-minute paced breathing',
    },
  },
  'c2_behavioral_changes': {
    'status': 'not_validated',
    'score': null,
    'baseline_ready': true,
    'patterns': [
      {
        'label': 'Screen activity',
        'direction': 'above',
        'within_person_z': 1.4,
      },
      {'label': 'Mobility', 'direction': 'stable'},
    ],
    'change_detection': {
      'detected': true,
      'feature': 'screen activity',
      'direction': 'above',
      'ewma_z': 2.1,
    },
    'data_quality': {'recent_usable_days': 6},
  },
};

void main() {
  final generated = DateTime(2026, 10, 8);

  test('30-day summary has four sections with readable rows', () {
    final s = ClinicianSummary.fromContext(
      _context(),
      participantId: 'P_0123456789ABCDEF',
      period: SummaryPeriod.month,
      generatedAt: generated,
    );
    expect(s.sections.map((x) => x.id), SummarySectionId.values);

    final self = s.sections[0].rows;
    expect(self.first.value, '12');
    expect(
      self.firstWhere((r) => r.label.startsWith('Latest GAD-7')).value,
      '9 / 21 (Mild anxiety), +2 from previous',
    );
    expect(self.any((r) => r.label.contains('PSS-10')), isFalse);

    final body = s.sections[1].rows;
    expect(body.firstWhere((r) => r.label == 'Answered').value, '7 of 8');
    expect(
      body.firstWhere((r) => r.label.startsWith('I confirmed')).value,
      '5 of 7 answered',
    );

    final helped = s.sections[2].rows;
    expect(
      helped.firstWhere((r) => r.label.startsWith('Felt better')).value,
      '2 of 3 follow-ups',
    );

    final behaviour = s.sections[3].rows;
    expect(behaviour.first.value, 'Higher than usual');
    expect(
      behaviour.firstWhere((r) => r.label.startsWith('Lasting')).value,
      'Higher than usual - screen activity',
    );
  });

  test('7-day period uses the 7-day figures and shows empty states', () {
    final s = ClinicianSummary.fromContext(
      _context(),
      participantId: 'P_0123456789ABCDEF',
      period: SummaryPeriod.week,
      generatedAt: generated,
    );
    expect(s.sections[0].rows.first.value, '4');
    expect(s.sections[1].isEmpty, isTrue);
    expect(s.sections[2].isEmpty, isTrue);
  });

  test('baseline still building is stated plainly, with no patterns', () {
    final ctx = _context();
    ctx['c2_behavioral_changes'] = {
      'baseline_ready': false,
      'data_quality': {'baseline_usable_days': 9, 'baseline_days_required': 28},
    };
    final s = ClinicianSummary.fromContext(
      ctx,
      participantId: 'P_0123456789ABCDEF',
      period: SummaryPeriod.month,
    );
    expect(s.sections[3].rows, hasLength(1));
    expect(s.sections[3].rows.first.value, contains('9 usable days'));
  });

  test('plain text respects excluded sections and never shows a score', () {
    final s = ClinicianSummary.fromContext(
      _context(),
      participantId: 'P_0123456789ABCDEF',
      period: SummaryPeriod.month,
      generatedAt: generated,
    );
    final text = s.toPlainText(
      include: {SummarySectionId.selfReport, SummarySectionId.behaviour},
    );
    expect(text, contains('HOW I HAVE BEEN FEELING'));
    expect(text, contains('BEHAVIOURAL PATTERNS'));
    expect(text, isNot(contains('BODY-SIGNAL CHECK-INS')));
    expect(text, isNot(contains('WHAT HELPED')));
    expect(text, isNot(contains('ewma')));
    expect(text, isNot(contains('1.4')));
    expect(text, contains('8 Oct 2026'));
  });

  test('PDF is generated for the included sections', () async {
    final s = ClinicianSummary.fromContext(
      _context(),
      participantId: 'P_0123456789ABCDEF',
      period: SummaryPeriod.month,
      generatedAt: generated,
    );
    final bytes = await buildClinicianSummaryPdf(
      s,
      include: SummarySectionId.values.toSet(),
    );
    expect(latin1.decode(bytes.sublist(0, 5)), '%PDF-');
    expect(bytes.length, greaterThan(1000));
  });

  test('pdfSafe replaces characters the standard PDF font cannot draw', () {
    expect(pdfSafe('a — b ’ ≥ c'), "a - b ' >= c");
    expect(pdfSafe('Day 1–28 · ok'), 'Day 1-28 · ok');
    expect(pdfSafe('\u{1F600}'), '?');
  });
}
