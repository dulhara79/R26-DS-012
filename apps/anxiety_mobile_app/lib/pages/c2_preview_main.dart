// Development-only Chrome preview of the Component 2 Behavioural Context page.
//
//   flutter run -d chrome -t lib/pages/c2_preview_main.dart
//
// With no COMPONENT2_API_URL configured, Component2DataService seeds a
// clearly labelled synthetic 60-day fixture (debug + web only), so the
// baseline, observation and Day-57 change-detection states can be reviewed
// and captured for documentation. It is never available in a release build.
//
// Pick a state with the URL query, e.g. http://localhost:8080/?state=baseline
//   (default)            Day 60 — observations plus Day-57+ change card
//   ?state=change        Day 60 — the same, seeded without the backend sync
//   ?state=observations  Day 40 — personal-baseline observations only
//   ?state=baseline      Day 12 — "Building your personal baseline"
// Add &theme=dark to capture the dark theme, and &page=summary to open the
// "Prepare for my appointment" screen for that state.

import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../theme/app_theme.dart';
import '../theme/theme_controller.dart';
import 'clinician_summary_page.dart';
import 'component2_bootstrap_page.dart';
import 'participant_behavior_page.dart';

const String _previewParticipantId = 'P_0000000000PREVIEW';

/// Debug and profile builds only. On web, profile builds report
/// kReleaseMode as true, so kProfileMode is checked explicitly.
const bool _previewAllowed = kDebugMode || kProfileMode;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ThemeController.instance.initialize();
  final query = Uri.base.queryParameters;
  if (query['theme'] == 'dark') {
    await ThemeController.instance.setMode(AppThemeMode.dark);
  } else if (query['theme'] == 'light') {
    await ThemeController.instance.setMode(AppThemeMode.light);
  }
  final state = query['state'];
  final seeded =
      _previewAllowed &&
      const {'baseline', 'observations', 'change'}.contains(state);
  if (seeded) await _seedState(state!);
  runApp(
    _C2PreviewApp(
      seeded: seeded,
      summary: _previewAllowed && query['page'] == 'summary',
    ),
  );
}

/// Writes a labelled synthetic payload for an earlier point in the timeline.
Future<void> _seedState(String state) async {
  final prefs = await SharedPreferences.getInstance();
  final now = DateTime.now();
  String day(int offset) {
    final d = now.subtract(Duration(days: offset));
    return '${d.year.toString().padLeft(4, '0')}-'
        '${d.month.toString().padLeft(2, '0')}-'
        '${d.day.toString().padLeft(2, '0')}';
  }

  final baseline = state == 'baseline';
  final change = state == 'change';
  final daysEnrolled = baseline
      ? 12
      : change
      ? 60
      : 40;
  final coverageDays = baseline ? 11 : 14;
  final coverage = [
    for (var i = coverageDays; i >= 1; i--)
      {'date': day(i), 'usable': i != 4 && i != 9},
  ];

  Map<String, dynamic> obs(
    String label,
    double value,
    String unit,
    double z,
    String direction,
  ) => {
    'label': label,
    'value': value,
    'unit': unit,
    'z': z,
    'direction': direction,
    'confidence': 'demo',
  };

  final payload = <String, dynamic>{
    'participant_id': _previewParticipantId,
    'synthetic': true,
    'baseline_ready': !baseline,
    'reportable': !baseline,
    'window': {'start': day(7), 'end': day(1)},
    'observations': baseline
        ? <String, dynamic>{}
        : {
            'screen_activity': change
                ? obs('Screen activity', 6.4, 'hours/day', 2.3, 'above')
                : obs('Screen activity', 4.6, 'hours/day', 1.3, 'above'),
            'mobility': obs('Mobility', 3.1, 'km/day', -1.2, 'below'),
            'movement_proxy': obs(
              'Movement proxy',
              10.8,
              '% high-motion samples',
              -0.2,
              'stable',
            ),
            'social_media_use': obs(
              'Social media use',
              49.0,
              'min/day',
              0.3,
              'stable',
            ),
          },
    'change_detection': baseline
        ? null
        : change
        ? {
            'detected': true,
            'feature': 'screen activity',
            'direction': 'above',
            'ewma_z': 2.18,
            'message':
                'A sustained change in screen activity compared with your '
                'usual pattern was detected.',
          }
        : {'detected': false},
    'data_quality': {
      'days_enrolled': daysEnrolled,
      'days_with_data': coverage.where((d) => d['usable'] == true).length,
      'baseline_calendar_days_elapsed': baseline ? 11 : 28,
      'baseline_days_with_features': baseline ? 10 : 27,
      'baseline_days_available': baseline ? 10 : 27,
      'baseline_days_required': 28,
      'baseline_usable_days': baseline ? 9 : 25,
      'baseline_min_usable_days': 14,
      'recent_usable_days': baseline ? 0 : 7,
    },
    'blocking_issues': baseline ? ['baseline_building'] : <String>[],
  };

  await prefs.setString('c2_observation_payload', jsonEncode(payload));
  await prefs.setString('c2_day_coverage', jsonEncode(coverage));
  await prefs.setString(
    'c2_last_sync_utc',
    now.subtract(const Duration(minutes: 5)).toUtc().toIso8601String(),
  );
}

class _C2PreviewApp extends StatelessWidget {
  final bool seeded;
  final bool summary;

  const _C2PreviewApp({required this.seeded, required this.summary});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: ThemeController.instance,
      builder: (context, _) => MaterialApp(
        title: 'Aura — Component 2 preview',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: ThemeController.instance.themeMode,
        home: !_previewAllowed
            ? const Scaffold(
                body: Center(
                  child: Text('The Component 2 preview is development-only.'),
                ),
              )
            : summary
            ? const ClinicianSummaryPage(userId: _previewParticipantId)
            : seeded
            ? const ParticipantBehaviorPage(userId: _previewParticipantId)
            : const Component2BootstrapPage(userId: _previewParticipantId),
      ),
    );
  }
}
