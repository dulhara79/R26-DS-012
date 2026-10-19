import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('app does not globally cap the clinician system text scale', () {
    final source = File('lib/main.dart').readAsStringSync();

    expect(source, isNot(contains('MediaQuery.withClampedTextScaling')));
    expect(source, isNot(contains('maxScaleFactor: 1.3')));
  });

  test('critical risk and lifecycle views contain textual state labels', () {
    final dashboard =
        File('lib/features/dashboard/server_dashboard_screen.dart').readAsStringSync();
    final overview =
        File('lib/features/patients/patient_overview_screen.dart').readAsStringSync();
    final activity =
        File('lib/features/attention_events/activity_screen.dart').readAsStringSync();
    final detail = File(
      'lib/features/attention_events/attention_event_detail_screen.dart',
    ).readAsStringSync();

    expect(dashboard.toLowerCase(), contains('unavailable'));
    expect(overview.toLowerCase(), contains('unavailable'));
    expect(activity, contains('OPEN'));
    expect(detail, contains('ACKNOWLEDGED'));
    expect(detail, contains('RESOLVED'));
  });
}
