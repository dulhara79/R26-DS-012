import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('clinical SharedPreferences stores are clinician scoped', () {
    final records = File('lib/data/local/stores.dart').readAsStringSync();
    final dashboard =
        File('lib/data/local/dashboard_cache.dart').readAsStringSync();
    final notifications = File(
      'lib/data/local/attention_notification_store.dart',
    ).readAsStringSync();

    expect(records, contains('ClinicianStorageScope.keyForCurrent'));
    expect(dashboard, contains('ClinicianStorageScope.keyForCurrent'));
    expect(notifications, contains('ClinicianStorageScope(deliveryScope)'));

    expect(records, isNot(contains("static const _kRoster = 'roster_v2'")));
    expect(dashboard, isNot(contains("static const _key = 'server_dashboard_v1'")));
  });

  test('hardening keeps server clinical authority intact', () {
    final attentionRepo = File(
      'lib/data/repositories/central_backend_repositories.dart',
    ).readAsStringSync();
    final overview =
        File('lib/features/patients/patient_overview_screen.dart').readAsStringSync();
    final activity =
        File('lib/features/attention_events/activity_screen.dart').readAsStringSync();

    expect(attentionRepo, isNot(contains('AlertBandX.fromScore')));
    expect(overview, isNot(contains('AlertBandX.fromScore')));
    expect(activity, isNot(contains('AlertBandX.fromScore')));
    expect(attentionRepo, contains('/v1/attention-events'));
  });

  test('release-safe defaults and accessibility cannot silently regress', () {
    final env = File('lib/core/config/env.dart').readAsStringSync();
    final main = File('lib/main.dart').readAsStringSync();

    expect(env, contains("'DEMO_DATA',\n    defaultValue: false"));
    expect(main, isNot(contains('MediaQuery.withClampedTextScaling')));
  });
}
