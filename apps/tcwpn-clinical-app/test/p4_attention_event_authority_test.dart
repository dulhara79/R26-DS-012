import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('P4 Activity is server-event based and not legacy ClinicalAlert based', () {
    final activity =
        File('lib/features/attention_events/activity_screen.dart').readAsStringSync();
    final shell = File('lib/features/shell.dart').readAsStringSync();

    expect(activity, isNot(contains('ClinicalAlert')));
    expect(activity, isNot(contains('RosterController')));
    expect(activity, isNot(contains('AlertBandX.fromScore')));
    expect(shell, isNot(contains('roster.unacknowledgedCount')));
    expect(shell, isNot(contains('AlertsScreen()')));
  });

  test('P4 lifecycle controller never manufactures actor time or success', () {
    final source = File('lib/state/attention_event_detail_controller.dart')
        .readAsStringSync();

    expect(source, isNot(contains('DateTime.now()')));
    expect(source, isNot(contains('SecureStore.clinicianId')));
    expect(source, isNot(contains('copyWith(status:')));
    expect(source, contains('repository.acknowledge(eventId)'));
    expect(source, contains('repository.resolve(eventId)'));
    expect(source, contains('repository.eventById(eventId)'));
  });

  test('P4 production adapter contains no guessed attention-event route', () {
    final source = File('lib/data/repositories/central_backend_repositories.dart')
        .readAsStringSync();

    expect(source, isNot(contains("'/attention-events")));
    expect(source, isNot(contains("'/acknowledge")));
    expect(source, isNot(contains("'/resolve")));
    expect(source, contains("_contractGate('attention-event-detail-contract')"));
  });

  test('active controllers still do not mint authoritative ClinicalAlert events',
      () {
    final source = File('lib/state/controllers.dart').readAsStringSync();

    expect(source, isNot(contains('_raiseIfEscalated(')));
    expect(source, isNot(contains('roster.raiseAlert(ClinicalAlert(')));
  });

  test('P4 server-backed event views make no direct model prediction call', () {
    final activity =
        File('lib/features/attention_events/activity_screen.dart').readAsStringSync();
    final detail = File('lib/features/attention_events/attention_event_detail_screen.dart')
        .readAsStringSync();

    expect(activity, isNot(contains("'/predict'")));
    expect(detail, isNot(contains("'/predict'")));
    expect(detail, isNot(contains('AlertBandX.fromScore')));
  });
}
