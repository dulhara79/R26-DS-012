import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('Phase 6 notification delivery modules exist', () {
    expect(File('lib/core/notifications/attention_notification_gateway.dart').existsSync(), isTrue);
    expect(File('lib/core/notifications/flutter_attention_notification_gateway.dart').existsSync(), isTrue);
    expect(File('lib/data/local/attention_notification_store.dart').existsSync(), isTrue);
    expect(File('lib/state/attention_notification_controller.dart').existsSync(), isTrue);
  });

  test('notification layer does not manufacture clinical authority', () {
    final files = [
      'lib/core/notifications/attention_notification_gateway.dart',
      'lib/core/notifications/flutter_attention_notification_gateway.dart',
      'lib/data/local/attention_notification_store.dart',
      'lib/state/attention_notification_controller.dart',
    ];

    final source = files
        .where((path) => File(path).existsSync())
        .map((path) => File(path).readAsStringSync())
        .join('\n');

    expect(source, isNot(contains('AlertBandX.fromScore')));
    expect(source, isNot(contains('current_score')));
    expect(source, isNot(contains('forecast_score')));
    expect(source, isNot(contains('acknowledged_by')));
    expect(source, isNot(contains('acknowledged_at')));
    expect(source, isNot(contains('resolved_by')));
    expect(source, isNot(contains('resolved_at')));
    expect(source, isNot(contains("'/predict'")));
    expect(source, isNot(contains('ClinicalAlert(')));
  });

  test('notification content is generic and event-id based', () {
    const path = 'lib/core/notifications/flutter_attention_notification_gateway.dart';
    expect(File(path).existsSync(), isTrue);
    if (!File(path).existsSync()) return;

    final source = File(path).readAsStringSync();
    expect(source, contains('New attention event available.'));
    expect(source, contains('eventId'));
    expect(source, isNot(contains('patientMrn')));
    expect(source, isNot(contains('patientName')));
    expect(source, isNot(contains('reason: event.reason')));
  });

  test('live attention-event HTTP adapter remains contract gated', () {
    final source = File('lib/data/repositories/central_backend_repositories.dart')
        .readAsStringSync();

    expect(source, contains("_contractGate('attention-events-open-contract')"));
    expect(source, contains("_contractGate('attention-event-detail-contract')"));
    expect(source, isNot(contains("'/v1/attention-events")));
  });
}
