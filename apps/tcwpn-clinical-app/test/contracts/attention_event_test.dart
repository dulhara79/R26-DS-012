import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/attention_event.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

import 'fixture_loader.dart';

void main() {
  test('OPEN event has no fabricated acknowledgement', () {
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_open.json'),
    );
    expect(event.status, AttentionEventStatus.open);
    expect(event.severity, AttentionSeverity.high);
    expect(event.acknowledgedAt, isNull);
    expect(event.resolvedAt, isNull);
  });

  test('ACKNOWLEDGED event preserves server actor and time', () {
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_acknowledged.json'),
    );
    expect(event.status, AttentionEventStatus.acknowledged);
    expect(event.acknowledgedBy, 'DR001');
    expect(event.acknowledgedAt, isNotNull);
    expect(event.resolvedAt, isNull);
  });

  test('RESOLVED event preserves the same event and source result ids', () {
    final event = AttentionEvent.fromJson(
      loadContractFixture('attention_event_resolved.json'),
    );
    expect(event.id, 'evt-001');
    expect(event.fusionResultId, 123);
    expect(event.forecastResultId, 'fcst-001');
    expect(event.status, AttentionEventStatus.resolved);
    expect(event.resolvedBy, 'DR001');
  });
}
