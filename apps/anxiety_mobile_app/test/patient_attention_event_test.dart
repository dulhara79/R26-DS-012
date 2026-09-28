import 'package:anxiety_mobile_app/services/patient_attention_event_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('parses the privacy-minimal server attention event projection', () {
    final event = PatientAttentionEvent.fromJson({
      'id': 'evt_shared_1',
      'event_type': 'acute_escalation_forecast',
      'severity': 'high',
      'forecast_horizon': 10,
      'status': 'OPEN',
      'created_at': '2026-09-27T12:00:00Z',
      'policy_version': 'escalation-v1',
    });

    expect(event.id, 'evt_shared_1');
    expect(event.forecastHorizon, 10);
    expect(event.status, 'OPEN');
    expect(event.createdAt, DateTime.utc(2026, 9, 27, 12));
  });

  test('rejects incomplete events instead of inventing defaults', () {
    expect(
      () => PatientAttentionEvent.fromJson({
        'id': 'evt_incomplete',
        'status': 'OPEN',
      }),
      throwsFormatException,
    );
  });
}
