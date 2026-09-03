import 'package:anxiety_mobile_app/services/chest_strap_service.dart';
import 'package:anxiety_mobile_app/services/anxiety_feedback_service.dart';
import 'package:anxiety_mobile_app/services/anxiety_level_update_throttle.dart';
import 'package:anxiety_mobile_app/services/participant_identity_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() {
    SharedPreferences.setMockInitialValues({});
  });

  test('parses the exact 12-field ChestStrap_V3 BLE contract', () {
    final reading = ChestStrapReading.fromCsv(
      '1234,72.0,833.33,46.0,43.0,15.5,0.55,36.6,0.04,1.0,0.018,1',
    );

    expect(reading.timestamp, 1234);
    expect(reading.meanHR, 72.0);
    expect(reading.meanRR, 833.33);
    expect(reading.isWorn, isTrue);
  });

  test('in-app level alerts are limited to one update per minute', () {
    final throttle = AnxietyLevelUpdateThrottle();
    final start = DateTime.utc(2026, 8, 10, 12);
    throttle.seed('Low');

    final first = throttle.observe('Moderate', start);
    expect(first?.message, 'Your anxiety level changed from Low to Moderate.');

    expect(
      throttle.observe('Elevated', start.add(const Duration(seconds: 10))),
      isNull,
    );
    expect(
      throttle.observe('High', start.add(const Duration(seconds: 30))),
      isNull,
    );
    expect(throttle.flush(start.add(const Duration(seconds: 59))), isNull);

    final combined = throttle.flush(start.add(const Duration(minutes: 1)));
    expect(
      combined?.message,
      'Your anxiety level changed from Moderate to High.',
    );
  });

  test('in-app alerts ignore unavailable readings and cancelled changes', () {
    final throttle = AnxietyLevelUpdateThrottle();
    final start = DateTime.utc(2026, 8, 10, 12);
    throttle.seed('Low');

    throttle.observe('Moderate', start);
    throttle.observe('Elevated', start.add(const Duration(seconds: 10)));
    throttle.observe('Moderate', start.add(const Duration(seconds: 20)));
    expect(throttle.flush(start.add(const Duration(minutes: 1))), isNull);

    expect(
      throttle.observe('Unavailable', start.add(const Duration(minutes: 2))),
      isNull,
    );
    expect(
      throttle.observe('Low', start.add(const Duration(minutes: 3))),
      isNull,
    );
  });

  test('participant IDs contain no entered display name', () async {
    SharedPreferences.setMockInitialValues({});

    final participantId = await ParticipantIdentityService.createForDisplayName(
      'Real Person Name',
    );
    final preferences = await SharedPreferences.getInstance();

    expect(ParticipantIdentityService.isParticipantId(participantId), isTrue);
    expect(participantId, isNot(contains('Real')));
    expect(preferences.getString('display_name'), 'Real Person Name');
    expect(preferences.getString('user_id'), participantId);
  });

  test('display name can change without changing participant ID', () async {
    SharedPreferences.setMockInitialValues({});

    final participantId = await ParticipantIdentityService.createForDisplayName(
      'First Name',
    );
    await ParticipantIdentityService.updateDisplayName('New Name');
    final preferences = await SharedPreferences.getInstance();

    expect(preferences.getString('display_name'), 'New Name');
    expect(preferences.getString('participant_id'), participantId);
    expect(preferences.getString('user_id'), participantId);
  });

  test('predictive alert requires a rise five or more minutes ahead', () {
    final gate = PredictiveEscalationGate();
    final start = DateTime.utc(2026, 1, 1, 12);

    // A spike only in minutes 1-4 is too late to count as a five-minute
    // early-warning forecast.
    expect(
      gate.evaluate(
        currentRisk: 20,
        riskForecast: [80, 80, 80, 80, 25, 25, 25, 25, 25, 25],
        observedAt: start,
      ),
      isNull,
    );

    const risingForecast = <double>[25, 30, 35, 40, 48, 55, 62, 72, 78, 80];
    // One forecast is not enough; a second independent poll confirms the
    // trend without sacrificing the five-minute lead window.
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(seconds: 30)),
      ),
      isNull,
    );
    final escalation = gate.evaluate(
      currentRisk: 25,
      riskForecast: risingForecast,
      observedAt: start.add(const Duration(seconds: 60)),
    );
    expect(escalation, isNotNull);
    expect(escalation!.leadMinutes, greaterThanOrEqualTo(5));
    expect(escalation.predictedPeakRisk, 80);
    expect(escalation.increase, 55);

    // Do not repeat during the same episode.
    expect(
      gate.evaluate(
        currentRisk: 28,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(seconds: 90)),
      ),
      isNull,
    );

    // Low current and future risk re-arm the gate for a later episode.
    expect(
      gate.evaluate(
        currentRisk: 18,
        riskForecast: const [18, 18, 20, 20, 22, 22, 23, 23, 24, 24],
        observedAt: start.add(const Duration(minutes: 2)),
      ),
      isNull,
    );
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(minutes: 2, seconds: 30)),
      ),
      isNull,
    );
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(minutes: 3)),
      ),
      isNotNull,
    );
  });

  test('predictive alert does not reuse a stale confirmation', () {
    final gate = PredictiveEscalationGate();
    final start = DateTime.utc(2026, 1, 1, 12);
    const risingForecast = <double>[25, 30, 35, 40, 48, 55, 62, 72, 78, 80];

    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start,
      ),
      isNull,
    );
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(minutes: 5)),
      ),
      isNull,
    );
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(minutes: 5, seconds: 30)),
      ),
      isNotNull,
    );
  });

  test('a high current anxiety level triggers a check-in', () {
    final gate = PredictiveEscalationGate();
    final start = DateTime.utc(2026, 1, 1, 12);
    const flatHighForecast = <double>[98, 98, 97, 97, 96, 96, 95, 95, 94, 94];

    expect(
      gate.evaluate(
        currentRisk: 100,
        riskForecast: flatHighForecast,
        observedAt: start,
      ),
      isNull,
    );
    final alert = gate.evaluate(
      currentRisk: 100,
      riskForecast: flatHighForecast,
      observedAt: start.add(const Duration(seconds: 30)),
    );

    expect(alert, isNotNull);
    expect(alert!.leadMinutes, 0);
    expect(alert.predictedPeakRisk, 100);
  });

  test('predictive gate can retry after notification delivery fails', () {
    final gate = PredictiveEscalationGate();
    final start = DateTime.utc(2026, 1, 1, 12);
    const risingForecast = <double>[25, 30, 35, 40, 48, 55, 62, 72, 78, 80];

    gate.evaluate(
      currentRisk: 25,
      riskForecast: risingForecast,
      observedAt: start,
    );
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(seconds: 30)),
      ),
      isNotNull,
    );

    gate.allowRetry();
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(minutes: 1)),
      ),
      isNull,
    );
    expect(
      gate.evaluate(
        currentRisk: 25,
        riskForecast: risingForecast,
        observedAt: start.add(const Duration(minutes: 1, seconds: 30)),
      ),
      isNotNull,
    );
  });

}
