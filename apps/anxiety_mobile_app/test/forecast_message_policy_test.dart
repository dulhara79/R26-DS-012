import 'package:anxiety_mobile_app/services/forecast_message_policy.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('small model movement within the lower range stays calm', () {
    final summary = describeForecast(
      currentRisk: 18,
      forecast: const [20, 22, 24, 27, 29, 31, 30, 28, 26, 24],
    );

    expect(summary.tone, ForecastMessageTone.calm);
    expect(summary.title, 'Your recent readings look steady');
  });

  test('a real threshold crossing still produces an early warning', () {
    final summary = describeForecast(
      currentRisk: 20,
      forecast: const [24, 28, 32, 38, 45, 52, 60, 72, 78, 80],
    );

    expect(summary.tone, ForecastMessageTone.warning);
    expect(summary.leadMinutes, 8);
    expect(summary.title, 'Aura noticed a possible change in your readings');
  });

  test('a currently high reading takes priority over future wording', () {
    final summary = describeForecast(
      currentRisk: 76,
      forecast: const [74, 72, 70, 68, 66, 64, 62, 60, 58, 56],
    );

    expect(summary.tone, ForecastMessageTone.high);
    expect(summary.title, 'Take a gentle moment to check in');
  });
}
