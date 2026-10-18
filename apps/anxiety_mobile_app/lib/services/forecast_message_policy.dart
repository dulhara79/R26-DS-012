import 'dart:math';

enum ForecastMessageTone { calm, easing, elevated, warning, high }

class ForecastMessageSummary {
  final String title;
  final ForecastMessageTone tone;
  final double predictedPeak;
  final double forecastIncrease;
  final int? leadMinutes;

  const ForecastMessageSummary({
    required this.title,
    required this.tone,
    required this.predictedPeak,
    required this.forecastIncrease,
    this.leadMinutes,
  });

  bool get isUrgent => tone == ForecastMessageTone.warning;
}

/// Describes physiological anomaly magnitude and changes, never anxiety probability.
///
/// A high current anomaly with a flat forecast is not a predicted escalation.
/// Existing display bands describe score magnitude only; they are not clinical
/// anxiety thresholds. Use explicit +5/+10 horizons for the deployed model.
ForecastMessageSummary describeForecast({
  required double currentRisk,
  required List<double> forecast,
  List<int>? forecastHorizonsMinutes,
}) {
  final current = currentRisk.clamp(0.0, 100.0).toDouble();
  final scaledForecast = forecast
      .map((value) => value.clamp(0.0, 100.0).toDouble())
      .toList();
  final predictedPeak = scaledForecast.isEmpty
      ? current
      : scaledForecast.reduce(max);
  final increase = predictedPeak - current;

  final currentHigh = current >= 70;
  final currentElevated = current >= 45;
  final highEscalation = predictedPeak >= 70 && increase >= 10;
  final elevatedEscalation = predictedPeak >= 45 && increase >= 20;
  final escalationPredicted = highEscalation || elevatedEscalation;

  int? leadMinutes;
  if (escalationPredicted) {
    final target = highEscalation ? 70.0 : 45.0;
    final requiredIncrease = highEscalation ? 10.0 : 20.0;
    for (var index = 0; index < scaledForecast.length; index++) {
      final value = scaledForecast[index];
      if (value >= target && value - current >= requiredIncrease) {
        leadMinutes =
            forecastHorizonsMinutes != null &&
                forecastHorizonsMinutes.length == scaledForecast.length
            ? forecastHorizonsMinutes[index]
            : index + 1;
        break;
      }
    }
    leadMinutes ??= 1;
  }

  if (currentHigh) {
    return ForecastMessageSummary(
      title: 'Your readings differ from your resting baseline',
      tone: ForecastMessageTone.high,
      predictedPeak: predictedPeak,
      forecastIncrease: increase,
    );
  }

  if (escalationPredicted) {
    return ForecastMessageSummary(
      title: 'Your physiological deviation score may rise',
      tone: ForecastMessageTone.warning,
      predictedPeak: predictedPeak,
      forecastIncrease: increase,
      leadMinutes: leadMinutes,
    );
  }

  if (current < 45 && predictedPeak < 45) {
    return ForecastMessageSummary(
      title: 'Your recent readings look steady',
      tone: ForecastMessageTone.calm,
      predictedPeak: predictedPeak,
      forecastIncrease: increase,
    );
  }

  if (increase <= -10 && current > 20) {
    return ForecastMessageSummary(
      title: 'Your recent readings may be settling',
      tone: ForecastMessageTone.easing,
      predictedPeak: predictedPeak,
      forecastIncrease: increase,
    );
  }

  if (currentElevated) {
    return ForecastMessageSummary(
      title: 'Your readings differ from your resting baseline',
      tone: ForecastMessageTone.elevated,
      predictedPeak: predictedPeak,
      forecastIncrease: increase,
    );
  }

  return ForecastMessageSummary(
    title: 'Your recent readings look steady',
    tone: ForecastMessageTone.calm,
    predictedPeak: predictedPeak,
    forecastIncrease: increase,
  );
}
