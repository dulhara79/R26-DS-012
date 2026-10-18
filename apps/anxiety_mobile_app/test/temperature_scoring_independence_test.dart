import 'package:anxiety_mobile_app/services/chest_strap_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('temperature values remain available but cannot affect local risk', () {
    const calmTemperature = ChestStrapReading(
      timestamp: 1,
      meanHR: 95,
      meanRR: 631.6,
      sdnn: 55,
      rmssd: 25,
      meanBR: 21,
      stdBR: 1.2,
      meanTemp: 34,
      stdTemp: 0.05,
      meanAccMag: 0.95,
      stdAccMag: 0.02,
      isWorn: true,
    );
    const elevatedTemperature = ChestStrapReading(
      timestamp: 2,
      meanHR: 95,
      meanRR: 631.6,
      sdnn: 55,
      rmssd: 25,
      meanBR: 21,
      stdBR: 1.2,
      meanTemp: 40,
      stdTemp: 2,
      meanAccMag: 0.95,
      stdAccMag: 0.02,
      isWorn: true,
    );

    expect(calmTemperature.meanTemp, isNot(elevatedTemperature.meanTemp));
    expect(calmTemperature.stdTemp, isNot(elevatedTemperature.stdTemp));
    expect(calmTemperature.riskScore, elevatedTemperature.riskScore);
    expect(calmTemperature.riskLabel, elevatedTemperature.riskLabel);
  });
}
