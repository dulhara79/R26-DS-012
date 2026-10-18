import 'dart:convert';

import 'package:anxiety_mobile_app/services/chest_strap_service.dart';
import 'package:anxiety_mobile_app/services/model_feature_window.dart';
import 'package:anxiety_mobile_app/services/sensor_manager.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

ModelFeatureWindow minute(int start, {int status = 1, int version = 2}) =>
    ModelFeatureWindow(
      version: version,
      startMilliseconds: start,
      endMilliseconds: start + 60000,
      status: status,
      features: const [60, 1000, 20, 30, 15, 1, 37, .02, .01, .005],
    );

ChestStrapReading packet(int timestamp, ModelFeatureWindow? window) =>
    ChestStrapReading(
      timestamp: timestamp,
      meanHR: 90,
      meanRR: 667,
      sdnn: 400,
      rmssd: 700,
      meanBR: 16,
      stdBR: 7,
      meanTemp: 37,
      stdTemp: .03,
      meanAccMag: .9,
      stdAccMag: .01,
      isWorn: true,
      modelWindow: window,
    );

void main() {
  test(
    'new packets retain live display values and separate model features',
    () {
      final reading = ChestStrapReading.fromCsv(
        '120001,90,667,400,700,16,7,41,.03,.9,.01,1,'
        '2,60000,120000,1,60,1000,20,30,15,1,41,.02,.01,.005',
      );
      expect(reading.meanHR, 90);
      expect(reading.meanAccMag, .9);
      expect(reading.modelWindow!.features[0], 60);
      expect(reading.modelWindow!.features[6], 37);
      expect(reading.modelWindow!.features[8], .01);
      expect(reading.modelWindow!.isReady, isTrue);
      final restored = ChestStrapReading.fromJson(reading.toJson());
      expect(restored.modelWindow!.features, reading.modelWindow!.features);
    },
  );

  test('calibration ignores cached pre-start minutes and duplicates', () {
    final sequence = CalibrationMinuteSequence(65000);
    expect(sequence.accept(minute(0), 66000, true), isFalse);
    expect(sequence.accept(minute(60000), 120000, true), isFalse);
    expect(sequence.accept(minute(120000), 180000, true), isTrue);
    expect(sequence.accept(minute(120000), 181000, true), isFalse);
    expect(sequence.accept(minute(180000), 240000, true), isTrue);
    expect(
      () => sequence.accept(minute(300000), 360000, true),
      throwsStateError,
    );
  });

  test(
    'calibration rejects legacy, incomplete, removed, and restarted streams',
    () {
      for (final invalid in [
        null,
        minute(120000, version: 1),
        minute(120000, status: 3),
      ]) {
        expect(
          () => CalibrationMinuteSequence(65000).accept(invalid, 180000, true),
          throwsStateError,
        );
      }
      expect(
        () => CalibrationMinuteSequence(
          65000,
        ).accept(minute(120000), 180000, false),
        throwsStateError,
      );
      expect(
        () =>
            CalibrationMinuteSequence(65000).accept(minute(120000), 1000, true),
        throwsStateError,
      );
    },
  );

  test(
    'uploads complete model minutes once without averaging live display values',
    () async {
      final requests = <Map<String, dynamic>>[];
      await http.runWithClient(
        () async {
          final manager = SensorManager(userId: 'test_participant');
          manager.startCollection();
          await manager.addLiveChestStrapData(packet(120000, minute(60000)));
          await manager.addLiveChestStrapData(packet(121000, minute(60000)));
          await manager.addLiveChestStrapData(packet(180000, minute(120000)));
          await manager.addLiveChestStrapData(packet(181000, null));
          await manager.addLiveChestStrapData(
            packet(182000, minute(120000, status: 3)),
          );
          manager.stopCollection();
        },
        () => MockClient((request) async {
          requests.add(jsonDecode(request.body) as Map<String, dynamic>);
          return http.Response('{"status":"success"}', 200);
        }),
      );
      expect(requests.length, 2);
      expect(requests.first['feature_schema_version'], 2);
      expect(requests.first['mean_hr'], 60);
      expect(requests.first['sdnn'], 20);
      expect(requests.first['mean_acc_mag'], .01);
    },
  );
}
