import 'dart:async';
import 'api_service.dart';
import 'chest_strap_service.dart';

class SensorManager {
  final String userId;
  final int samplingRate;

  // This in-memory list stores the live features from the chest strap
  final List<ChestStrapReading> _readingsBuffer = [];

  Timer? _oneMinuteTimer;
  bool isCollecting = false;
  bool _isFlushing = false;

  SensorManager({required this.userId, this.samplingRate = 1});

  // START HDS COLLECTION
  void startCollection() {
    if (isCollecting) return;

    isCollecting = true;
    _readingsBuffer.clear();

    // This periodic timer fires exactly every 60 seconds
    _oneMinuteTimer = Timer.periodic(
      const Duration(seconds: 60),
      (_) => _flushMinuteWindow(),
    );
  }

  Future<void> _flushMinuteWindow() async {
    if (_isFlushing) {
      print('Minute upload is still running; keeping new packets buffered.');
      return;
    }
    if (_readingsBuffer.isEmpty) {
      print('Buffer warning: No feature data accumulated in the last minute.');
      return;
    }

    _isFlushing = true;
    final readingsToSend = List<ChestStrapReading>.from(_readingsBuffer);
    _readingsBuffer.clear();

    try {
      print(
        '60 seconds up! Averaging ${readingsToSend.length} feature samples for the server...',
      );

      final wornReadings = readingsToSend.where((r) => r.isWorn).toList();
      final isWorn = wornReadings.length > (readingsToSend.length / 2);

      // If the strap was worn for most of the minute, exclude off-body zero
      // packets from the averages. Otherwise send an explicit not-worn block
      // so the server's data-quality guard can reject it as designed.
      final readingsForAverages = isWorn ? wornReadings : readingsToSend;

      double sumHr = 0, sumRr = 0, sumSdnn = 0, sumRmssd = 0;
      double sumBr = 0, sumStdBr = 0, sumTemp = 0, sumStdTemp = 0;
      double sumAccMag = 0, sumStdAccMag = 0;

      for (var r in readingsForAverages) {
        sumHr += r.meanHR;
        sumRr += r.meanRR;
        sumSdnn += r.sdnn;
        sumRmssd += r.rmssd;
        sumBr += r.meanBR;
        sumStdBr += r.stdBR;
        sumTemp += r.meanTemp;
        sumStdTemp += r.stdTemp;
        sumAccMag += r.meanAccMag;
        sumStdAccMag += r.stdAccMag;
      }

      final count = readingsForAverages.length;

      final success = await ApiService.sendFeatureData(
        userId: userId,
        isWorn: isWorn,
        meanHr: sumHr / count,
        meanRr: sumRr / count,
        sdnn: sumSdnn / count,
        rmssd: sumRmssd / count,
        meanBr: sumBr / count,
        stdBr: sumStdBr / count,
        meanTemp: sumTemp / count,
        stdTemp: sumStdTemp / count,
        meanAccMag: sumAccMag / count,
        stdAccMag: sumStdAccMag / count,
      );

      if (!success) {
        print('Failed to transmit this minute feature block.');
        if (isWorn && isCollecting) {
          // Preserve a worn block after a network/server failure instead of
          // silently losing it. Cap the retry buffer at five minutes.
          _readingsBuffer.insertAll(0, readingsToSend);
          final maxBufferedReadings = samplingRate * 60 * 5;
          if (_readingsBuffer.length > maxBufferedReadings) {
            _readingsBuffer.removeRange(
              0,
              _readingsBuffer.length - maxBufferedReadings,
            );
          }
        }
      }
    } finally {
      _isFlushing = false;
    }
  }

  // CLEAN UP
  void stopCollection() {
    isCollecting = false;
    _oneMinuteTimer?.cancel();
    _readingsBuffer.clear();
  }

  void addLiveChestStrapData(ChestStrapReading reading) {
    if (!isCollecting) return;
    _readingsBuffer.add(reading);
  }

  // FOR BACKWARDS COMPATIBILITY IF NEEDED (e.g. from BLE Bridge if not updated)
  @Deprecated('Use addLiveChestStrapData with ChestStrapReading instead')
  void addLiveData(
    double ecg,
    double accX,
    double accY,
    double accZ,
    double temp,
  ) {
    // This is no longer used but kept to avoid breaking compilation
    if (!isCollecting) return;
    final r = ChestStrapReading(
      timestamp: DateTime.now().millisecondsSinceEpoch,
      meanHR: ecg,
      meanRR: 0,
      sdnn: 0,
      rmssd: 0,
      meanBR: 0,
      stdBR: 0,
      meanTemp: temp,
      stdTemp: 0,
      meanAccMag: accX,
      stdAccMag: accY,
      isWorn: true,
    );
    _readingsBuffer.add(r);
  }
}
