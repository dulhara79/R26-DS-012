import 'api_service.dart';
import 'chest_strap_service.dart';

class SensorManager {
  final String userId;
  final int samplingRate;
  bool isCollecting = false;
  int? _lastWindowEnd;
  int? _lastPacketTimestamp;

  SensorManager({required this.userId, this.samplingRate = 1});

  void startCollection() {
    isCollecting = true;
    _lastWindowEnd = null;
    _lastPacketTimestamp = null;
  }

  void stopCollection() {
    isCollecting = false;
    _lastWindowEnd = null;
    _lastPacketTimestamp = null;
  }

  Future<void> addLiveChestStrapData(ChestStrapReading reading) async {
    if (!isCollecting || !reading.isWorn) return;
    if (_lastPacketTimestamp != null &&
        reading.timestamp < _lastPacketTimestamp!) {
      // Firmware restarted (or uptime wrapped); never reuse the prior block ID.
      _lastWindowEnd = null;
    }
    _lastPacketTimestamp = reading.timestamp;
    final window = reading.modelWindow;
    if (window == null ||
        !window.isReady ||
        window.endMilliseconds == _lastWindowEnd) {
      return;
    }
    final age = window.ageAt(reading.timestamp);
    if (age > 90000) return;
    _lastWindowEnd = window.endMilliseconds;
    final values = window.features;
    // The firmware already computed one real minute. Averaging its SD/RMSSD
    // again changes their definitions and must never be done here.
    await ApiService.sendFeatureData(
      userId: userId,
      isWorn: true,
      meanHr: values[0],
      meanRr: values[1],
      sdnn: values[2],
      rmssd: values[3],
      meanBr: values[4],
      stdBr: values[5],
      meanTemp: values[6],
      stdTemp: values[7],
      meanAccMag: values[8],
      stdAccMag: values[9],
      featureSchemaVersion: window.version,
      observedAt: DateTime.now().subtract(Duration(milliseconds: age)),
    );
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
    addLiveChestStrapData(r);
  }
}
