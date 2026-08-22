import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum ChestStrapState { disabled, scanning, connecting, connected, disconnected }

class ChestStrapReading {
  final int timestamp;
  final double meanHR;
  final double meanRR;
  final double sdnn;
  final double rmssd;
  final double meanBR;
  final double stdBR;
  final double meanTemp;
  final double stdTemp;
  final double meanAccMag;
  final double stdAccMag;
  final bool isWorn;

  const ChestStrapReading({
    required this.timestamp,
    required this.meanHR,
    required this.meanRR,
    required this.sdnn,
    required this.rmssd,
    required this.meanBR,
    required this.stdBR,
    required this.meanTemp,
    required this.stdTemp,
    required this.meanAccMag,
    required this.stdAccMag,
    required this.isWorn,
  });

  factory ChestStrapReading.fromCsv(String csvLine) {
    final parts = csvLine.split(',');
    if (parts.length != 12) {
      throw FormatException(
        'Invalid CSV length: expected 12, got ${parts.length}',
      );
    }
    final rawIsWorn = parts[11].trim();
    return ChestStrapReading(
      timestamp: int.parse(parts[0].trim()),
      meanHR: double.parse(parts[1].trim()),
      meanRR: double.parse(parts[2].trim()),
      sdnn: double.parse(parts[3].trim()),
      rmssd: double.parse(parts[4].trim()),
      meanBR: double.parse(parts[5].trim()),
      stdBR: double.parse(parts[6].trim()),
      meanTemp: double.parse(parts[7].trim()),
      stdTemp: double.parse(parts[8].trim()),
      meanAccMag: double.parse(parts[9].trim()),
      stdAccMag: double.parse(parts[10].trim()),
      isWorn:
          rawIsWorn == '1' ||
          rawIsWorn == '1.0' ||
          rawIsWorn.toLowerCase() == 'true',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'timestamp': timestamp,
      'meanHR': meanHR,
      'meanRR': meanRR,
      'sdnn': sdnn,
      'rmssd': rmssd,
      'meanBR': meanBR,
      'stdBR': stdBR,
      'meanTemp': meanTemp,
      'stdTemp': stdTemp,
      'meanAccMag': meanAccMag,
      'stdAccMag': stdAccMag,
      'isWorn': isWorn,
    };
  }

  factory ChestStrapReading.fromJson(Map<String, dynamic> json) {
    final double meanHR = (json['meanHR'] as num).toDouble();
    final double meanTemp = (json['meanTemp'] as num).toDouble();
    return ChestStrapReading(
      timestamp: json['timestamp'] as int,
      meanHR: meanHR,
      meanRR: (json['meanRR'] as num).toDouble(),
      sdnn: (json['sdnn'] as num).toDouble(),
      rmssd: (json['rmssd'] as num).toDouble(),
      meanBR: (json['meanBR'] as num).toDouble(),
      stdBR: (json['stdBR'] as num).toDouble(),
      meanTemp: meanTemp,
      stdTemp: (json['stdTemp'] as num).toDouble(),
      meanAccMag: (json['meanAccMag'] as num).toDouble(),
      stdAccMag: (json['stdAccMag'] as num).toDouble(),
      isWorn: json['isWorn'] as bool? ?? (meanHR >= 30.0 && meanTemp >= 30.0),
    );
  }

  double get riskScore {
    if (!isWorn) return 0.0;

    double hrScore = 0.0;
    if (meanHR > 110) {
      hrScore = 100.0;
    } else if (meanHR > 90) {
      hrScore = 40.0 + (meanHR - 90) / (110 - 90) * (80 - 40);
    } else if (meanHR > 70) {
      hrScore = (meanHR - 70) / (90 - 70) * 40.0;
    }

    double brScore = 0.0;
    if (meanBR > 26) {
      brScore = 100.0;
    } else if (meanBR > 20) {
      brScore = 40.0 + (meanBR - 20) / (26 - 20) * (80 - 40);
    } else if (meanBR > 16) {
      brScore = (meanBR - 16) / (20 - 16) * 40.0;
    }

    double tempScore = 0.0;
    double tempDeviation = (meanTemp - 36.75).abs();
    if (tempDeviation > 0.6) {
      tempScore = 100.0;
    } else if (tempDeviation > 0.3) {
      tempScore = (tempDeviation - 0.3) / (0.6 - 0.3) * 50.0 + 50.0;
    }

    double hrvScore = 0.0;
    if (rmssd >= 40) {
      hrvScore = 0.0;
    } else if (rmssd >= 20) {
      hrvScore = (40.0 - rmssd) / 20.0 * 50.0;
    } else {
      hrvScore = 50.0 + (20.0 - rmssd) / 20.0 * 50.0;
      if (hrvScore > 100.0) hrvScore = 100.0;
    }

    double total =
        (hrScore * 0.35) +
        (brScore * 0.25) +
        (tempScore * 0.15) +
        (hrvScore * 0.25);
    return total.clamp(0.0, 100.0);
  }

  String get riskLabel {
    if (!isWorn) return 'Not Worn';
    final score = riskScore;
    if (score <= 20) return 'Low';
    if (score <= 45) return 'Moderate';
    if (score <= 70) return 'Elevated';
    return 'High';
  }

  String get hrStatus {
    if (!isWorn) return 'Not Worn';
    if (meanHR <= 60) return 'Low';
    if (meanHR <= 90) return 'Normal';
    if (meanHR <= 110) return 'Elevated';
    return 'High';
  }

  String get brStatus {
    if (!isWorn) return 'Not Worn';
    if (meanBR <= 12) return 'Low';
    if (meanBR <= 20) return 'Normal';
    if (meanBR <= 26) return 'Elevated';
    return 'High';
  }

  String get tempStatus {
    if (!isWorn) return 'Not Worn';
    if (meanTemp < 36.1) return 'Low';
    if (meanTemp <= 37.2) return 'Normal';
    if (meanTemp <= 37.8) return 'Elevated';
    return 'High';
  }

  String get hrvStatus {
    if (!isWorn) return 'Not Worn';
    if (rmssd >= 40) return 'Calm';
    if (rmssd >= 25) return 'Normal';
    if (rmssd >= 15) return 'Stressed';
    return 'High Stress';
  }

  String get motionStatus {
    if (!isWorn) return 'Not Worn';
    if (stdAccMag <= 0.03) return 'Still';
    if (stdAccMag <= 0.12) return 'Light';
    if (stdAccMag <= 0.30) return 'Active';
    return 'High';
  }
}

class ChestStrapService {
  static final ChestStrapService _instance = ChestStrapService._internal();

  factory ChestStrapService() => _instance;

  final ValueNotifier<ChestStrapState> connectionState = ValueNotifier(
    ChestStrapState.disconnected,
  );

  // Broadcast stream for real-time readings
  final StreamController<ChestStrapReading> _readingsController =
      StreamController<ChestStrapReading>.broadcast();
  Stream<ChestStrapReading> get readingsStream => _readingsController.stream;

  @Deprecated('Use readingsStream instead to support multiple listeners')
  Function(ChestStrapReading)? onDataReceived;
  ChestStrapReading? lastReading;

  BluetoothDevice? _connectedDevice;
  StreamSubscription? _scanSubscription;
  StreamSubscription? _connectionSubscription;
  StreamSubscription? _txSubscription;
  String _receiveBuffer = '';
  int _reconnectAttempts = 0;
  bool _manualDisconnect = false;
  static const int _maxReconnectAttempts = 5;

  Timer? _simulationTimer;
  final Random _simulationRandom = Random();
  final ValueNotifier<bool> simulationEnabled = ValueNotifier(false);
  final ValueNotifier<bool> simulatedIsWorn = ValueNotifier(true);
  final ValueNotifier<bool> simulatedStressIncreasing = ValueNotifier(false);
  DateTime? _stressSimulationStartedAt;
  double _stressRampStartLevel = 0.0;
  DateTime? _stressRecoveryStartedAt;
  double _stressRecoveryStartLevel = 0.0;

  static const Duration _stressRampDuration = Duration(minutes: 5);
  static const Duration _liveReadingTimeout = Duration(seconds: 5);
  final ValueNotifier<bool> liveReadingAvailable = ValueNotifier(false);
  Timer? _readingExpiryTimer;

  static const String _nusServiceUuid = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E';
  static const String _nusTxUuid = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E';
  static const String _prefKey = 'chest_strap_last_reading';

  ChestStrapService._internal() {
    _loadPersistedReading();
  }

  Future<void> _loadPersistedReading() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonStr = prefs.getString(_prefKey);
      if (jsonStr != null) {
        lastReading = ChestStrapReading.fromJson(jsonDecode(jsonStr));
      }
    } catch (e) {
      debugPrint('Error loading persisted reading: $e');
    }
  }

  Future<void> _saveReading(ChestStrapReading reading) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefKey, jsonEncode(reading.toJson()));
    } catch (e) {
      debugPrint('Error saving reading: $e');
    }
  }

  bool get isConnected => connectionState.value == ChestStrapState.connected;
  bool get hasLiveWornReading =>
      isConnected &&
      liveReadingAvailable.value &&
      (lastReading?.isWorn ?? false);

  double get simulatedStressProgress {
    if (simulatedStressIncreasing.value &&
        _stressSimulationStartedAt != null) {
      final elapsed = DateTime.now().difference(_stressSimulationStartedAt!);
      return simulationStressLevelForElapsed(
        startLevel: _stressRampStartLevel,
        increasing: true,
        elapsed: elapsed,
      );
    }

    if (_stressRecoveryStartedAt != null) {
      final elapsed = DateTime.now().difference(_stressRecoveryStartedAt!);
      final level = simulationStressLevelForElapsed(
        startLevel: _stressRecoveryStartLevel,
        increasing: false,
        elapsed: elapsed,
      );
      if (level <= 0.0) {
        _stressRecoveryStartedAt = null;
        _stressRecoveryStartLevel = 0.0;
      }
      return level;
    }

    return 0.0;
  }

  @visibleForTesting
  double simulationStressLevelForElapsed({
    required double startLevel,
    required bool increasing,
    required Duration elapsed,
  }) {
    final linear =
        (elapsed.inMilliseconds / _stressRampDuration.inMilliseconds)
            .clamp(0.0, 1.0)
            .toDouble();
    final curved = pow(linear, 1.25).toDouble();
    final start = startLevel.clamp(0.0, 1.0).toDouble();
    return increasing
        ? start + (1.0 - start) * curved
        : start * (1.0 - curved);
  }

  /// Starts a phone-side physiological simulator. This never changes or
  /// depends on the chest-strap firmware. Simulated packets use the exact
  /// 12-field feature contract produced by ChestStrap_V3 and are published
  /// through the same stream as real BLE packets.
  Future<void> startSimulation({bool isWorn = true}) async {
    await disconnect();

    simulationEnabled.value = true;
    simulatedIsWorn.value = isWorn;
    simulatedStressIncreasing.value = false;
    _stressSimulationStartedAt = null;
    _stressRampStartLevel = 0.0;
    _stressRecoveryStartedAt = null;
    _stressRecoveryStartLevel = 0.0;
    connectionState.value = ChestStrapState.connected;

    _emitSimulatedReading();
    _simulationTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) => _emitSimulatedReading(),
    );
  }

  /// Changes the simulated contact state without restarting the simulator.
  /// Off-body values intentionally become zero, matching HealthEngine.cpp.
  void setSimulationWorn(bool isWorn) {
    if (!simulationEnabled.value) return;
    simulatedIsWorn.value = isWorn;
    if (!isWorn) {
      simulatedStressIncreasing.value = false;
      _stressSimulationStartedAt = null;
      _stressRampStartLevel = 0.0;
      _stressRecoveryStartedAt = null;
      _stressRecoveryStartLevel = 0.0;
    }
    _emitSimulatedReading();
  }

  /// Starts or stops a five-minute progressive stress simulation. The stream
  /// starts calm, then raises heart and breathing rate while lowering HRV.
  void setSimulationStress(bool stressIncreasing) {
    if (!simulationEnabled.value || !simulatedIsWorn.value) return;

    final currentLevel = simulatedStressProgress;
    simulatedStressIncreasing.value = stressIncreasing;
    if (stressIncreasing) {
      _stressRampStartLevel = currentLevel;
      _stressSimulationStartedAt = DateTime.now();
      _stressRecoveryStartedAt = null;
      _stressRecoveryStartLevel = 0.0;
    } else {
      _stressSimulationStartedAt = null;
      _stressRampStartLevel = 0.0;
      _stressRecoveryStartLevel = currentLevel;
      _stressRecoveryStartedAt = currentLevel > 0.0 ? DateTime.now() : null;
    }
    _emitSimulatedReading();
  }

  Future<void> stopSimulation() async {
    _simulationTimer?.cancel();
    _simulationTimer = null;
    simulationEnabled.value = false;
    simulatedStressIncreasing.value = false;
    _stressSimulationStartedAt = null;
    _stressRampStartLevel = 0.0;
    _stressRecoveryStartedAt = null;
    _stressRecoveryStartLevel = 0.0;
    lastReading = null;
    liveReadingAvailable.value = false;
    _readingExpiryTimer?.cancel();
    connectionState.value = ChestStrapState.disconnected;
  }

  double _jitter(double amplitude) {
    return (_simulationRandom.nextDouble() * 2.0 - 1.0) * amplitude;
  }

  void _emitSimulatedReading() {
    if (!simulationEnabled.value) return;

    final worn = simulatedIsWorn.value;
    final reading = _buildSimulatedReading(
      timestamp: DateTime.now().millisecondsSinceEpoch,
      isWorn: worn,
      stressLevel: simulatedStressProgress,
      includeJitter: true,
    );

    // Do not persist test data across app launches.
    _publishReading(reading, persist: false);
  }

  double _lerp(double calm, double stressed, double stressLevel) {
    return calm + (stressed - calm) * stressLevel;
  }

  ChestStrapReading _buildSimulatedReading({
    required int timestamp,
    required bool isWorn,
    required double stressLevel,
    required bool includeJitter,
  }) {
    if (!isWorn) {
      return ChestStrapReading(
        timestamp: timestamp,
        meanHR: 0.0,
        meanRR: 0.0,
        sdnn: 0.0,
        rmssd: 0.0,
        meanBR: 0.0,
        stdBR: 0.0,
        meanTemp: 0.0,
        stdTemp: 0.0,
        meanAccMag: 0.0,
        stdAccMag: 0.0,
        isWorn: false,
      );
    }

    final level = stressLevel.clamp(0.0, 1.0).toDouble();
    final jitterScale = includeJitter ? 1.0 + level * 0.8 : 0.0;
    final hr = _lerp(72.0, 150.0, level) + _jitter(2.5 * jitterScale);

    return ChestStrapReading(
      timestamp: timestamp,
      meanHR: hr,
      meanRR: 60000.0 / hr,
      sdnn: _lerp(46.0, 12.0, level) + _jitter(3.0 * jitterScale),
      rmssd: _lerp(43.0, 7.0, level) + _jitter(3.0 * jitterScale),
      meanBR: _lerp(15.5, 40.0, level) + _jitter(0.8 * jitterScale),
      stdBR: _lerp(0.55, 4.75, level) + _jitter(0.12 * jitterScale),
      meanTemp: _lerp(36.60, 37.50, level) + _jitter(0.05 * jitterScale),
      stdTemp: _lerp(0.04, 0.22, level) + _jitter(0.01 * jitterScale),
      meanAccMag: _lerp(1.0, 1.12, level) + _jitter(0.02 * jitterScale),
      stdAccMag: _lerp(0.018, 0.358, level) + _jitter(0.008 * jitterScale),
      isWorn: true,
    );
  }

  @visibleForTesting
  ChestStrapReading buildSimulatedReadingForTest(double stressLevel) {
    return _buildSimulatedReading(
      timestamp: 1,
      isWorn: true,
      stressLevel: stressLevel,
      includeJitter: false,
    );
  }

  Future<void> startScan() async {
    if (simulationEnabled.value) {
      await stopSimulation();
    }

    final scanCompleter = Completer<void>();

    try {
      debugPrint('[ChestStrap] Checking Bluetooth adapter state...');
      if (await FlutterBluePlus.adapterState.first ==
          BluetoothAdapterState.off) {
        debugPrint('[ChestStrap] Bluetooth adapter is OFF');
        connectionState.value = ChestStrapState.disabled;
        return;
      }

      connectionState.value = ChestStrapState.scanning;

      // ── Step 1: Check bonded (paired) devices first ──
      // Bonded devices often stop advertising, so a BLE scan won't find them.
      // Connect directly if ChestStrap_V3 is already paired.
      try {
        final bondedDevices = await FlutterBluePlus.bondedDevices;
        debugPrint(
          '[ChestStrap] Found ${bondedDevices.length} bonded device(s)',
        );
        for (var device in bondedDevices) {
          final name = device.platformName.isNotEmpty
              ? device.platformName
              : device.advName;
          debugPrint(
            '[ChestStrap] Bonded device: "$name" (${device.remoteId})',
          );
          if (name.contains('ChestStrap_V3')) {
            debugPrint(
              '[ChestStrap] ✅ ChestStrap_V3 found in bonded devices! Connecting directly...',
            );
            await connectToDevice(device);
            return; // Done — no scan needed
          }
        }
        debugPrint(
          '[ChestStrap] ChestStrap_V3 not in bonded list, falling back to BLE scan...',
        );
      } catch (e) {
        debugPrint(
          '[ChestStrap] Could not check bonded devices: $e — falling back to scan',
        );
      }

      // ── Step 2: Fall back to BLE scan ──
      debugPrint('[ChestStrap] Scanning for ChestStrap_V3...');

      bool deviceFound = false;

      // Listen for scan results BEFORE starting scan to avoid race condition
      _scanSubscription?.cancel();
      _scanSubscription = FlutterBluePlus.onScanResults.listen(
        (results) async {
          if (deviceFound) return; // Already found, ignore further results
          try {
            ScanResult? targetResult;
            for (var result in results) {
              // Check all possible name fields – advName can be empty on
              // many Android 12+ devices, so also check platformName and
              // the advertisement data's localName.
              final advName = result.device.advName;
              final platformName = result.device.platformName;
              final advDataName = result.advertisementData.advName;

              final names = [advName, platformName, advDataName];
              debugPrint(
                '[ChestStrap] Found device: advName="$advName" '
                'platformName="$platformName" advDataName="$advDataName" '
                '(${result.device.remoteId})',
              );

              if (names.any((n) => n.contains('ChestStrap_V3'))) {
                targetResult = result;
                break;
              }
            }

            if (targetResult != null) {
              deviceFound = true;
              debugPrint('[ChestStrap] Target device found! Stopping scan...');
              await FlutterBluePlus.stopScan();
              await connectToDevice(targetResult.device);
              if (!scanCompleter.isCompleted) scanCompleter.complete();
            }
          } catch (e) {
            debugPrint('[ChestStrap] Error processing scan results: $e');
            if (!scanCompleter.isCompleted) scanCompleter.complete();
          }
        },
        onError: (e) {
          debugPrint('[ChestStrap] Scan error: $e');
          connectionState.value = ChestStrapState.disconnected;
          if (!scanCompleter.isCompleted) scanCompleter.complete();
        },
      );

      // startScan() itself completes when the timeout expires or stopScan() is
      // called — so we await it to know the scan window has closed.
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 15));

      // If we reach here without having found the device, the scan timed out.
      if (!deviceFound && !scanCompleter.isCompleted) {
        connectionState.value = ChestStrapState.disconnected;
        scanCompleter.complete();
      }
    } catch (e) {
      debugPrint('[ChestStrap] Error starting scan: $e');
      connectionState.value = ChestStrapState.disconnected;
      if (!scanCompleter.isCompleted) scanCompleter.complete();
    }

    // Wait until scanning is truly done (device found or timeout).
    return scanCompleter.future;
  }

  Future<void> connectToDevice(BluetoothDevice device) async {
    try {
      if (simulationEnabled.value) {
        await stopSimulation();
      }
      connectionState.value = ChestStrapState.connecting;
      _manualDisconnect = false;
      _connectedDevice = device;
      lastReading = null;
      liveReadingAvailable.value = false;
      _readingExpiryTimer?.cancel();
      _receiveBuffer = ''; // Clear stale buffer from previous connection

      debugPrint(
        '[ChestStrap] Connecting to ${device.advName} (${device.remoteId})...',
      );
      await _connectedDevice!.connect(timeout: const Duration(seconds: 10));

      _reconnectAttempts = 0;
      debugPrint('[ChestStrap] Connected! Requesting MTU 256...');

      // Request larger MTU so the ~90-byte CSV arrives in 1-2 packets
      // instead of 4-5 with the default 23-byte MTU.
      try {
        final mtu = await _connectedDevice!.requestMtu(256);
        debugPrint('[ChestStrap] MTU negotiated: $mtu');
      } catch (e) {
        // The firmware already sends 20-byte chunks, so MTU negotiation is
        // only an optimisation and must not make an otherwise valid BLE
        // connection fail (notably on iOS).
        debugPrint('[ChestStrap] MTU request skipped/failed safely: $e');
      }

      _connectionSubscription?.cancel();
      _connectionSubscription = _connectedDevice!.connectionState.listen(
        _onConnectionStateChanged,
      );

      debugPrint('[ChestStrap] Discovering services...');
      final services = await _connectedDevice!.discoverServices();
      debugPrint('[ChestStrap] Found ${services.length} services');

      BluetoothService? nusService;
      for (var s in services) {
        if (s.uuid.toString().toUpperCase() == _nusServiceUuid.toUpperCase()) {
          nusService = s;
          break;
        }
      }

      if (nusService != null) {
        debugPrint(
          '[ChestStrap] NUS service found. Looking for TX characteristic...',
        );
        BluetoothCharacteristic? txChar;
        for (var c in nusService.characteristics) {
          if (c.uuid.toString().toUpperCase() == _nusTxUuid.toUpperCase()) {
            txChar = c;
            break;
          }
        }

        if (txChar != null) {
          debugPrint(
            '[ChestStrap] TX characteristic found. Subscribing to notifications...',
          );
          await txChar.setNotifyValue(true);
          _txSubscription?.cancel();
          _txSubscription = txChar.lastValueStream.listen((value) {
            if (value.isNotEmpty) {
              final str = utf8.decode(value);
              _receiveBuffer += str;
              _processBuffer();
            }
          });
          connectionState.value = ChestStrapState.connected;
          debugPrint('[ChestStrap] ✅ Fully connected and listening for data.');
        } else {
          debugPrint(
            '[ChestStrap] ⚠️ TX characteristic NOT found in NUS service!',
          );
          await disconnect();
        }
      } else {
        debugPrint(
          '[ChestStrap] ⚠️ NUS service NOT found! Available services:',
        );
        for (var s in services) {
          debugPrint('[ChestStrap]   - ${s.uuid}');
        }
        await disconnect();
      }
    } catch (e) {
      debugPrint('[ChestStrap] Error connecting to device: $e');
      connectionState.value = ChestStrapState.disconnected;
    }
  }

  void _processBuffer() {
    // Primary path: newline-delimited parsing (firmware sends \n)
    int newlineIndex = _receiveBuffer.indexOf('\n');
    while (newlineIndex != -1) {
      String line = _receiveBuffer.substring(0, newlineIndex).trim();
      _receiveBuffer = _receiveBuffer.substring(newlineIndex + 1);

      if (line.isNotEmpty) {
        _parseCsvLine(line);
      }

      newlineIndex = _receiveBuffer.indexOf('\n');
    }

    // Fallback: if no newline found but buffer has a complete 12-field CSV
    // (11 commas), extract and parse it. Guards against firmware versions
    // that don't append \n.
    if (_receiveBuffer.isNotEmpty && !_receiveBuffer.contains('\n')) {
      final commaCount = ','.allMatches(_receiveBuffer).length;
      if (commaCount >= 11) {
        // Find the end of the first complete 12-field CSV line
        int count = 0;
        int endIndex = -1;
        for (int i = 0; i < _receiveBuffer.length; i++) {
          if (_receiveBuffer[i] == ',') count++;
          if (count == 11) {
            // Find the end of the 12th field (next comma or end of string)
            int fieldEnd = _receiveBuffer.indexOf(',', i + 1);
            endIndex = fieldEnd == -1 ? _receiveBuffer.length : fieldEnd;
            break;
          }
        }
        if (endIndex != -1) {
          String line = _receiveBuffer.substring(0, endIndex).trim();
          _receiveBuffer = _receiveBuffer.substring(endIndex).trimLeft();
          if (line.isNotEmpty) {
            debugPrint(
              '[ChestStrap] Fallback parser extracted line (no newline in buffer)',
            );
            _parseCsvLine(line);
          }
        }
      }
    }

    // Safety: prevent unbounded buffer growth from corrupt data
    if (_receiveBuffer.length > 1024) {
      debugPrint(
        '[ChestStrap] ⚠️ Receive buffer overflow (${_receiveBuffer.length} bytes). Clearing.',
      );
      _receiveBuffer = '';
    }
  }

  void _parseCsvLine(String line) {
    try {
      final reading = ChestStrapReading.fromCsv(line);
      _publishReading(reading);
    } catch (e) {
      debugPrint('[ChestStrap] Error parsing CSV "$line": $e');
    }
  }

  void _publishReading(ChestStrapReading reading, {bool persist = true}) {
    lastReading = reading;
    liveReadingAvailable.value = reading.isWorn;
    _readingExpiryTimer?.cancel();
    final publishedTimestamp = reading.timestamp;
    _readingExpiryTimer = Timer(_liveReadingTimeout, () {
      if (lastReading?.timestamp == publishedTimestamp) {
        lastReading = null;
        liveReadingAvailable.value = false;
      }
    });
    if (persist) {
      _saveReading(reading);
    }
    debugPrint(
      '[ChestStrap] 📊 HR=${reading.meanHR.toStringAsFixed(1)} '
      'BR=${reading.meanBR.toStringAsFixed(1)} '
      'Temp=${reading.meanTemp.toStringAsFixed(1)} '
      'RMSSD=${reading.rmssd.toStringAsFixed(1)} '
      'worn=${reading.isWorn} '
      'source=${simulationEnabled.value ? "simulation" : "ble"}',
    );

    _readingsController.add(reading);
    onDataReceived?.call(reading);
  }

  void _onConnectionStateChanged(BluetoothConnectionState state) {
    debugPrint('[ChestStrap] Connection state changed: $state');
    if (state == BluetoothConnectionState.disconnected) {
      connectionState.value = ChestStrapState.disconnected;
      lastReading = null;
      _readingExpiryTimer?.cancel();
      liveReadingAvailable.value = false;
      _txSubscription?.cancel();
      _connectionSubscription?.cancel();

      if (!_manualDisconnect &&
          _reconnectAttempts < _maxReconnectAttempts &&
          _connectedDevice != null) {
        _reconnectAttempts++;
        debugPrint(
          '[ChestStrap] Reconnect attempt $_reconnectAttempts/$_maxReconnectAttempts in 3s...',
        );
        Future.delayed(const Duration(seconds: 3), () {
          if (_connectedDevice != null) {
            connectToDevice(_connectedDevice!);
          }
        });
      } else if (_reconnectAttempts >= _maxReconnectAttempts) {
        debugPrint(
          '[ChestStrap] ⚠️ Max reconnect attempts reached. Giving up.',
        );
      }
    }
  }

  Future<void> disconnect() async {
    try {
      _manualDisconnect = true;
      _simulationTimer?.cancel();
      _simulationTimer = null;
      simulationEnabled.value = false;
      simulatedStressIncreasing.value = false;
      _stressSimulationStartedAt = null;
      _stressRampStartLevel = 0.0;
      _stressRecoveryStartedAt = null;
      _stressRecoveryStartLevel = 0.0;
      _readingExpiryTimer?.cancel();
      liveReadingAvailable.value = false;
      _scanSubscription?.cancel();
      _connectionSubscription?.cancel();
      _txSubscription?.cancel();

      if (_connectedDevice != null) {
        await _connectedDevice!.disconnect();
        _connectedDevice = null;
      }
      lastReading = null;
      connectionState.value = ChestStrapState.disconnected;
    } catch (e) {
      debugPrint('Error disconnecting: $e');
    }
  }
}
