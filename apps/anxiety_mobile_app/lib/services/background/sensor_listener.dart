import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:screen_state/screen_state.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../background_service_helper.dart';

class SensorListener {
  StreamSubscription? _screenSubscription;
  StreamSubscription? _accelSubscription;
  Timer? _movementFlushTimer;

  String? _cachedUserId;

  int _movementSamples = 0;
  int _highMotionSamples = 0;
  double _magnitudeSum = 0;
  double _magnitudeSqSum = 0;
  DateTime? _movementWindowStart;

  void startListening(String userId) {
    _cachedUserId = userId;
    _startScreenListener(userId);
    _startAccelerometerListener(userId);
  }

  void stopListening() {
    _screenSubscription?.cancel();
    _accelSubscription?.cancel();
    _movementFlushTimer?.cancel();
  }

  // ─────────────────────────────────────────────────────────────
  // SCREEN STATE — event driven
  // ─────────────────────────────────────────────────────────────

  void _startScreenListener(String userId) {
    try {
      debugPrint('🔍 SensorListener: Starting Screen State Listener...');
      final screen = Screen();
      _screenSubscription = screen.screenStateStream.listen(
        (ScreenStateEvent event) {
          final String status;
          switch (event) {
            case ScreenStateEvent.SCREEN_ON:
              status = 'Screen_On';
              break;
            case ScreenStateEvent.SCREEN_OFF:
              status = 'Screen_Off';
              break;
            case ScreenStateEvent.SCREEN_UNLOCKED:
              status = 'Screen_Unlocked';
              break;
          }

          debugPrint('📱 Screen Event: $status');
          // The event is persisted to the local offline queue immediately, but
          // network flushing remains batched to avoid one HTTP request per tap.
          _sendData(userId, 'Screen_Event', {'state': status});
        },
        onError: (e) => debugPrint('Screen State Stream Error: $e'),
      );
    } catch (e) {
      debugPrint('Screen State Setup Error: $e');
    }
  }

  // ─────────────────────────────────────────────────────────────
  // MOVEMENT PROXY — five-minute aggregate windows
  // ─────────────────────────────────────────────────────────────

  void _startAccelerometerListener(String userId) {
    try {
      debugPrint('🔍 SensorListener: Starting Accelerometer Listener...');
      _movementWindowStart = DateTime.now();

      _accelSubscription = accelerometerEventStream().listen(
        (AccelerometerEvent event) {
          final magnitude = sqrt(
            event.x * event.x + event.y * event.y + event.z * event.z,
          );
          _movementSamples++;
          _magnitudeSum += magnitude;
          _magnitudeSqSum += magnitude * magnitude;
          if (magnitude > 15.0) _highMotionSamples++;
        },
        onError: (e) => debugPrint('Accelerometer Stream Error: $e'),
      );

      _movementFlushTimer?.cancel();
      _movementFlushTimer = Timer.periodic(
        const Duration(minutes: 5),
        (_) => _flushMovementWindow(userId),
      );
    } catch (e) {
      debugPrint('Sensor Setup Error: $e');
    }
  }

  Future<void> _flushMovementWindow(String userId) async {
    final samples = _movementSamples;
    final start = _movementWindowStart ?? DateTime.now();
    final end = DateTime.now();

    if (samples > 0) {
      final mean = _magnitudeSum / samples;
      final variance = max(0.0, (_magnitudeSqSum / samples) - mean * mean);
      final std = sqrt(variance);

      await _sendData(userId, 'Movement_Window_5m', {
        'window_start': start.toIso8601String(),
        'window_end': end.toIso8601String(),
        'sample_count': samples,
        'mean_magnitude': double.parse(mean.toStringAsFixed(3)),
        'std_magnitude': double.parse(std.toStringAsFixed(3)),
        'high_motion_fraction': double.parse(
          (_highMotionSamples / samples).toStringAsFixed(4),
        ),
      });
    }

    _movementSamples = 0;
    _highMotionSamples = 0;
    _magnitudeSum = 0;
    _magnitudeSqSum = 0;
    _movementWindowStart = end;
  }

  Future<void> _sendData(
    String userId,
    String dataType,
    dynamic value,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload();
      final currentId = prefs.getString('user_id') ?? _cachedUserId ?? userId;
      await BackgroundServiceHelper.enqueueResearchEvent(
        currentId,
        dataType,
        value,
      );
    } catch (e) {
      debugPrint('SensorListener send error [$dataType]: $e');
    }
  }
}
