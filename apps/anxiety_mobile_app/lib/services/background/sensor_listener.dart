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

  String? _cachedUserId;

  void startListening(String userId) {
    _cachedUserId = userId;
    _startScreenListener(userId);
    _startAccelerometerListener(userId);
  }

  void stopListening() {
    _screenSubscription?.cancel();
    _accelSubscription?.cancel();
  }

  // ─────────────────────────────────────────────────────────────
  // SCREEN STATE
  // ─────────────────────────────────────────────────────────────

  void _startScreenListener(String userId) {
    try {
      debugPrint("🔍 SensorListener: Starting Screen State Listener...");
      final Screen screen = Screen();
      _screenSubscription = screen.screenStateStream?.listen(
        (ScreenStateEvent event) {
          String status;
          switch (event) {
            case ScreenStateEvent.SCREEN_ON:
              status = "Screen_On";
              break;
            case ScreenStateEvent.SCREEN_OFF:
              status = "Screen_Off";
              break;
            case ScreenStateEvent.SCREEN_UNLOCKED:
              status = "Screen_Unlocked";
              break;
            default:
              status = "Screen_Unknown";
          }

          debugPrint("📱 Screen Event: $status");

          // BUG FIX: immediate=true ensures this fires an HTTP request right
          // away instead of waiting 10 seconds for the debounce timer.
          // Without this, if the phone's screen turns off and Android kills
          // the process before the timer fires, the event is lost.
          _sendData(userId, "Screen_Event", status, immediate: true);
        },
        onError: (e) => debugPrint("Screen State Stream Error: $e"),
      );
    } catch (e) {
      debugPrint("Screen State Setup Error: $e");
    }
  }

  // ─────────────────────────────────────────────────────────────
  // ACCELEROMETER
  // ─────────────────────────────────────────────────────────────

  void _startAccelerometerListener(String userId) {
    try {
      debugPrint("🔍 SensorListener: Starting Accelerometer Listener...");

      // Throttle: only upload one high-motion event per 3 seconds to avoid
      // flooding the queue during sustained shaking.
      DateTime? _lastMotionUpload;

      _accelSubscription = accelerometerEventStream().listen(
        (AccelerometerEvent event) {
          final double magnitude =
              sqrt(event.x * event.x + event.y * event.y + event.z * event.z);

          if (magnitude > 15.0) {
            final now = DateTime.now();
            if (_lastMotionUpload == null ||
                now.difference(_lastMotionUpload!) >
                    const Duration(seconds: 3)) {
              _lastMotionUpload = now;
              // BUG FIX: immediate=true — same reason as screen events.
              _sendData(
                userId,
                "High_Motion_Event",
                magnitude.toStringAsFixed(2),
                immediate: true,
              );
            }
          }
        },
        onError: (e) => debugPrint("Accelerometer Stream Error: $e"),
      );
    } catch (e) {
      debugPrint("Sensor Setup Error: $e");
    }
  }

  // ─────────────────────────────────────────────────────────────
  // HELPER
  // ─────────────────────────────────────────────────────────────

  Future<void> _sendData(
    String userId,
    String dataType,
    String value, {
    bool immediate = false,
  }) async {
    // Always prefer the freshly stored userId in case it changed.
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload();
      final String currentId =
          prefs.getString('user_id') ?? _cachedUserId ?? userId;
      await BackgroundServiceHelper.sendToSheet(
        currentId,
        dataType,
        value,
        immediate: immediate,
      );
    } catch (e) {
      debugPrint("SensorListener send error [$dataType]: $e");
    }
  }
}