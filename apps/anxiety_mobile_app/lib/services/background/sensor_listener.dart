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

  void startListening(String userId) {
    _startScreenListener(userId);
    _startAccelerometerListener(userId);
  }

  void stopListening() {
    _screenSubscription?.cancel();
    _accelSubscription?.cancel();
  }

  void _startScreenListener(String userId) {
    try {
      Screen screen = Screen();
      _screenSubscription = screen.screenStateStream?.listen((
        ScreenStateEvent event,
      ) {
        String status = "Unknown";
        if (event == ScreenStateEvent.SCREEN_ON) status = "Screen_On";
        if (event == ScreenStateEvent.SCREEN_OFF) status = "Screen_Off";
        if (event == ScreenStateEvent.SCREEN_UNLOCKED)
          status = "Screen_Unlocked";

        _sendData(userId, "Screen_Event", status);
      }, onError: (e) => debugPrint("Screen State Stream Error: $e"));
    } catch (e) {
      debugPrint("Screen State Error: $e");
    }
  }

  void _startAccelerometerListener(String userId) {
    try {
      _accelSubscription = accelerometerEventStream().listen((
        AccelerometerEvent event,
      ) {
        double magnitude = sqrt(
          event.x * event.x + event.y * event.y + event.z * event.z,
        );
        // Filter for significant movements
        if (magnitude > 15.0) {
          _sendData(userId, "High_Motion_Event", magnitude.toStringAsFixed(2));
        }
      }, onError: (e) => debugPrint("Accelerometer Stream Error: $e"));
    } catch (e) {
      debugPrint("Sensor Error: $e");
    }
  }

  Future<void> _sendData(String userId, String dataType, String value) async {
    final prefs = await SharedPreferences.getInstance();
    String currentId = prefs.getString('user_id') ?? userId;
    await BackgroundServiceHelper.sendToSheet(currentId, dataType, value);
  }
}
