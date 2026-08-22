import 'dart:async';

import 'sensor_manager.dart';
import 'chest_strap_service.dart';
import 'ble_bridge.dart';
import 'anxiety_feedback_service.dart';

class UserManager {
  // This is the magic line that creates the single, permanent instance of UserManager
  static final UserManager _instance = UserManager._internal();

  // Whenever any file calls UserManager(), it always gets the exact same instance
  factory UserManager() => _instance;

  // An internal empty constructor needed for the singleton pattern
  UserManager._internal();

  // This is our active user ID storage slot
  String? _currentUserId;

  // This holds the active sensor tracker for the logged-in user
  SensorManager? sensorManager;

  // A simple checker to see if anyone is logged in right now
  bool get isLoggedIn => _currentUserId != null;

  // A safe way to grab the current user ID from anywhere in the app
  String get currentUserId {
    if (_currentUserId == null) {
      return 'guest_user'; // Safe fallback string so the app never crashes
    }
    return _currentUserId!;
  }

  // LOGIN METHOD: Call this when the user types their ID and clicks submit
  void login(String userId) {
    if (_currentUserId == userId && sensorManager?.isCollecting == true) {
      // Re-wiring is harmless and recovers the stream subscription if a page
      // or lifecycle transition cancelled it.
      BleBridge().wireChestStrap();
      return;
    }

    sensorManager?.stopCollection();
    BleBridge().unwireChestStrap();
    _currentUserId = userId;
    print('User session initialized for identity: $userId');

    // Automatically create a fresh SensorManager dedicated entirely to this user
    sensorManager = SensorManager(
      userId: userId,
      samplingRate: 1,
    ); // Real hardware and the simulator send one feature packet per second.

    // Instantly start the 60-second background background tracking loop
    sensorManager!.startCollection();
    print('Background data collection loop kicked off for $userId');

    // Wire chest strap data to SensorManager via BleBridge
    BleBridge().wireChestStrap();
    unawaited(AnxietyFeedbackService().initializeForUser(userId));
  }

  // LOGOUT METHOD: Call this if the user wants to switch identities
  void logout() {
    print('Shutting down session for user: $_currentUserId');

    // Safely stop the background timers and empty the chest strap memory buffers
    sensorManager?.stopCollection();
    sensorManager = null;

    // Unwire BLE routing to SensorManager
    BleBridge().unwireChestStrap();
    unawaited(AnxietyFeedbackService().stop());

    // Disconnect bluetooth
    ChestStrapService().disconnect();

    // Clear out the user ID completely
    _currentUserId = null;
  }
}
