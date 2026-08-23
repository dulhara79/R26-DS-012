import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:usage_stats/usage_stats.dart';

/// Centralised Android/iOS permission onboarding for research data collection.
///
/// Permission prompts should happen during the normal participant setup flow,
/// not when a participant opens the detailed behavioural/sensing page.
class ResearchPermissionService {
  static const String _promptedKey = 'research_permissions_prompted_v2';

  static Future<bool> hasCompletedPermissionOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_promptedKey) ?? false;
  }

  /// Requests the permissions used by the existing multimodal collection
  /// pipeline. When [force] is false, the flow runs at most once per install.
  ///
  /// Participants are not blocked if they decline a permission. The sensing
  /// pipeline will simply report reduced coverage and the UI can direct them
  /// to system settings later if they choose to enable it.
  static Future<void> requestMissingPermissions({bool force = false}) async {
    if (kIsWeb || !(Platform.isAndroid || Platform.isIOS)) return;

    final prefs = await SharedPreferences.getInstance();
    if (!force && (prefs.getBool(_promptedKey) ?? false)) return;

    try {
      final standardPermissions = <Permission>[
        Permission.location,
        Permission.phone,
        Permission.sms,
        Permission.notification,
      ];

      if (Platform.isAndroid) {
        standardPermissions.addAll([
          Permission.bluetoothScan,
          Permission.bluetoothConnect,
        ]);
      }

      await standardPermissions.request();

      // Android handles background location separately on recent versions.
      if (Platform.isAndroid && await Permission.location.isGranted) {
        await Permission.locationAlways.request();
      }

      if (Platform.isAndroid) {
        // These open Android-managed settings screens when needed.
        if (!await Permission.ignoreBatteryOptimizations.isGranted) {
          await Permission.ignoreBatteryOptimizations.request();
        }

        final usageGranted = await UsageStats.checkUsagePermission() ?? false;
        if (!usageGranted) {
          await UsageStats.grantUsagePermission();
        }
      }
    } catch (e) {
      debugPrint('Research permission onboarding error: $e');
    } finally {
      // Mark the onboarding as shown even if some permissions were declined so
      // the app does not repeatedly interrupt the participant on every launch.
      await prefs.setBool(_promptedKey, true);
    }
  }

  /// Clears only the local "already prompted" marker. Useful for controlled
  /// testing/debugging; it does not change Android permission state.
  static Future<void> resetOnboardingMarker() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_promptedKey);
  }
}
