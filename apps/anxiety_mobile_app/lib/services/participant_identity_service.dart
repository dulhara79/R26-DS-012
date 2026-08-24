import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Keeps a participant's research identifier separate from their display name.
///
/// Only [participantIdKey] is used in API, InfluxDB, and model requests. The
/// display name stays in local app preferences and is never sent by this
/// service.
class ParticipantIdentityService {
  static const String participantIdKey = 'participant_id';
  static const String displayNameKey = 'display_name';
  static const String legacyUserIdKey = 'legacy_user_id';
  static const String centralSubjectIdKey = 'central_subject_id';

  static final RegExp _participantPattern = RegExp(r'^P_[A-F0-9]{16}$');

  static bool isParticipantId(String value) =>
      _participantPattern.hasMatch(value);

  static String generateParticipantId() {
    final random = Random.secure();
    final bytes = List<int>.generate(8, (_) => random.nextInt(256));
    final hex = bytes
        .map((byte) => byte.toRadixString(16).padLeft(2, '0'))
        .join()
        .toUpperCase();
    return 'P_$hex';
  }

  static Future<String> createForDisplayName(String displayName) async {
    final prefs = await SharedPreferences.getInstance();
    final participantId = generateParticipantId();
    await prefs.setString(displayNameKey, displayName.trim());
    await prefs.setString(participantIdKey, participantId);

    // Keep the old preference key temporarily because the rest of the app and
    // the background isolate already read it. Its value is now pseudonymous.
    await prefs.setString('user_id', participantId);
    return participantId;
  }

  /// Changes only the friendly name shown inside Aura.
  ///
  /// The research participant ID is intentionally left untouched so changing
  /// a display name cannot split calibration, sensor, or feedback records.
  static Future<void> updateDisplayName(String displayName) async {
    final trimmedName = displayName.trim();
    if (trimmedName.isEmpty || trimmedName.length > 80) {
      throw ArgumentError('Display name must contain 1 to 80 characters.');
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(displayNameKey, trimmedName);
  }

  /// One-time migration for installations created by the old login flow.
  ///
  /// The old app stored `{entered_name}_{four_digits}` as `user_id`, which is
  /// still identifying when a participant entered their real name. We retain
  /// that value only as a local migration reference, generate a research ID,
  /// and require a new baseline because server calibration belongs to the old
  /// identifier.
  static Future<bool> migrateLegacyIdentity() async {
    final prefs = await SharedPreferences.getInstance();
    final existingParticipantId = prefs.getString(participantIdKey);
    final oldUserId = prefs.getString('user_id');

    if (existingParticipantId != null &&
        isParticipantId(existingParticipantId)) {
      if (oldUserId != existingParticipantId) {
        await prefs.setString('user_id', existingParticipantId);
      }
      return false;
    }

    if (oldUserId == null || oldUserId.isEmpty) return false;

    if (isParticipantId(oldUserId)) {
      await prefs.setString(participantIdKey, oldUserId);
      return false;
    }

    final participantId = generateParticipantId();
    final inferredDisplayName = oldUserId.replaceFirst(RegExp(r'_\d{4}$'), '');

    await prefs.setString(legacyUserIdKey, oldUserId);
    if ((prefs.getString(displayNameKey) ?? '').isEmpty) {
      await prefs.setString(displayNameKey, inferredDisplayName);
    }
    await prefs.setString(participantIdKey, participantId);
    await prefs.setString('user_id', participantId);

    // The old norm_params row is tagged with the old identifier. Reusing its
    // local completion flag would make /predict fail for the new participant
    // ID, so the app must collect a fresh calm baseline.
    await prefs.setBool('calibration_complete', false);
    await prefs.remove('chest_strap_last_reading');

    debugPrint(
      'Migrated legacy research identity to pseudonymous participant ID.',
    );
    return true;
  }

  static Future<void> saveCentralSubjectId(String subjectId) async {
    final trimmed = subjectId.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('Central subject ID cannot be empty.');
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(centralSubjectIdKey, trimmed);
  }

  static Future<String?> getCentralSubjectId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(centralSubjectIdKey);
  }

  static Future<void> clearLocalIdentity() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(participantIdKey);
    await prefs.remove(displayNameKey);
    await prefs.remove(legacyUserIdKey);
    await prefs.remove(centralSubjectIdKey);
    await prefs.remove('user_id');
  }
}
