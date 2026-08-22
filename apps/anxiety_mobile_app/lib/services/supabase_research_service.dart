import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

/// Main-isolate Supabase integration for research data.
///
/// Credentials are supplied at build/run time. The preferred Flutter names are
/// SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY; NEXT_PUBLIC_* aliases are also
/// accepted so the same values can be reused from web-oriented env files.
///
/// Never place a service-role/secret key in the mobile app.
class SupabaseResearchService {
  SupabaseResearchService._();

  static const String _url = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: String.fromEnvironment('NEXT_PUBLIC_SUPABASE_URL'),
  );

  static const String _publishableKey = String.fromEnvironment(
    'SUPABASE_PUBLISHABLE_KEY',
    defaultValue: String.fromEnvironment(
      'NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY',
    ),
  );

  static bool _initialized = false;

  static bool get isConfigured =>
      _url.trim().isNotEmpty && _publishableKey.trim().isNotEmpty;

  static bool get isInitialized => _initialized;

  static Future<bool> initialize() async {
    if (_initialized) return true;

    if (!isConfigured) {
      debugPrint(
        'Supabase: not configured. Provide SUPABASE_URL / '
        'SUPABASE_PUBLISHABLE_KEY (or NEXT_PUBLIC_* aliases) with '
        '--dart-define.',
      );
      return false;
    }

    try {
      await Supabase.initialize(
        url: _url,
        publishableKey: _publishableKey,
      );

      _initialized = true;
      debugPrint('Supabase: initialized.');
      return true;
    } catch (e, st) {
      debugPrint('Supabase initialization failed: $e');
      debugPrint('$st');
      return false;
    }
  }

  static SupabaseClient? get client {
    if (!_initialized) return null;
    return Supabase.instance.client;
  }

  /// Ensures this installation has an authenticated Supabase identity and a
  /// row mapping that identity to the pseudonymous research participant code.
  static Future<String?> ensureParticipant(String participantCode) async {
    if (participantCode.isEmpty || participantCode == 'No_User_ID') {
      return null;
    }

    if (!await initialize()) return null;

    final supabase = client!;

    try {
      if (supabase.auth.currentSession == null) {
        await supabase.auth.signInAnonymously(
          data: {'participant_code': participantCode},
        );
      }

      final authUser = supabase.auth.currentUser;

      if (authUser == null) {
        debugPrint('Supabase: anonymous authentication returned no user.');
        return null;
      }

      await supabase.from('participants').upsert(
        {
          'auth_user_id': authUser.id,
          'participant_code': participantCode,
          'active': true,
        },
        onConflict: 'auth_user_id',
      );

      return authUser.id;
    } catch (e, st) {
      debugPrint('Supabase participant registration failed: $e');
      debugPrint('$st');
      return null;
    }
  }

  /// Inserts a batch of already-normalized research events.
  ///
  /// RLS verifies that every row belongs to the currently authenticated
  /// Supabase user.
  static Future<void> insertSensorEvents(
    String participantCode,
    List<Map<String, dynamic>> queuedEvents,
  ) async {
    if (queuedEvents.isEmpty) return;

    final authUserId = await ensureParticipant(participantCode);

    if (authUserId == null) {
      throw StateError('Supabase participant is not available.');
    }

    final rows = queuedEvents.map((event) {
      final rawValue = event['value'];

      dynamic valueJson = rawValue;

      if (rawValue is String) {
        try {
          valueJson = jsonDecode(rawValue);
        } catch (_) {
          valueJson = {'value': rawValue};
        }
      }

      if (valueJson is! Map && valueJson is! List) {
        valueJson = {'value': valueJson};
      }

      return <String, dynamic>{
        'event_id': event['eventId'],
        'auth_user_id': authUserId,
        'participant_code': participantCode,
        'event_time': event['timestamp'],
        'event_type': event['dataType'],
        'value_json': valueJson,
        'source': event['source'] ?? 'android',
      };
    }).toList();

    // Raw sensor rows intentionally have INSERT-only RLS access for the mobile
    // participant. Using upsert/onConflict can require additional row access
    // during conflict handling and is rejected by that policy.
    //
    // Use plain INSERT and keep retries idempotent with the unique event_id
    // constraint instead.
    try {
      await client!.from('sensor_events').insert(rows);
    } on PostgrestException catch (e) {
      if (e.code != '23505') {
        rethrow;
      }

      // A duplicate makes a multi-row INSERT fail atomically.
      // Retry each row separately so previously uploaded event IDs are ignored
      // while genuinely new events in the same batch are still stored.
      for (final row in rows) {
        try {
          await client!.from('sensor_events').insert(row);
        } on PostgrestException catch (rowError) {
          if (rowError.code != '23505') {
            rethrow;
          }
        }
      }
    }
  }
}
