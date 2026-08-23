import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'supabase_research_service.dart';

class BackgroundServiceHelper {
  static bool _isSyncing = false;
  static const int _batchSize = 50;
  static const int _batchIntervalSeconds = 10;
  static Timer? _timer;
  static final Random _random = Random.secure();

  // Chest-strap vitals belong to the physiological component and must not be
  // stored in the Component 2 / general Supabase sensor_events stream.
  static const Set<String> _blockedResearchEventTypes = {
    'ChestStrap_Vitals',
  };

  /// true  → main UI isolate  → writes to 'offline_queue_main'
  /// false → background isolate → writes to 'offline_queue_bg'
  static bool isMainIsolate = true;

  static String get _queueKey =>
      isMainIsolate ? 'offline_queue_main' : 'offline_queue_bg';

  /// Storage-neutral research event entry point.
  ///
  /// Events are first persisted locally. Supabase upload is intentionally
  /// performed only by the main isolate; Android's background isolate keeps
  /// collecting safely even when no authenticated Supabase client is active.
  static Future<void> enqueueResearchEvent(
    String userId,
    String type,
    dynamic value, {
    bool immediate = false,
    DateTime? eventTime,
  }) async {
    if (_blockedResearchEventTypes.contains(type)) {
      debugPrint('Research event blocked from Supabase: $type');
      return;
    }

    final dataMap = <String, dynamic>{
      'eventId': _newEventId(),
      'userId': userId,
      'dataType': type,
      'value': value is String ? value : jsonEncode(value),
      'timestamp': (eventTime ?? DateTime.now()).toUtc().toIso8601String(),
      'source': kIsWeb ? 'web' : 'android',
    };

    await _saveToOfflineQueue([dataMap]);

    if (!isMainIsolate) {
      return;
    }

    if (immediate) {
      _timer?.cancel();
      _timer = null;
      await retryOfflineQueue();
    } else {
      _timer ??= Timer(
        const Duration(seconds: _batchIntervalSeconds),
        retryOfflineQueue,
      );
    }
  }

  /// Backwards-compatible alias while older app features are migrated away
  /// from the Google-Sheets-specific method name.
  static Future<void> sendToSheet(
    String userId,
    String type,
    String value, {
    bool immediate = false,
  }) =>
      enqueueResearchEvent(
        userId,
        type,
        value,
        immediate: immediate,
      );

  static Future<void> _saveToOfflineQueue(
    List<Map<String, dynamic>> items,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();

    final queue = prefs.getStringList(_queueKey) ?? <String>[];

    for (final item in items) {
      if (queue.length >= 10000) {
        debugPrint('⚠️ Offline queue full — oldest item dropped.');
        queue.removeAt(0);
      }
      queue.add(jsonEncode(item));
    }

    await prefs.setStringList(_queueKey, queue);
  }

  /// Flushes main/background/legacy queues to Supabase.
  ///
  /// This method is a no-op in the background isolate. The next foreground app
  /// start or connectivity restoration uploads both queues using the persisted
  /// authenticated Supabase session.
  static Future<void> retryOfflineQueue() async {
    _timer?.cancel();
    _timer = null;

    if (!isMainIsolate) {
      debugPrint('Supabase sync deferred: background isolate is collection-only.');
      return;
    }
    if (_isSyncing) return;
    _isSyncing = true;

    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload();

      final legacy = prefs.getStringList('offline_queue') ?? <String>[];
      if (legacy.isNotEmpty) {
        final mainQueue =
            prefs.getStringList('offline_queue_main') ?? <String>[];
        mainQueue.insertAll(0, legacy);
        await prefs.setStringList('offline_queue_main', mainQueue);
        await prefs.remove('offline_queue');
        debugPrint('📦 Migrated ${legacy.length} legacy queued events.');
      }

      await _syncQueueKey(prefs, 'offline_queue_main');
      await _syncQueueKey(prefs, 'offline_queue_bg');
    } finally {
      _isSyncing = false;
    }
  }

  static Future<void> _syncQueueKey(
    SharedPreferences prefs,
    String queueKey,
  ) async {
    await prefs.reload();
    var queue = prefs.getStringList(queueKey) ?? <String>[];
    if (queue.isEmpty) return;

    // Normalize legacy rows once and persist generated event IDs before any
    // network call. Retries then remain idempotent.
    queue = queue
        .map(_normalizeQueuedString)
        .where((encoded) => !_isBlockedQueuedEvent(encoded))
        .toList();
    await prefs.setStringList(queueKey, queue);

    if (queue.isEmpty) {
      debugPrint('✅ Queue [$queueKey] contains no uploadable events.');
      return;
    }

    debugPrint('🔄 Supabase sync [$queueKey]: ${queue.length} events');
    var completed = 0;

    while (completed < queue.length) {
      final end = min(completed + _batchSize, queue.length);
      final chunkStrings = queue.sublist(completed, end);
      final chunk = chunkStrings
          .map((s) => Map<String, dynamic>.from(jsonDecode(s) as Map))
          .toList();

      try {
        final byParticipant = <String, List<Map<String, dynamic>>>{};
        for (final event in chunk) {
          final participant = (event['userId'] ?? '').toString();
          if (participant.isEmpty || participant == 'No_User_ID') {
            throw StateError('Queued event has no valid participant ID.');
          }
          byParticipant.putIfAbsent(participant, () => []).add(event);
        }

        for (final entry in byParticipant.entries) {
          await SupabaseResearchService.insertSensorEvents(
            entry.key,
            entry.value,
          );
        }

        completed = end;
        debugPrint('✅ Supabase chunk uploaded: ${chunk.length} events');
      } catch (e, st) {
        debugPrint('❌ Supabase queue upload failed: $e');
        debugPrint('$st');
        break;
      }
    }

    await prefs.reload();
    final latest = prefs.getStringList(queueKey) ?? <String>[];

    // Preserve items appended while the upload was in flight. The prefix we
    // processed corresponds to the normalized snapshot stored above.
    final remaining = <String>[];
    if (completed < queue.length) {
      remaining.addAll(queue.sublist(completed));
    }
    if (latest.length > queue.length) {
      remaining.addAll(
        latest
            .sublist(queue.length)
            .map(_normalizeQueuedString)
            .where((encoded) => !_isBlockedQueuedEvent(encoded)),
      );
    }

    await prefs.setStringList(queueKey, remaining);
    if (remaining.isEmpty) {
      debugPrint('✅ Queue [$queueKey] fully cleared.');
    } else {
      debugPrint('⚠️ ${remaining.length} events remain in [$queueKey].');
    }
  }

  static bool _isBlockedQueuedEvent(String encoded) {
    try {
      final event = Map<String, dynamic>.from(jsonDecode(encoded) as Map);
      final type = (event['dataType'] ?? '').toString();
      if (_blockedResearchEventTypes.contains(type)) {
        debugPrint('Removed blocked queued research event: $type');
        return true;
      }
    } catch (e) {
      debugPrint('Could not inspect queued event type: $e');
    }
    return false;
  }

  static String _normalizeQueuedString(String encoded) {
    final event = Map<String, dynamic>.from(jsonDecode(encoded) as Map);
    event['eventId'] ??= _newEventId();
    event['source'] ??= kIsWeb ? 'web' : 'android';
    event.remove('token');

    final timestamp = DateTime.tryParse((event['timestamp'] ?? '').toString());
    if (timestamp != null) {
      event['timestamp'] = timestamp.toUtc().toIso8601String();
    }

    return jsonEncode(event);
  }

  static String _newEventId() {
    final micros = DateTime.now().toUtc().microsecondsSinceEpoch;
    final randomHex = List<int>.generate(8, (_) => _random.nextInt(256))
        .map((b) => b.toRadixString(16).padLeft(2, '0'))
        .join();
    return 'evt_${micros}_$randomHex';
  }

  static Future<String> getCachedId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('user_id') ?? 'No_User_ID';
  }

  static Future<bool> isServiceRunning() async {
    if (kIsWeb) return false;
    try {
      return await FlutterBackgroundService().isRunning();
    } catch (e) {
      debugPrint('Background service status unavailable: $e');
      return false;
    }
  }

  static Future<int> getOfflineQueueSize() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    final mainQ = prefs.getStringList('offline_queue_main') ?? <String>[];
    final bgQ = prefs.getStringList('offline_queue_bg') ?? <String>[];
    final legacyQ = prefs.getStringList('offline_queue') ?? <String>[];
    return mainQ.length + bgQ.length + legacyQ.length;
  }
}
