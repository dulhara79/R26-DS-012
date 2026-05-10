import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/foundation.dart';
import 'background/service_config.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
class BackgroundServiceHelper {
  // _isSyncing is per-isolate — each isolate has its own copy, which is correct.
  static bool _isSyncing = false;
  static const int _batchIntervalSeconds = 10;
  static Timer? _timer;

  /// true  → main UI isolate  → writes to 'offline_queue_main'
  /// false → background isolate → writes to 'offline_queue_bg'
  static bool isMainIsolate = true;

  static String get _queueKey =>
      isMainIsolate ? 'offline_queue_main' : 'offline_queue_bg';

  // ─────────────────────────────────────────────────────────────
  // PUBLIC API
  // ─────────────────────────────────────────────────────────────

  /// Enqueue [value] for [userId]/[type] and schedule (or immediately trigger)
  /// a sync to Google Sheets.
  ///
  /// [immediate] = true forces an instant upload — used for sensor events
  /// (screen on/off, high-motion) that must not wait for the 10-second batch
  /// window, because the process may be killed before the timer fires.
  static Future<void> sendToSheet(
    String userId,
    String type,
    String value, {
    bool immediate = false,
  }) async {
    final dataMap = {
      "userId": userId,
      "dataType": type,
      "value": value,
      "timestamp": DateTime.now().toIso8601String(),
      "token": ServiceConfig.authToken,
    };

    await _saveToOfflineQueue([dataMap]);

    if (immediate) {
      // Cancel any pending debounce timer and upload right now.
      _timer?.cancel();
      _timer = null;
      await retryOfflineQueue();
    } else {
      // Debounce: only start a new timer if one is not already running.
      _timer ??= Timer(
        const Duration(seconds: _batchIntervalSeconds),
        retryOfflineQueue,
      );
    }
  }

  // ─────────────────────────────────────────────────────────────
  // QUEUE PERSISTENCE
  // ─────────────────────────────────────────────────────────────

  static Future<void> _saveToOfflineQueue(
    List<Map<String, dynamic>> items,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    // Always reload so we see any writes from the other isolate.
    await prefs.reload();

    List<String> queue = prefs.getStringList(_queueKey) ?? [];

    for (var item in items) {
      if (queue.length >= 10000) {
        debugPrint("⚠️ Offline queue full — oldest item dropped.");
        queue.removeAt(0);
      }
      queue.add(jsonEncode(item));
    }

    await prefs.setStringList(_queueKey, queue);
  }

  // ─────────────────────────────────────────────────────────────
  // SYNC
  // ─────────────────────────────────────────────────────────────

  /// Upload every queued item from *this isolate's* queue to Google Sheets.
  /// Also migrates the legacy 'offline_queue' key on first run.
  static Future<void> retryOfflineQueue() async {
    _timer?.cancel();
    _timer = null;

    if (_isSyncing) return;
    _isSyncing = true;

    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload();

      List<String> queue = prefs.getStringList(_queueKey) ?? [];

      // ── One-time migration of the old single-key queue ──
      final List<String> oldQueue =
          prefs.getStringList('offline_queue') ?? [];
      if (oldQueue.isNotEmpty) {
        queue.insertAll(0, oldQueue);
        await prefs.remove('offline_queue');
        debugPrint("📦 Migrated ${oldQueue.length} items from legacy queue.");
      }

      if (queue.isEmpty) return;

      debugPrint("🔄 Syncing queue [$_queueKey]: ${queue.length} items");

      const chunkSize = 50;
      int failedFrom = -1; // index of the first chunk that failed

      for (int i = 0; i < queue.length; i += chunkSize) {
        final int end =
            (i + chunkSize < queue.length) ? i + chunkSize : queue.length;
        final List<String> chunkStrings = queue.sublist(i, end);

        final List<Map<String, dynamic>> batch = chunkStrings
            .map((s) => jsonDecode(s) as Map<String, dynamic>)
            .toList();

        try {
          final response = await http
              .post(
                Uri.parse(ServiceConfig.googleScriptUrl),
                headers: {"Content-Type": "application/json"},
                body: jsonEncode(batch),
              )
              .timeout(const Duration(seconds: 30));

          final bool ok = response.statusCode == 200 ||
              response.statusCode == 302 ||
              _isSuccessBody(response.body);

          if (ok) {
            debugPrint("✅ Chunk [$i–${end - 1}] sent (${batch.length} items)");
          } else {
            debugPrint(
                "⚠️ Server error on chunk [$i–${end - 1}]: ${response.statusCode}");
            failedFrom = i;
            break;
          }
        } catch (e) {
          debugPrint("❌ Chunk [$i–${end - 1}] failed: $e");
          failedFrom = i;
          break;
        }
      }

      // ── Build the remaining list ──
      // Reload to pick up any new items written while we were uploading.
      await prefs.reload();
      final List<String> freshQueue =
          prefs.getStringList(_queueKey) ?? [];

      List<String> remaining = [];

      if (failedFrom >= 0) {
        // Keep everything from the failed chunk onward.
        remaining = queue.sublist(failedFrom);
      }

      // Append any newly queued items (written after we started this sync).
      if (freshQueue.length > queue.length) {
        remaining.addAll(freshQueue.sublist(queue.length));
      }

      await prefs.setStringList(_queueKey, remaining);

      if (remaining.isEmpty) {
        debugPrint("✅ Queue [$_queueKey] fully cleared.");
      } else {
        debugPrint("⚠️ ${remaining.length} items still pending in [$_queueKey].");
      }
    } finally {
      _isSyncing = false;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // HELPERS
  // ─────────────────────────────────────────────────────────────

  static bool _isSuccessBody(String body) {
    try {
      final decoded = jsonDecode(body);
      return decoded['status'] == 'success';
    } catch (_) {
      return body.contains('success');
    }
  }

  static Future<String> getCachedId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('user_id') ?? "No_User_ID";
  }

  static Future<bool> isServiceRunning() async {
    return await FlutterBackgroundService().isRunning();
  }
}