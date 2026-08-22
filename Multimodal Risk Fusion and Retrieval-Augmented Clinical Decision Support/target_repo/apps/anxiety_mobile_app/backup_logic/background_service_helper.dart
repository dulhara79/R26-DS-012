import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'background_service.dart';
import 'config.dart';
import 'package:flutter/foundation.dart';

class BackgroundServiceHelper {
  static final List<Map<String, dynamic>> _buffer = [];
  static bool _isSyncing = false;
  static const int _batchIntervalSeconds = 10;
  static Timer? _timer;

  // ── AUTH TOKEN (Injected at build time via --dart-define=AUTH_TOKEN=...) ──
  static const String _authToken = AppConfig.authToken;

  /// Add data to the buffer. Saves to persistent storage immediately to prevent loss on shutdown.
  static Future<void> sendToSheet(
    String userId,
    String type,
    String value,
  ) async {
    final dataMap = {
      "userId": userId,
      "dataType": type,
      "value": value,
      "timestamp": DateTime.now().toIso8601String(),
      "token": _authToken,
    };

    // Immediate persistence: Save to offline queue right away.
    // The flush timer will attempt to send it to the sheet shortly.
    await _saveToOfflineQueue([dataMap]);

    _timer ??= Timer(
      const Duration(seconds: _batchIntervalSeconds),
      retryOfflineQueue, // Use retry instead of flush for a unified logic
    );
  }

  // _flushBuffer is no longer needed with immediate persistence.
  // We rely on retryOfflineQueue which handles chunking and sending.

  /// Check if the response body indicates success.
  static bool _isSuccessBody(String body) {
    try {
      final decoded = jsonDecode(body);
      return decoded['status'] == 'success' || decoded['status'] == 'partial';
    } catch (_) {
      return false;
    }
  }

  /// Save failed items to SharedPreferences for later retry.
  static Future<void> _saveToOfflineQueue(
    List<Map<String, dynamic>> items,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    List<String> queue = prefs.getStringList('offline_queue') ?? [];

    // Cap offline queue at 10,000 items (~2-3MB max)
    // 10k items covers ~10 days of heavy data for one user
    for (var item in items) {
      if (queue.length >= 10000) {
        debugPrint("⚠️ Offline queue full (10,000 items). Oldest item dropped.");
        queue.removeAt(0);
      }
      queue.add(jsonEncode(item));
    }
    await prefs.setStringList('offline_queue', queue);
    debugPrint("📦 Offline queue size: ${queue.length}");
  }

  /// Retry all queued offline items. Call on app start and when connectivity restored.
  static Future<void> retryOfflineQueue() async {
    _timer?.cancel();
    _timer = null;

    if (_isSyncing) return;
    _isSyncing = true;

    final prefs = await SharedPreferences.getInstance();
    List<String> queue = prefs.getStringList('offline_queue') ?? [];
    if (queue.isEmpty) {
      _isSyncing = false;
      return;
    }

    debugPrint("🔄 Syncing queue: ${queue.length} items");

    const chunkSize = 50;
    List<String> remaining = [];

    for (int i = 0; i < queue.length; i += chunkSize) {
      final chunk = queue.skip(i).take(chunkSize).toList();
      final batch = chunk
          .map((s) => jsonDecode(s) as Map<String, dynamic>)
          .toList();

      try {
        final response = await http
            .post(
              Uri.parse(kGoogleScriptUrl),
              headers: {"Content-Type": "application/json"},
              body: jsonEncode(batch),
            )
            .timeout(const Duration(seconds: 25));

        if (response.statusCode == 200 ||
            response.statusCode == 302 ||
            _isSuccessBody(response.body)) {
          debugPrint("✅ Offline chunk sent: ${batch.length} items");
        } else {
          remaining.addAll(chunk);
        }
      } catch (e) {
        debugPrint("❌ Offline retry chunk failed: $e");
        remaining.addAll(chunk);
        break;
      }
    }

    await prefs.setStringList('offline_queue', remaining);

    if (remaining.isEmpty) {
      debugPrint("✅ Queue fully cleared");
    } else {
      debugPrint("⚠️ ${remaining.length} items still pending");
    }

    _isSyncing = false;
  }

  static Future<String> getCachedId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('user_id') ?? "Unknown";
  }

  /// Get current offline queue size (for debug display).
  static Future<int> getOfflineQueueSize() async {
    final prefs = await SharedPreferences.getInstance();
    return (prefs.getStringList('offline_queue') ?? []).length;
  }

  /// Check if the background service is currently running.
  static Future<bool> isServiceRunning() async {
    return await FlutterBackgroundService().isRunning();
  }
}
