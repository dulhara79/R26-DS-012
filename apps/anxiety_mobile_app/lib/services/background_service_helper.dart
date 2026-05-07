import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/foundation.dart';
import 'background/service_config.dart';

class BackgroundServiceHelper {
  static bool _isSyncing = false;
  static const int _batchIntervalSeconds = 10;
  static Timer? _timer;

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
      "token": ServiceConfig.authToken,
    };

    // Immediate persistence: Save to offline queue right away.
    await _saveToOfflineQueue([dataMap]);

    _timer ??= Timer(
      const Duration(seconds: _batchIntervalSeconds),
      retryOfflineQueue,
    );
  }

  /// Check if the response body indicates success.
  static bool _isSuccessBody(String body) {
    try {
      final decoded = jsonDecode(body);
      return decoded['status'] == 'success';
    } catch (_) {
      return body.contains('success');
    }
  }

  static Future<void> _saveToOfflineQueue(
    List<Map<String, dynamic>> items,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    List<String> queue = prefs.getStringList('offline_queue') ?? [];

    // Cap offline queue at 10,000 items (~2-3MB max)
    for (var item in items) {
      if (queue.length >= 10000) {
        debugPrint("⚠️ Offline queue full (10,000 items). Oldest item dropped.");
        queue.removeAt(0);
      }
      queue.add(jsonEncode(item));
    }
    await prefs.setStringList('offline_queue', queue);
  }

  /// Retry all queued offline items.
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

    const chunkSize = 100;
    List<String> remaining = [];
    
    // Process in chunks to avoid large POST body issues
    for (int i = 0; i < queue.length; i += chunkSize) {
      int end = (i + chunkSize < queue.length) ? i + chunkSize : queue.length;
      List<String> chunkStrings = queue.sublist(i, end);
      
      List<Map<String, dynamic>> batch = chunkStrings
          .map((s) => jsonDecode(s) as Map<String, dynamic>)
          .toList();

      try {
        var response = await http
            .post(
              Uri.parse(ServiceConfig.googleScriptUrl),
              headers: {"Content-Type": "application/json"},
              body: jsonEncode(batch),
            )
            .timeout(const Duration(seconds: 25));

        if (response.statusCode == 200 || 
            response.statusCode == 302 || 
            _isSuccessBody(response.body)) {
          debugPrint("✅ Chunk of ${batch.length} sent successfully");
        } else {
          debugPrint("⚠️ Server error on chunk: ${response.statusCode}");
          remaining.addAll(queue.sublist(i));
          break;
        }
      } catch (e) {
        debugPrint("❌ Chunk send failed: $e");
        remaining.addAll(queue.sublist(i));
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
    return prefs.getString('user_id') ?? "No_User_ID";
  }
}
