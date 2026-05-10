import 'dart:convert';
import 'dart:developer' as dev;
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/models.dart';

class ApiService {
  static const String _baseUrl =
      'https://dulharakaushalya-tc-wpn-demo.hf.space';

  static const Duration _timeout = Duration(seconds: 180);

  // ─── Health check ─────────────────────────────────────────────────────────
  // Call this when app starts to warm up the Space
  static Future<Map<String, dynamic>?> healthCheck() async {
    try {
      dev.log('Checking health: $_baseUrl/health');
      final response = await http
          .get(Uri.parse('$_baseUrl/health'))
          .timeout(const Duration(seconds: 30));
      dev.log('Health response: ${response.statusCode} ${response.body}');
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
    } catch (e) {
      dev.log('Health check failed: $e');
    }
    return null;
  }

  // ─── Predict ──────────────────────────────────────────────────────────────
  static Future<PredictionResult> predict({
    required String noteText,
    required String noteType,
    List<String> anxietySupport = const [],
    List<String> controlSupport = const [],
  }) async {
    final stopwatch = Stopwatch()..start();

    final requestBody = {
      'note_text':       noteText,
      'note_type':       noteType,
      'anxiety_support': anxietySupport,
      'control_support': controlSupport,
    };

    dev.log('POST $_baseUrl/predict');
    dev.log('Body: ${jsonEncode(requestBody).substring(0, 80)}...');

    late http.Response response;
    try {
      response = await http
          .post(
            Uri.parse('$_baseUrl/predict'),
            headers: {
              'Content-Type': 'application/json',
              'Accept':       'application/json',
            },
            body: jsonEncode(requestBody),
          )
          .timeout(_timeout);
    } catch (e) {
      String msg = e.toString();
      if (e is SocketException) {
        msg = 'No internet connection or server unreachable. '
              'Please check your data/Wi-Fi.';
      } else if (e is http.ClientException) {
        msg = 'HTTP Client error: ${e.message}';
      }

      throw ApiException(
        statusCode: 0,
        message:    msg,
      );
    }

    stopwatch.stop();
    dev.log('Response: ${response.statusCode}');
    dev.log('Body: ${response.body.substring(0, response.body.length.clamp(0, 200))}');

    if (response.statusCode == 200) {
      late Map<String, dynamic> json;
      try {
        json = jsonDecode(response.body) as Map<String, dynamic>;
      } catch (e) {
        throw ApiException(
          statusCode: 200,
          message:    'JSON parse error: $e\nRaw: ${response.body}',
        );
      }

      // Inject latency if server didn't measure it
      if (!json.containsKey('latency_ms')) {
        json['latency_ms'] = stopwatch.elapsedMilliseconds;
      }

      // Ensure key_phrases is always a List<String>
      if (!json.containsKey('key_phrases') || json['key_phrases'] == null) {
        json['key_phrases'] = <String>[];
      }

      // Ensure temporal_context is always a String
      if (!json.containsKey('temporal_context') ||
          json['temporal_context'] == null) {
        json['temporal_context'] = '';
      }

      try {
        return PredictionResult.fromJson(json);
      } catch (e) {
        throw ApiException(
          statusCode: 200,
          message:    'Failed to parse PredictionResult: $e\nJSON: $json',
        );
      }
    } else if (response.statusCode == 422) {
      // FastAPI validation error — parse the detail
      try {
        final err = jsonDecode(response.body);
        final detail = err['detail']?.toString() ?? response.body;
        throw ApiException(statusCode: 422, message: 'Validation error: $detail');
      } catch (e) {
        throw ApiException(statusCode: 422, message: response.body);
      }
    } else if (response.statusCode == 500) {
      String msg = response.body;
      try {
        final err = jsonDecode(response.body);
        msg = err['error']?.toString() ?? msg;
      } catch (_) {}
      throw ApiException(statusCode: 500, message: 'Server error: $msg');
    } else {
      throw ApiException(
        statusCode: response.statusCode,
        message:    response.body,
      );
    }
  }

  // ─── Support set endpoints ────────────────────────────────────────────────
  static Future<void> addSupportNote(String text, String label) async {
    await http
        .post(
          Uri.parse('$_baseUrl/support'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'text': text, 'label': label}),
        )
        .timeout(const Duration(seconds: 30));
  }

  static Future<void> clearSupportNotes() async {
    await http
        .delete(Uri.parse('$_baseUrl/support/clear'))
        .timeout(const Duration(seconds: 30));
  }
}

class ApiException implements Exception {
  final int statusCode;
  final String message;

  ApiException({required this.statusCode, required this.message});

  // Human-readable version shown in the Flutter snackbar
  String get userMessage {
    if (statusCode == 0) {
      return 'Cannot reach the server.\n'
             'Check your internet connection and make sure the '
             'Hugging Face Space is running.';
    }
    if (statusCode == 422) {
      return 'The server rejected the request.\n$message';
    }
    if (statusCode == 500) {
      return 'The model encountered an error.\n$message';
    }
    return 'Error $statusCode: $message';
  }

  @override
  String toString() => 'ApiException($statusCode): $message';
}