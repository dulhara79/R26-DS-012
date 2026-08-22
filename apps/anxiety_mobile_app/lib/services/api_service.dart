import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Replace this with your actual Hugging Face Space URL
  static const String baseUrl =
      'https://dewdu-physiological-anxiety-escalation.hf.space';

  // INGEST ENDPOINT: Sends averaged features directly to the server
  static Future<bool> sendFeatureData({
    required String userId,
    required bool isWorn,
    required double meanHr,
    required double meanRr,
    required double sdnn,
    required double rmssd,
    required double meanBr,
    required double stdBr,
    required double meanTemp,
    required double stdTemp,
    required double meanAccMag,
    required double stdAccMag,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl/ingest'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'user_id': userId,
              'timestamp': DateTime.now().toUtc().toIso8601String(),
              'is_worn': isWorn,
              'mean_hr': meanHr,
              'mean_rr': meanRr,
              'sdnn': sdnn,
              'rmssd': rmssd,
              'mean_br': meanBr,
              'std_br': stdBr,
              'mean_temp': meanTemp,
              'std_temp': stdTemp,
              'mean_acc_mag': meanAccMag,
              'std_acc_mag': stdAccMag,
            }),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        print(
          'Averaged feature window processed by server and saved to InfluxDB!',
        );
        return true;
      } else {
        print(
          'Server data quality guard rejected the window: ${response.statusCode} - ${response.body}',
        );
        return false;
      }
    } catch (e) {
      print('Network exception during feature ingest: $e');
      return false;
    }
  }

  // CALIBRATION ENDPOINT: Caches per-user baseline stats for live Z-score scaling
  static Future<bool> setNormalizationParams({
    required String userId,
    required List<double> bMean,
    required List<double> bStd,
    required List<List<double>> baselineWindows,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl/set_norm_params/$userId'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'b_mean': bMean,
              'b_std': bStd,
              'baseline_windows': baselineWindows,
            }),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        print(
          'User calibration parameters successfully loaded into server memory.',
        );
        return true;
      } else {
        print('Calibration failed: ${response.body}');
        return false;
      }
    } catch (e) {
      print('Network exception during calibration: $e');
      return false;
    }
  }

  // PREDICT ENDPOINT: Requests the rolling 19-minute anomaly forecasting array
  static Future<Map<String, dynamic>> getEscalationForecast(
    String userId,
  ) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/predict/$userId'))
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print('Prediction pipeline failed: ${response.body}');
        return {'status': 'error', 'message': 'Forecast unavailable right now.'};
      }
    } catch (e) {
      print('Network exception during prediction: $e');
      return {'status': 'error', 'message': 'No internet connection.'};
    }
  }

  static Future<Map<String, dynamic>> getPhysiologicalHistory(
    String userId, {
    int days = 30,
  }) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/history/$userId?days=$days'))
          .timeout(const Duration(seconds: 15));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      return {
        'status': 'error',
        'message': response.statusCode == 404
            ? 'Your history is not available yet.'
            : 'Could not load your history.',
      };
    } catch (e) {
      return {'status': 'error', 'message': 'Could not load your history.'};
    }
  }

  static Future<bool> sendAnxietyFeedback(Map<String, dynamic> feedback) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl/feedback/anxiety'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(feedback),
          )
          .timeout(const Duration(seconds: 15));
      return response.statusCode == 200;
    } catch (e) {
      print('Anxiety feedback upload failed: $e');
      return false;
    }
  }

  static Future<Map<String, dynamic>> getWeeklyFeedbackSummary(
    String userId,
  ) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/feedback/weekly/$userId'))
          .timeout(const Duration(seconds: 15));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      return {'status': 'error'};
    } catch (_) {
      return {'status': 'error'};
    }
  }

  // ─── FUSION ENDPOINT ─────────────────────────────────────────────────────────
  // Sends our physiological trajectory to the teammate's multi-modal fusion model.
  // The fusion model receives our 10-step forecast and a single aggregated
  // physiological risk score, then weights and combines them with other modalities
  // (e.g. digital phenotyping) to produce a final holistic risk decision.
  //
  // TODO: Replace [fusionBaseUrl] with the teammate's actual endpoint URL
  //       once their Hugging Face Space is deployed.
  static const String _fusionBaseUrl =
      'https://PLACEHOLDER_FUSION_ENDPOINT.hf.space'; // ← swap this URL

  static Future<Map<String, dynamic>> sendToFusionModel({
    required String userId,
    required List<double> trajectory,
    required double physiologicalRiskScore,
  }) async {
    if (_fusionBaseUrl.contains('PLACEHOLDER_FUSION_ENDPOINT')) {
      return {
        'success': false,
        'status': 'not_configured',
        'message': 'Fusion endpoint is not configured yet',
      };
    }

    try {
      final response = await http
          .post(
            Uri.parse('$_fusionBaseUrl/fuse'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'user_id': userId,
              // 10-step future anxiety trajectory from our LSTM-AE
              'physiological_trajectory': trajectory,
              // Single aggregated risk score (0–100) derived from the trajectory
              'physiological_risk_score': physiologicalRiskScore,
              'timestamp': DateTime.now().toIso8601String(),
            }),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        print('[Fusion] Score received: $decoded');
        return {'success': true, ...decoded};
      } else {
        print('[Fusion] Endpoint rejected request: ${response.body}');
        return {'success': false, 'message': 'Fusion endpoint error'};
      }
    } catch (e) {
      // Silently fail — fusion is a cross-team integration and should never
      // crash our own app if the teammate's server is offline.
      print('[Fusion] Could not reach fusion endpoint: $e');
      return {'success': false, 'message': 'Fusion offline'};
    }
  }
}
