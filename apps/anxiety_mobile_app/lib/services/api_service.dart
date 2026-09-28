import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'patient_session_service.dart';

class ApiService {
  // Replace this with your actual Hugging Face Space URL
  static const String baseUrl =
      'https://dewdu-physiological-anxiety-escalation.hf.space';

  // The shared R26-DS-012 central backend used by both patient and clinician
  // apps. BACKEND_BASE can override this default when the deployment changes.
  static const String centralBackendBaseUrl = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: 'https://finalize-humbly-monastery.ngrok-free.dev',
  );

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
        debugPrint(
          'Averaged feature window processed by server and saved to InfluxDB!',
        );
        return true;
      } else {
        debugPrint(
          'Server data quality guard rejected the window: ${response.statusCode} - ${response.body}',
        );
        return false;
      }
    } catch (e) {
      debugPrint('Network exception during feature ingest: $e');
      return false;
    }
  }

  /// Links this app's pseudonymous participant ID to the subject created by
  /// the clinician. The central backend intentionally exposes this pairing route
  /// without the clinician bearer token because the short-lived code is the
  /// credential being redeemed by the patient.
  static Future<Map<String, dynamic>> pairWithCentralBackend({
    required String participantId,
    required String pairingCode,
  }) async {
    final backendBase = centralBackendBaseUrl.trim().replaceFirst(
      RegExp(r'/$'),
      '',
    );
    if (backendBase.isEmpty) {
      return {
        'success': false,
        'message': 'The central backend is not configured in this app build.',
      };
    }

    try {
      final response = await http
          .post(
            Uri.parse('$backendBase/v1/subjects/pair'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'pairing_code': pairingCode.trim().toUpperCase(),
              'app_user_id': participantId,
            }),
          )
          .timeout(const Duration(seconds: 15));

      Map<String, dynamic> decoded = <String, dynamic>{};
      if (response.body.isNotEmpty) {
        final body = jsonDecode(response.body);
        if (body is Map) {
          decoded = Map<String, dynamic>.from(body);
        }
      }

      final subjectId = decoded['subject_id']?.toString() ?? '';
      if (response.statusCode == 200 && subjectId.isNotEmpty) {
        return {'success': true, 'subject_id': subjectId};
      }

      return {
        'success': false,
        'message':
            decoded['detail']?.toString() ??
            'The central backend rejected the pairing request.',
      };
    } catch (_) {
      return {
        'success': false,
        'message': 'Could not connect to the central backend.',
      };
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
        debugPrint(
          'User calibration parameters successfully loaded into server memory.',
        );
        return true;
      } else {
        debugPrint('Calibration failed: ${response.body}');
        return false;
      }
    } catch (e) {
      debugPrint('Network exception during calibration: $e');
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
        debugPrint('Prediction pipeline failed: ${response.body}');
        return {
          'status': 'error',
          'message': 'Forecast unavailable right now.',
        };
      }
    } catch (e) {
      debugPrint('Network exception during prediction: $e');
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
      debugPrint('Anxiety feedback upload failed: $e');
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

  // ─── CENTRAL BACKEND INTEGRATION ──────────────────────────────────────────
  // These methods talk to the R26-DS-012 central backend (the RAGF fusion
  // engine). They replace the dead sendToFusionModel placeholder.

  static String get _backendRoot =>
      centralBackendBaseUrl.trim().replaceFirst(RegExp(r'/$'), '');

  /// Claims a subject for this AURA installation on the central backend.
  /// Idempotent — safe to retry on every app launch.
  static Future<String?> selfEnrol(String participantId) async {
    try {
      final installationSecret = await PatientSessionService.instance
          .getOrCreateInstallationSecret();
      final res = await http
          .post(
            Uri.parse('$_backendRoot/v1/subjects/self'),
            headers: const {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: jsonEncode({
              'app_user_id': participantId,
              'installation_secret': installationSecret,
            }),
          )
          .timeout(const Duration(seconds: 20));
      if (res.statusCode == 200) {
        final body = jsonDecode(res.body);
        if (body is! Map) return null;
        final subjectId = body['subject_id']?.toString() ?? '';
        final accessToken = body['access_token']?.toString() ?? '';
        final expiresAt = DateTime.tryParse(
          body['expires_at']?.toString() ?? '',
        );
        if (subjectId.isEmpty || accessToken.isEmpty || expiresAt == null) {
          return null;
        }
        await PatientSessionService.instance.saveSession(
          subjectId: subjectId,
          accessToken: accessToken,
          expiresAt: expiresAt,
        );
        return subjectId;
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Sends GAD-7 + demographics to the central backend for C4/DCAR scoring.
  /// Returns true on success. Triggers fusion server-side.
  static Future<bool> submitContextualIntake({
    required String participantId,
    required List<int> gad7Items,
    String? gender,
    int? age,
    String? edu,
  }) async {
    try {
      final headers = await PatientSessionService.instance
          .authenticatedHeaders();
      if (headers == null) return false;
      final payload = <String, dynamic>{
        'app_user_id': participantId,
        'gad7_items': gad7Items,
      };
      if (gender != null) payload['gender'] = gender.toLowerCase();
      if (age != null) payload['age'] = age;
      if (edu != null) payload['edu'] = edu;
      final res = await http
          .post(
            Uri.parse('$_backendRoot/v1/ingest/contextual'),
            headers: headers,
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 20));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Notifies the central backend to fetch C1's latest prediction.
  static Future<bool> submitPhysiologicalWindow({
    required String participantId,
  }) async {
    try {
      final headers = await PatientSessionService.instance
          .authenticatedHeaders();
      if (headers == null) return false;
      final res = await http
          .post(
            Uri.parse('$_backendRoot/v1/ingest/physiological'),
            headers: headers,
            body: jsonEncode({
              'app_user_id': participantId,
              'device_user_id': participantId,
            }),
          )
          .timeout(const Duration(seconds: 20));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Reads the latest fusion composite for the AURA home page.
  /// Returns {composite, band, message} or null on failure.
  static Future<Map<String, dynamic>?> getPatientRisk(String subjectId) async {
    try {
      final headers = await PatientSessionService.instance
          .authenticatedHeaders();
      if (headers == null) return null;
      final res = await http
          .get(
            Uri.parse('$_backendRoot/v1/patients/$subjectId/risk'),
            headers: headers,
          )
          .timeout(const Duration(seconds: 15));
      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Reads the patient-safe projection of server-created OPEN events.
  static Future<List<Map<String, dynamic>>?> getOpenAttentionEvents() async {
    try {
      final headers = await PatientSessionService.instance.authenticatedHeaders(
        includeJsonContentType: false,
      );
      if (headers == null) return null;
      final res = await http
          .get(
            Uri.parse(
              '$_backendRoot/v1/patients/me/attention-events?status=OPEN',
            ),
            headers: headers,
          )
          .timeout(const Duration(seconds: 15));
      if (res.statusCode != 200) return null;
      final decoded = jsonDecode(res.body);
      if (decoded is! Map || decoded['events'] is! List) return null;
      return (decoded['events'] as List)
          .whereType<Map>()
          .map((event) => Map<String, dynamic>.from(event))
          .toList();
    } catch (_) {
      return null;
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
        debugPrint('[Fusion] Score received: $decoded');
        return {'success': true, ...decoded};
      } else {
        debugPrint('[Fusion] Endpoint rejected request: ${response.body}');
        return {'success': false, 'message': 'Fusion endpoint error'};
      }
    } catch (e) {
      // Silently fail — fusion is a cross-team integration and should never
      // crash our own app if the teammate's server is offline.
      debugPrint('[Fusion] Could not reach fusion endpoint: $e');
      return {'success': false, 'message': 'Fusion offline'};
    }
  }
}
