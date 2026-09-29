import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'patient_session_service.dart';
import 'participant_identity_service.dart';

class AssignmentInvite {
  const AssignmentInvite(this.code, this.expiresAt);
  final String code;
  final DateTime expiresAt;
}

class ApiService {
  // Replace this with your actual Hugging Face Space URL
  static const String baseUrl =
      'https://dewdu-physiological-anxiety-escalation.hf.space';

  // Backend URLs are injected per environment; an old tunnel is never a safe
  // release fallback. Local HTTP is permitted only during development.
  static const String centralBackendBaseUrl = String.fromEnvironment(
    'BACKEND_BASE',
    defaultValue: '',
  );

  static String? backendRoot([String? override]) {
    final raw = (override ?? centralBackendBaseUrl).trim();
    final uri = Uri.tryParse(raw);
    if (uri == null ||
        uri.host.isEmpty ||
        (uri.scheme != 'https' &&
            (kReleaseMode ||
                uri.scheme != 'http' ||
                !const [
                  'localhost',
                  '127.0.0.1',
                  '10.0.2.2',
                ].contains(uri.host)))) {
      return null;
    }
    return raw.replaceFirst(RegExp(r'/$'), '');
  }

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

  /// Claims a subject for this AURA installation on the central backend.
  /// Idempotent — safe to retry on every app launch.
  static Future<String?> selfEnrol(
    String participantId, {
    String? pairingCode,
    String? expectedSubjectId,
    http.Client? client,
    PatientSessionService? sessionService,
    String? backendBase,
  }) async {
    final transport = client ?? http.Client();
    try {
      final root = backendRoot(backendBase);
      if (root == null) return null;
      final session = sessionService ?? PatientSessionService.instance;
      final installationSecret = await session.getOrCreateInstallationSecret();
      final res = await transport
          .post(
            Uri.parse('$root/v1/subjects/self'),
            headers: const {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: jsonEncode({
              'app_user_id': participantId,
              'installation_secret': installationSecret,
              if (pairingCode != null)
                'pairing_code': pairingCode.trim().toUpperCase(),
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
        if (subjectId.isEmpty ||
            accessToken.isEmpty ||
            expiresAt == null ||
            (expectedSubjectId != null && subjectId != expectedSubjectId)) {
          return null;
        }
        await session.saveSession(
          subjectId: subjectId,
          accessToken: accessToken,
          expiresAt: expiresAt,
        );
        return subjectId;
      }
      return null;
    } catch (_) {
      return null;
    } finally {
      if (client == null) transport.close();
    }
  }

  /// Existing patient-first subjects authorize clinicians through a short-lived
  /// patient-issued invite; the clinician app redeems the code with its JWT.
  static Future<AssignmentInvite?> createAssignmentInvite({
    http.Client? client,
    PatientSessionService? sessionService,
    String? participantId,
    String? backendBase,
  }) async {
    final root = backendRoot(backendBase);
    if (root == null) return null;
    final expectedSubject =
        await ParticipantIdentityService.getCentralSubjectId() ??
        (await (sessionService ?? PatientSessionService.instance)
                .currentSession())
            ?.subjectId;
    if (expectedSubject == null) return null;
    try {
      final res = await _requestWithRecovery(
        Uri.parse('$root/v1/patients/me/assignment-invites'),
        expectedSubjectId: expectedSubject,
        postBody: '{}',
        client: client,
        sessionService: sessionService,
        participantId: participantId,
        backendBase: backendBase,
      );
      if (res == null) return null;
      if (res.statusCode != 200) return null;
      final body = jsonDecode(res.body);
      if (body is! Map || body['invite_code'] is! String) return null;
      final expiry = DateTime.tryParse(body['expires_at']?.toString() ?? '');
      if (expiry == null || !expiry.isAfter(DateTime.now().toUtc())) {
        return null;
      }
      return AssignmentInvite(body['invite_code'] as String, expiry);
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
      final root = backendRoot();
      final subjectId = await ParticipantIdentityService.getCentralSubjectId();
      if (root == null || subjectId == null) return false;
      final payload = <String, dynamic>{
        'app_user_id': participantId,
        'gad7_items': gad7Items,
      };
      if (gender != null) payload['gender'] = gender.toLowerCase();
      if (age != null) payload['age'] = age;
      if (edu != null) payload['edu'] = edu;
      final res = await _requestWithRecovery(
        Uri.parse('$root/v1/ingest/contextual'),
        expectedSubjectId: subjectId,
        postBody: jsonEncode(payload),
      );
      return res?.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Notifies the central backend to fetch C1's latest prediction.
  static Future<bool> submitPhysiologicalWindow({
    required String participantId,
  }) async {
    try {
      final root = backendRoot();
      final subjectId = await ParticipantIdentityService.getCentralSubjectId();
      if (root == null || subjectId == null) return false;
      final res = await _requestWithRecovery(
        Uri.parse('$root/v1/ingest/physiological'),
        expectedSubjectId: subjectId,
        postBody: jsonEncode({
          'app_user_id': participantId,
          'device_user_id': participantId,
        }),
      );
      return res?.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Reads the latest fusion composite for the AURA home page.
  /// Returns {composite, band, message} or null on failure.
  static Future<Map<String, dynamic>?> getPatientRisk(
    String subjectId, {
    http.Client? client,
    PatientSessionService? sessionService,
    String? participantId,
    String? backendBase,
  }) async {
    try {
      final root = backendRoot(backendBase);
      if (root == null) return null;
      final res = await _requestWithRecovery(
        Uri.parse('$root/v1/patients/${Uri.encodeComponent(subjectId)}/risk'),
        expectedSubjectId: subjectId,
        client: client,
        sessionService: sessionService,
        participantId: participantId,
        backendBase: backendBase,
      );
      if (res == null) return null;
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
      final root = backendRoot();
      final subjectId = await ParticipantIdentityService.getCentralSubjectId();
      if (root == null || subjectId == null) return null;
      final res = await _requestWithRecovery(
        Uri.parse('$root/v1/patients/me/attention-events?status=OPEN'),
        expectedSubjectId: subjectId,
      );
      if (res == null) return null;
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

  static Future<bool>? _renewalInFlight;
  static String? _renewalSubject;

  static Future<bool> _renewSession(
    String expectedSubjectId, {
    http.Client? client,
    PatientSessionService? sessionService,
    String? participantId,
    String? backendBase,
  }) async {
    final pending = _renewalInFlight;
    if (client == null && sessionService == null && pending != null) {
      return _renewalSubject == expectedSubjectId ? pending : false;
    }
    final refresh = () async {
      final id =
          participantId ?? await ParticipantIdentityService.getParticipantId();
      if (id == null) return false;
      return (await selfEnrol(
            id,
            expectedSubjectId: expectedSubjectId,
            client: client,
            sessionService: sessionService,
            backendBase: backendBase,
          )) ==
          expectedSubjectId;
    }();
    if (client == null && sessionService == null) {
      _renewalInFlight = refresh;
      _renewalSubject = expectedSubjectId;
    }
    try {
      return await refresh;
    } finally {
      if (identical(_renewalInFlight, refresh)) {
        _renewalInFlight = null;
        _renewalSubject = null;
      }
    }
  }

  /// One installation-proof renewal and one retry at most. The old subject ID
  /// is checked before storing a new JWT; a mismatched response cannot retarget
  /// a cached clinical view to a different patient.
  static Future<http.Response?> _requestWithRecovery(
    Uri url, {
    required String expectedSubjectId,
    String? postBody,
    http.Client? client,
    PatientSessionService? sessionService,
    String? participantId,
    String? backendBase,
  }) async {
    final sessions = sessionService ?? PatientSessionService.instance;
    final transport = client ?? http.Client();
    try {
      var current = await sessions.currentSession();
      if (current == null) {
        if (!await _renewSession(
          expectedSubjectId,
          client: client,
          sessionService: sessionService,
          participantId: participantId,
          backendBase: backendBase,
        )) {
          return null;
        }
        current = await sessions.currentSession();
      }
      if (current == null || current.subjectId != expectedSubjectId) {
        return null;
      }
      for (var attempt = 0; attempt < 2; attempt++) {
        final headers = {
          'Accept': 'application/json',
          'Authorization': 'Bearer ${current!.accessToken}',
          if (postBody != null) 'Content-Type': 'application/json',
        };
        final result =
            await (postBody == null
                    ? transport.get(url, headers: headers)
                    : transport.post(url, headers: headers, body: postBody))
                .timeout(const Duration(seconds: 20));
        if (result.statusCode != 401) return result;
        await sessions.clearSession();
        if (attempt == 1 ||
            !await _renewSession(
              expectedSubjectId,
              client: client,
              sessionService: sessionService,
              participantId: participantId,
              backendBase: backendBase,
            )) {
          return null;
        }
        current = await sessions.currentSession();
        if (current == null || current.subjectId != expectedSubjectId) {
          return null;
        }
      }
      return null;
    } finally {
      if (client == null) transport.close();
    }
  }
}
