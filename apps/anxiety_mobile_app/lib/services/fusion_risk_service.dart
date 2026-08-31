import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'api_service.dart';
import 'participant_identity_service.dart';

/// The patient-facing view of the fusion result.
///
/// This is deliberately thin. The backend serves two different views of the
/// same fusion row: the clinician gets per-modality contributions, gate
/// reasons and conformal sets; the patient gets only a composite, a band and
/// a plain-language message. We do not ask for more than that here, and we
/// must not display anything the backend did not send.
class FusionRisk {
  /// Backend composite, on its native 0..1 scale.
  final double? composite;

  /// GREEN / AMBER / RED / GREY. GREY means the fusion gate refused to
  /// produce a score (for example only one modality was available), and it
  /// must never be rendered as if it were a low score.
  final String band;

  /// Plain-language message written by the backend for the patient.
  final String? message;

  final DateTime? updatedAt;

  /// Whether the composite used two (provisional) or all three (complete)
  /// fusion modalities. C2 is intentionally excluded from this count.
  final String assessmentStatus;

  final List<String> missingModalities;

  const FusionRisk({
    required this.composite,
    required this.band,
    this.message,
    this.updatedAt,
    this.assessmentStatus = 'insufficient',
    this.missingModalities = const [],
  });

  /// True only when the backend actually produced a usable score.
  bool get hasScore => composite != null && band != 'GREY';

  String get assessmentLabel => assessmentStatus == 'complete'
      ? 'Complete assessment'
      : assessmentStatus == 'provisional'
      ? 'Provisional assessment'
      : 'Assessment incomplete';

  /// The gauge on the home page works on a 0..100 scale, but the backend
  /// composite is 0..1. Converting here, once, keeps the mistake from being
  /// repeated at each call site.
  double? get scoreOutOf100 =>
      composite == null ? null : (composite! * 100).clamp(0.0, 100.0);

  factory FusionRisk.fromJson(Map<String, dynamic> json) {
    final rawComposite = json['composite'];
    DateTime? parsedUpdatedAt;
    final rawUpdatedAt = json['updated_at'];
    if (rawUpdatedAt is String && rawUpdatedAt.isNotEmpty) {
      parsedUpdatedAt = DateTime.tryParse(rawUpdatedAt);
    }
    return FusionRisk(
      composite: rawComposite is num ? rawComposite.toDouble() : null,
      band: json['band']?.toString() ?? 'GREY',
      message: json['message']?.toString(),
      updatedAt: parsedUpdatedAt,
      assessmentStatus: json['assessment_status']?.toString() ?? 'insufficient',
      missingModalities:
          (json['missing_modalities'] as List<dynamic>? ?? const <dynamic>[])
          .map((value) => value.toString())
          .toList(growable: false),
    );
  }
}

/// Reads the composite risk produced by the fusion engine.
///
/// This endpoint is intentionally unauthenticated on the backend, so no token
/// is sent. If the call fails for any reason the result is null: a missing
/// score is shown as "Unavailable", never as a low score.
class FusionRiskService {
  FusionRiskService._();

  static final FusionRiskService instance = FusionRiskService._();

  final ValueNotifier<FusionRisk?> latest = ValueNotifier(null);

  Timer? _pollTimer;

  static const Duration _pollInterval = Duration(minutes: 5);
  static const Duration _timeout = Duration(seconds: 10);

  /// Fetches once. Returns null when unpaired, unreachable, or on any
  /// non-200 response.
  Future<FusionRisk?> fetch() async {
    final subjectId = await ParticipantIdentityService.getCentralSubjectId();
    if (subjectId == null || subjectId.isEmpty) {
      debugPrint('FusionRiskService: not paired with the central backend yet.');
      return null;
    }

    final base = ApiService.centralBackendBaseUrl.trim().replaceFirst(
      RegExp(r'/+$'),
      '',
    );
    if (base.isEmpty) {
      debugPrint('FusionRiskService: central backend base URL is not set.');
      return null;
    }

    try {
      final response = await http
          .get(
            Uri.parse('$base/v1/patients/$subjectId/risk'),
            headers: const {'Accept': 'application/json'},
          )
          .timeout(_timeout);

      if (response.statusCode != 200) {
        debugPrint(
          'FusionRiskService: backend returned ${response.statusCode}.',
        );
        return null;
      }

      final decoded = jsonDecode(response.body);
      if (decoded is! Map<String, dynamic>) return null;

      final risk = FusionRisk.fromJson(decoded);
      latest.value = risk;
      return risk;
    } catch (error) {
      debugPrint('FusionRiskService: fetch failed: $error');
      return null;
    }
  }

  void startPolling() {
    _pollTimer?.cancel();
    fetch();
    _pollTimer = Timer.periodic(_pollInterval, (_) => fetch());
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }
}
