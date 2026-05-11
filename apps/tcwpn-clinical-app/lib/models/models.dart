// lib/models/patient.dart
class Patient {
  final String id;
  final String name;
  final int age;
  final String gender;
  final String ward;
  final String referralDate;
  final List<Assessment> assessments;
  final RiskLevel latestRisk;
  final int totalVisits;
  final bool hasAlert;

  const Patient({
    required this.id,
    required this.name,
    required this.age,
    required this.gender,
    required this.ward,
    required this.referralDate,
    required this.assessments,
    required this.latestRisk,
    required this.totalVisits,
    this.hasAlert = false,
  });

  String get initials {
    final parts = name.trim().split(' ');
    if (parts.length >= 2) {
      return '${parts.first[0]}${parts.last[0]}'.toUpperCase();
    }
    return parts.first.substring(0, 2).toUpperCase();
  }

  Assessment? get latestAssessment =>
      assessments.isEmpty ? null : assessments.last;
}

// ─── Assessment ─────────────────────────────────────────────────────────────
class Assessment {
  final String id;
  final String patientId;
  final DateTime timestamp;
  final String noteText;
  final String noteType;
  final PredictionResult? result;
  final String clinicianId;
  final String? clinicianComment;

  const Assessment({
    required this.id,
    required this.patientId,
    required this.timestamp,
    required this.noteText,
    required this.noteType,
    this.result,
    required this.clinicianId,
    this.clinicianComment,
  });
}

// ─── Prediction result ───────────────────────────────────────────────────────
class PredictionResult {
  final String prediction;   // 'ANXIETY' or 'NO ANXIETY'
  final RiskLevel riskLevel;
  final double riskScore;
  final double confidence;
  final List<String> keyPhrases;
  final int supportK;
  final double threshold;
  final int latencyMs;
  final String temporalContext;

  const PredictionResult({
    required this.prediction,
    required this.riskLevel,
    required this.riskScore,
    required this.confidence,
    required this.keyPhrases,
    required this.supportK,
    required this.threshold,
    required this.latencyMs,
    required this.temporalContext,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      prediction:      json['prediction'] as String,
      riskLevel:       RiskLevel.fromString(json['risk_level'] as String),
      riskScore:       (json['risk_score'] as num).toDouble(),
      confidence:      (json['confidence'] as num).toDouble(),
      keyPhrases:      List<String>.from(json['key_phrases'] as List),
      supportK:        json['support_k'] as int,
      threshold:       (json['threshold'] as num).toDouble(),
      latencyMs:       json['latency_ms'] as int,
      temporalContext: json['temporal_context'] as String? ?? '',
    );
  }
}

// ─── Risk level ──────────────────────────────────────────────────────────────
enum RiskLevel { 
  low, moderate, high, veryHigh;

  static RiskLevel fromString(String s) {
    switch (s.toLowerCase()) {
      case 'very high':
      case 'very_high':  return RiskLevel.veryHigh;
      case 'high':       return RiskLevel.high;
      case 'moderate':   return RiskLevel.moderate;
      default:           return RiskLevel.low;
    }
  }
}

extension RiskLevelX on RiskLevel {

  String get label {
    switch (this) {
      case RiskLevel.low:      return 'Low risk';
      case RiskLevel.moderate: return 'Moderate risk';
      case RiskLevel.high:     return 'High risk';
      case RiskLevel.veryHigh: return 'Very high risk';
    }
  }

  String get shortLabel {
    switch (this) {
      case RiskLevel.low:      return 'LOW';
      case RiskLevel.moderate: return 'MOD';
      case RiskLevel.high:     return 'HIGH';
      case RiskLevel.veryHigh: return 'VERY HIGH';
    }
  }
}

// ─── Support note ────────────────────────────────────────────────────────────
class SupportNote {
  final String id;
  final String text;
  final String label;  // 'anxiety' or 'control'
  final DateTime addedAt;
  final double weight;
  final double recencyWeight;

  const SupportNote({
    required this.id,
    required this.text,
    required this.label,
    required this.addedAt,
    required this.weight,
    required this.recencyWeight,
  });
}

// ─── App Notification ────────────────────────────────────────────────────────
enum NotificationType { riskAlert, info, system }

class AppNotification {
  final String id;
  final String title;
  final String body;
  final DateTime timestamp;
  final NotificationType type;
  final RiskLevel? riskLevel;
  final String? patientId;
  final String? patientName;
  bool isRead;

  AppNotification({
    required this.id,
    required this.title,
    required this.body,
    required this.timestamp,
    this.type = NotificationType.info,
    this.riskLevel,
    this.patientId,
    this.patientName,
    this.isRead = false,
  });
}
