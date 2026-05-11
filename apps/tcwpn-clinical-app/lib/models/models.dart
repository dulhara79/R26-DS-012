// lib/models/models.dart

// ─── Patient ────────────────────────────────────────────────────────────────
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
    return parts.first.isNotEmpty ? parts.first.substring(0, parts.first.length > 1 ? 2 : 1).toUpperCase() : '??';
  }

  Assessment? get latestAssessment =>
      assessments.isEmpty ? null : assessments.last;

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'age': age,
    'gender': gender,
    'ward': ward,
    'referralDate': referralDate,
    'assessments': assessments.map((a) => a.toJson()).toList(),
    'latestRisk': latestRisk.name,
    'totalVisits': totalVisits,
    'hasAlert': hasAlert,
  };

  factory Patient.fromJson(Map<String, dynamic> json) => Patient(
    id: json['id'],
    name: json['name'],
    age: json['age'],
    gender: json['gender'],
    ward: json['ward'],
    referralDate: json['referralDate'],
    assessments: (json['assessments'] as List).map((a) => Assessment.fromJson(a)).toList(),
    latestRisk: RiskLevel.values.firstWhere((e) => e.name == json['latestRisk'], orElse: () => RiskLevel.low),
    totalVisits: json['totalVisits'],
    hasAlert: json['hasAlert'] ?? false,
  );
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

  Map<String, dynamic> toJson() => {
    'id': id,
    'patientId': patientId,
    'timestamp': timestamp.toIso8601String(),
    'noteText': noteText,
    'noteType': noteType,
    'result': result?.toJson(),
    'clinicianId': clinicianId,
    'clinicianComment': clinicianComment,
  };

  factory Assessment.fromJson(Map<String, dynamic> json) => Assessment(
    id: json['id'],
    patientId: json['patientId'],
    timestamp: DateTime.parse(json['timestamp']),
    noteText: json['noteText'],
    noteType: json['noteType'],
    result: json['result'] != null ? PredictionResult.fromJson(json['result']) : null,
    clinicianId: json['clinicianId'],
    clinicianComment: json['clinicianComment'],
  );
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

  Map<String, dynamic> toJson() => {
    'prediction': prediction,
    'risk_level': riskLevel.label,
    'risk_score': riskScore,
    'confidence': confidence,
    'key_phrases': keyPhrases,
    'support_k': supportK,
    'threshold': threshold,
    'latency_ms': latencyMs,
    'temporal_context': temporalContext,
  };

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

  Map<String, dynamic> toJson() => {
    'id': id,
    'text': text,
    'label': label,
    'addedAt': addedAt.toIso8601String(),
    'weight': weight,
    'recencyWeight': recencyWeight,
  };

  factory SupportNote.fromJson(Map<String, dynamic> json) => SupportNote(
    id: json['id'],
    text: json['text'],
    label: json['label'],
    addedAt: DateTime.parse(json['addedAt']),
    weight: (json['weight'] as num).toDouble(),
    recencyWeight: (json['recencyWeight'] as num).toDouble(),
  );
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

  Map<String, dynamic> toJson() => {
    'id': id,
    'title': title,
    'body': body,
    'timestamp': timestamp.toIso8601String(),
    'type': type.name,
    'riskLevel': riskLevel?.name,
    'patientId': patientId,
    'patientName': patientName,
    'isRead': isRead,
  };

  factory AppNotification.fromJson(Map<String, dynamic> json) => AppNotification(
    id: json['id'],
    title: json['title'],
    body: json['body'],
    timestamp: DateTime.parse(json['timestamp']),
    type: NotificationType.values.firstWhere((e) => e.name == json['type'], orElse: () => NotificationType.info),
    riskLevel: json['riskLevel'] != null ? RiskLevel.values.firstWhere((e) => e.name == json['riskLevel']) : null,
    patientId: json['patientId'],
    patientName: json['patientName'],
    isRead: json['isRead'] ?? false,
  );
}
