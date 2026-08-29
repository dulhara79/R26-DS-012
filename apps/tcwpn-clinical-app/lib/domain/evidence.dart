// lib/domain/evidence.dart
//
// CARE-AnxRAG response model.
//
// The central design rule: this model NEVER manufactures an answer. The RAG
// service can fail, abstain, or refuse on safety grounds, and each of those is
// a distinct, first-class state the clinician must be able to tell apart. A
// missing answer is information, not an error to be swallowed.

import 'package:flutter/foundation.dart';

/// One retrieved source backing a claim in the answer.
@immutable
class EvidenceCitation {
  /// Marker used inline in the answer text, e.g. "S1".
  final String citationId;
  final String? title;

  /// Publisher, e.g. "PubMed Anxiety Research", "World Health Organization".
  final String? sourceName;

  /// The retrieved passage. Truncated server-side with an ellipsis.
  final String? excerpt;
  final String? url;

  /// Study design as classified by CARE-AnxRAG, e.g. "meta_analysis",
  /// "randomized_controlled_trial", "government_health_information".
  final String? evidenceLevel;

  const EvidenceCitation({
    required this.citationId,
    this.title,
    this.sourceName,
    this.excerpt,
    this.url,
    this.evidenceLevel,
  });

  /// Machine tokens like `randomized_controlled_trial` are unreadable in a
  /// clinical UI. This is presentation only — the raw value is preserved.
  String get evidenceLevelLabel {
    final raw = evidenceLevel;
    if (raw == null || raw.isEmpty) return 'Unclassified';
    return raw
        .split('_')
        .where((word) => word.isNotEmpty)
        .map((word) => word[0].toUpperCase() + word.substring(1))
        .join(' ');
  }

  bool get hasUrl => (url ?? '').isNotEmpty;

  factory EvidenceCitation.fromJson(Map<String, dynamic> json) {
    return EvidenceCitation(
      citationId: json['citation_id']?.toString() ?? '?',
      title: json['title']?.toString(),
      sourceName: json['source_name']?.toString(),
      excerpt: json['excerpt']?.toString(),
      url: json['url']?.toString(),
      evidenceLevel: json['evidence_level']?.toString(),
    );
  }
}

/// The four mutually exclusive outcomes of an evidence query.
///
/// These are ordered by severity of what the UI must communicate. The order
/// matters: a crisis bypass outranks everything, because in that case the RAG
/// service was never even called.
enum EvidenceState {
  /// The local crisis pre-screen matched. CARE-AnxRAG was NOT consulted.
  crisisBypass,

  /// The service ran but declined to answer: evidence was insufficient.
  /// There is a reason, and there is deliberately no answer.
  abstained,

  /// The service could not be reached at all.
  unavailable,

  /// A grounded answer with citations.
  answered,
}

@immutable
class EvidenceResult {
  /// False means the RAG call itself failed. Not an abstention.
  final bool available;
  final String? answer;
  final List<EvidenceCitation> citations;

  /// Retrieval/evidence quality on 0..1. This is NOT a diagnostic probability
  /// and must never be labelled as one in the UI.
  final double? confidence;

  /// 0..1. Higher means retrieved sources disagree with one another.
  final double? conflictScore;

  final bool abstained;
  final String? abstentionReason;
  final String safetyLevel;
  final String? safetyMessage;

  /// True when our own pre-screen fired before the network call.
  final bool localCrisisBypass;

  final String? error;
  final String? knowledgeBaseLastSyncAt;
  final int? latencyMs;

  const EvidenceResult({
    required this.available,
    this.answer,
    this.citations = const [],
    this.confidence,
    this.conflictScore,
    this.abstained = false,
    this.abstentionReason,
    this.safetyLevel = 'unknown',
    this.safetyMessage,
    this.localCrisisBypass = false,
    this.error,
    this.knowledgeBaseLastSyncAt,
    this.latencyMs,
  });

  /// Resolves the response into exactly one state. Evaluation order encodes
  /// the safety precedence and must not be reordered.
  EvidenceState get state {
    if (localCrisisBypass) return EvidenceState.crisisBypass;
    if (!available) return EvidenceState.unavailable;
    if (abstained) return EvidenceState.abstained;
    return EvidenceState.answered;
  }

  /// Guards against a service that returns available/non-abstained but no
  /// text. Rendering an empty answer card would imply an answer exists.
  bool get hasAnswer =>
      state == EvidenceState.answered && (answer ?? '').trim().isNotEmpty;

  /// Coarse banding for display. Deliberately three wide buckets: the
  /// underlying score is not precise enough to justify finer gradations.
  String get confidenceLabel {
    final value = confidence;
    if (value == null) return 'Not reported';
    if (value >= 0.75) return 'High';
    if (value >= 0.45) return 'Medium';
    return 'Low';
  }

  int? get confidencePercent =>
      confidence == null ? null : (confidence! * 100).round();

  /// Any non-zero disagreement is surfaced. The threshold is intentionally at
  /// zero rather than a tolerance: hiding small conflicts would defeat the
  /// contradiction-awareness this system exists to provide.
  bool get hasConflict => (conflictScore ?? 0) > 0;

  String get conflictLabel {
    final value = conflictScore;
    if (value == null) return 'Not assessed';
    if (value <= 0) return 'None detected';
    if (value < 0.34) return 'Low';
    if (value < 0.67) return 'Moderate';
    return 'High';
  }

  factory EvidenceResult.fromJson(Map<String, dynamic> json) {
    final rawCitations = json['citations'];
    return EvidenceResult(
      available: json['available'] == true,
      answer: json['answer']?.toString(),
      citations: rawCitations is List
          ? rawCitations
              .whereType<Map<String, dynamic>>()
              .map(EvidenceCitation.fromJson)
              .toList()
          : const [],
      confidence: (json['confidence'] as num?)?.toDouble(),
      conflictScore: (json['conflict_score'] as num?)?.toDouble(),
      abstained: json['abstained'] == true,
      abstentionReason: json['abstention_reason']?.toString(),
      safetyLevel: json['safety_level']?.toString() ?? 'unknown',
      safetyMessage: json['safety_message']?.toString(),
      localCrisisBypass: json['local_crisis_bypass'] == true,
      error: json['error']?.toString(),
      knowledgeBaseLastSyncAt: json['knowledge_base_last_sync_at']?.toString(),
      latencyMs: (json['latency_ms'] as num?)?.toInt(),
    );
  }

  /// Used when the request throws before any response is parsed, so the UI
  /// still receives a well-formed "unavailable" rather than an exception.
  factory EvidenceResult.failure(String message) =>
      EvidenceResult(available: false, error: message);
}
