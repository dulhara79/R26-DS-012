import 'dart:async';

import 'package:flutter/foundation.dart';

import 'anxiety_feedback_service.dart';
import 'api_service.dart';

class PatientAttentionEvent {
  const PatientAttentionEvent({
    required this.id,
    required this.eventType,
    required this.severity,
    required this.forecastHorizon,
    required this.status,
    required this.createdAt,
    required this.policyVersion,
  });

  final String id;
  final String eventType;
  final String severity;
  final int forecastHorizon;
  final String status;
  final DateTime createdAt;
  final String policyVersion;

  factory PatientAttentionEvent.fromJson(Map<String, dynamic> json) {
    String requiredString(String key) {
      final value = json[key];
      if (value is! String || value.trim().isEmpty) {
        throw FormatException('Missing patient attention event field: $key');
      }
      return value;
    }

    final rawHorizon = json['forecast_horizon'];
    if (rawHorizon is! num) {
      throw const FormatException(
        'Missing patient attention event field: forecast_horizon',
      );
    }
    final createdAt = DateTime.tryParse(requiredString('created_at'));
    if (createdAt == null) {
      throw const FormatException('Invalid patient attention event timestamp');
    }
    return PatientAttentionEvent(
      id: requiredString('id'),
      eventType: requiredString('event_type'),
      severity: requiredString('severity'),
      forecastHorizon: rawHorizon.toInt(),
      status: requiredString('status'),
      createdAt: createdAt,
      policyVersion: requiredString('policy_version'),
    );
  }
}

/// Polls the patient-safe endpoint. The central backend alone decides when an
/// episode becomes an AttentionEvent; Aura only presents each server event ID.
class PatientAttentionEventService {
  PatientAttentionEventService._();

  static final PatientAttentionEventService instance =
      PatientAttentionEventService._();

  static const Duration _pollInterval = Duration(minutes: 1);
  Timer? _timer;
  bool _requestInFlight = false;

  void startPolling() {
    _timer?.cancel();
    unawaited(fetchOpenEvents());
    _timer = Timer.periodic(_pollInterval, (_) => unawaited(fetchOpenEvents()));
  }

  void stopPolling() {
    _timer?.cancel();
    _timer = null;
  }

  Future<List<PatientAttentionEvent>> fetchOpenEvents() async {
    if (_requestInFlight) return const [];
    _requestInFlight = true;
    try {
      final payloads = await ApiService.getOpenAttentionEvents();
      if (payloads == null) return const [];
      final events = <PatientAttentionEvent>[];
      for (final payload in payloads) {
        try {
          final event = PatientAttentionEvent.fromJson(payload);
          events.add(event);
          await AnxietyFeedbackService().ingestServerAttentionEvent(
            eventId: event.id,
            createdAt: event.createdAt,
            leadMinutes: event.forecastHorizon,
          );
        } on FormatException catch (error) {
          debugPrint('Ignored malformed patient attention event: $error');
        }
      }
      return events;
    } finally {
      _requestInFlight = false;
    }
  }
}
