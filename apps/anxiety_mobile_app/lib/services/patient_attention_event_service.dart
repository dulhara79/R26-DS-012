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
  static const Duration _maximumRetryInterval = Duration(minutes: 5);
  Timer? _timer;
  bool _requestInFlight = false;
  bool _active = false;
  int _generation = 0;
  int _failures = 0;

  void startPolling() {
    _timer?.cancel();
    _active = true;
    _failures = 0;
    final generation = ++_generation;
    unawaited(_poll(generation));
  }

  void stopPolling() {
    _active = false;
    _generation++;
    _timer?.cancel();
    _timer = null;
  }

  Future<void> refreshNow() async {
    if (!_active) return;
    _timer?.cancel();
    await _poll(_generation);
  }

  Future<void> _poll(int generation) async {
    if (!_active || generation != _generation) return;
    final result = await fetchOpenEvents(generation: generation);
    if (!_active || generation != _generation) return;
    if (result == null) {
      _failures = (_failures + 1).clamp(0, 4);
    } else {
      _failures = 0;
    }
    final retry = Duration(seconds: 15 * (1 << _failures));
    final delay = _failures == 0
        ? _pollInterval
        : retry > _maximumRetryInterval
        ? _maximumRetryInterval
        : retry;
    _timer?.cancel();
    _timer = Timer(delay, () => unawaited(_poll(generation)));
  }

  Future<List<PatientAttentionEvent>?> fetchOpenEvents({
    int? generation,
  }) async {
    if (_requestInFlight) return null;
    _requestInFlight = true;
    try {
      final payloads = await ApiService.getOpenAttentionEvents();
      if (payloads == null) return null;
      if (generation != null && (!_active || generation != _generation)) {
        return const [];
      }
      final events = <PatientAttentionEvent>[];
      for (final payload in payloads) {
        if (generation != null && (!_active || generation != _generation)) {
          break;
        }
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
    } catch (error) {
      debugPrint('Patient attention polling failed: $error');
      return null;
    } finally {
      _requestInFlight = false;
    }
  }
}
