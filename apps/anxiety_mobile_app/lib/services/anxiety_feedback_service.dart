import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_service.dart';
import 'chest_strap_service.dart';
import 'notification_helper.dart';

class PredictedEscalation {
  final double currentRisk;
  final double predictedPeakRisk;
  final int leadMinutes;

  const PredictedEscalation({
    required this.currentRisk,
    required this.predictedPeakRisk,
    required this.leadMinutes,
  });

  double get increase => predictedPeakRisk - currentRisk;
}

/// Detects a meaningful increase from the participant's current calibrated
/// risk to the model's 5-to-10-minute forecast.
///
/// Two forecasts separated in time must agree before an alert is emitted. This
/// rejects a single noisy prediction while still providing several minutes of
/// lead time. The gate emits only once until both current and forecast risk
/// recover to the low range.
class PredictiveEscalationGate {
  final int minimumLeadMinutes;
  final double elevatedThreshold;
  final double highThreshold;
  final double minimumElevatedIncrease;
  final double minimumHighIncrease;
  final double recoveryThreshold;
  final int requiredConfirmations;
  final Duration minimumConfirmationSpacing;
  final Duration maximumConfirmationGap;

  int _confirmationCount = 0;
  DateTime? _lastConfirmationAt;
  bool _alertIssuedForCurrentEpisode = false;

  PredictiveEscalationGate({
    this.minimumLeadMinutes = 5,
    this.elevatedThreshold = 45.0,
    this.highThreshold = 70.0,
    this.minimumElevatedIncrease = 20.0,
    this.minimumHighIncrease = 10.0,
    this.recoveryThreshold = 40.0,
    this.requiredConfirmations = 2,
    this.minimumConfirmationSpacing = const Duration(seconds: 20),
    this.maximumConfirmationGap = const Duration(minutes: 2),
  });

  PredictedEscalation? evaluate({
    required double currentRisk,
    required List<double> riskForecast,
    List<int>? forecastHorizonsMinutes,
    required DateTime observedAt,
  }) {
    final hasDirectHorizons =
        forecastHorizonsMinutes != null &&
        forecastHorizonsMinutes.length == riskForecast.length &&
        forecastHorizonsMinutes.every(
          (minutes) => minutes >= minimumLeadMinutes,
        );
    if (!hasDirectHorizons && riskForecast.length < minimumLeadMinutes) {
      _clearCandidate();
      return null;
    }

    final current = currentRisk.clamp(0.0, 100.0).toDouble();
    final List<MapEntry<int, double>> futurePoints;
    if (hasDirectHorizons) {
      futurePoints = List.generate(
        riskForecast.length,
        (index) => MapEntry(
          forecastHorizonsMinutes![index],
          riskForecast[index].clamp(0.0, 100.0).toDouble(),
        ),
      );
    } else {
      futurePoints = List.generate(
        riskForecast.length - (minimumLeadMinutes - 1),
        (offset) {
          final index = offset + minimumLeadMinutes - 1;
          return MapEntry(
            index + 1,
            riskForecast[index].clamp(0.0, 100.0).toDouble(),
          );
        },
      );
    }
    if (futurePoints.isEmpty) {
      _clearCandidate();
      return null;
    }
    final peak = futurePoints
        .map((point) => point.value)
        .reduce((a, b) => a >= b ? a : b);
    final increase = peak - current;

    final highEscalation =
        peak >= highThreshold && increase >= minimumHighIncrease;
    final elevatedEscalation =
        peak >= elevatedThreshold && increase >= minimumElevatedIncrease;
    final currentHigh = current >= highThreshold;
    final qualifies = currentHigh || highEscalation || elevatedEscalation;

    if (!qualifies) {
      _clearCandidate();
      if (current < recoveryThreshold && peak < recoveryThreshold) {
        _alertIssuedForCurrentEpisode = false;
      }
      return null;
    }

    if (_alertIssuedForCurrentEpisode) return null;

    if (_lastConfirmationAt != null) {
      final confirmationGap = observedAt.difference(_lastConfirmationAt!);
      if (confirmationGap.isNegative ||
          confirmationGap > maximumConfirmationGap) {
        _clearCandidate();
      } else if (confirmationGap < minimumConfirmationSpacing) {
        return null;
      }
    }
    _lastConfirmationAt = observedAt;
    _confirmationCount++;
    if (_confirmationCount < requiredConfirmations) return null;

    final targetThreshold = currentHigh || highEscalation
        ? highThreshold
        : elevatedThreshold;
    final requiredIncrease = currentHigh
        ? 0.0
        : highEscalation
        ? minimumHighIncrease
        : minimumElevatedIncrease;
    var leadMinutes = currentHigh ? 0 : minimumLeadMinutes;
    if (!currentHigh) {
      for (final point in futurePoints) {
        final predicted = point.value;
        if (predicted >= targetThreshold &&
            predicted - current >= requiredIncrease) {
          leadMinutes = point.key;
          break;
        }
      }
    }

    _alertIssuedForCurrentEpisode = true;
    _clearCandidate();
    return PredictedEscalation(
      currentRisk: current,
      predictedPeakRisk: max(current, peak),
      leadMinutes: leadMinutes,
    );
  }

  void _clearCandidate() {
    _confirmationCount = 0;
    _lastConfirmationAt = null;
  }

  void reset() {
    _clearCandidate();
    _alertIssuedForCurrentEpisode = false;
  }

  void allowRetry() {
    reset();
  }
}

class AnxietyAlertEvent {
  final String eventId;
  final String userId;
  final DateTime detectedAt;
  final double initialRiskScore;
  final double initialHr;
  final double initialBr;
  final double initialMotion;
  final String riskSource;
  final double? predictedRiskScore;
  final int? predictedLeadMinutes;
  final double? forecastIncrease;
  bool? confirmedAnxious;
  String? activity;
  String? intervention;
  DateTime? interventionAt;
  bool? interventionCompleted;
  String? alternativeAction;
  DateTime? followupAt;
  double? followupRiskScore;
  double? followupHr;
  double? followupBr;
  double? followupMotion;
  bool? feltBetter;

  AnxietyAlertEvent({
    required this.eventId,
    required this.userId,
    required this.detectedAt,
    required this.initialRiskScore,
    required this.initialHr,
    required this.initialBr,
    required this.initialMotion,
    this.riskSource = 'physiological',
    this.predictedRiskScore,
    this.predictedLeadMinutes,
    this.forecastIncrease,
    this.confirmedAnxious,
    this.activity,
    this.intervention,
    this.interventionAt,
    this.interventionCompleted,
    this.alternativeAction,
    this.followupAt,
    this.followupRiskScore,
    this.followupHr,
    this.followupBr,
    this.followupMotion,
    this.feltBetter,
  });

  factory AnxietyAlertEvent.fromJson(Map<String, dynamic> json) {
    return AnxietyAlertEvent(
      eventId: json['event_id'] as String,
      userId: json['user_id'] as String,
      detectedAt: DateTime.parse(json['detected_at'] as String),
      initialRiskScore: (json['initial_risk_score'] as num).toDouble(),
      initialHr: (json['initial_hr'] as num).toDouble(),
      initialBr: (json['initial_br'] as num).toDouble(),
      initialMotion: (json['initial_motion'] as num).toDouble(),
      riskSource: json['risk_source'] as String? ?? 'physiological',
      predictedRiskScore: (json['predicted_risk_score'] as num?)?.toDouble(),
      predictedLeadMinutes: (json['predicted_lead_minutes'] as num?)?.toInt(),
      forecastIncrease: (json['forecast_increase'] as num?)?.toDouble(),
      confirmedAnxious: json['confirmed_anxious'] as bool?,
      activity: json['activity'] as String?,
      intervention: json['intervention'] as String?,
      interventionAt: json['intervention_at'] == null
          ? null
          : DateTime.parse(json['intervention_at'] as String),
      interventionCompleted: json['intervention_completed'] as bool?,
      alternativeAction: json['alternative_action'] as String?,
      followupAt: json['followup_at'] == null
          ? null
          : DateTime.parse(json['followup_at'] as String),
      followupRiskScore: (json['followup_risk_score'] as num?)?.toDouble(),
      followupHr: (json['followup_hr'] as num?)?.toDouble(),
      followupBr: (json['followup_br'] as num?)?.toDouble(),
      followupMotion: (json['followup_motion'] as num?)?.toDouble(),
      feltBetter: json['felt_better'] as bool?,
    );
  }

  Map<String, dynamic> toJson() => {
    'event_id': eventId,
    'user_id': userId,
    'detected_at': detectedAt.toUtc().toIso8601String(),
    'initial_risk_score': initialRiskScore,
    'initial_hr': initialHr,
    'initial_br': initialBr,
    'initial_motion': initialMotion,
    'risk_source': riskSource,
    if (predictedRiskScore != null)
      'predicted_risk_score': predictedRiskScore,
    if (predictedLeadMinutes != null)
      'predicted_lead_minutes': predictedLeadMinutes,
    if (forecastIncrease != null) 'forecast_increase': forecastIncrease,
    if (confirmedAnxious != null) 'confirmed_anxious': confirmedAnxious,
    if (activity != null) 'activity': activity,
    if (intervention != null) 'intervention': intervention,
    if (interventionAt != null)
      'intervention_at': interventionAt!.toUtc().toIso8601String(),
    if (interventionCompleted != null)
      'intervention_completed': interventionCompleted,
    if (alternativeAction != null) 'alternative_action': alternativeAction,
    if (followupAt != null)
      'followup_at': followupAt!.toUtc().toIso8601String(),
    if (followupRiskScore != null) 'followup_risk_score': followupRiskScore,
    if (followupHr != null) 'followup_hr': followupHr,
    if (followupBr != null) 'followup_br': followupBr,
    if (followupMotion != null) 'followup_motion': followupMotion,
    if (feltBetter != null) 'felt_better': feltBetter,
  };
}

class AnxietyFeedbackService {
  static final AnxietyFeedbackService _instance =
      AnxietyFeedbackService._internal();

  factory AnxietyFeedbackService() => _instance;

  AnxietyFeedbackService._internal();

  static const String _eventsKey = 'anxiety_alert_events_v1';
  static const String _pendingUploadsKey = 'anxiety_feedback_pending_v1';
  String? _userId;
  StreamSubscription<ChestStrapReading>? _readingSubscription;
  DateTime? _lastReadingAt;
  final PredictiveEscalationGate _forecastGate = PredictiveEscalationGate();
  Timer? _forecastTimer;
  bool _forecastRequestInFlight = false;
  DateTime? _latestForecastAt;
  double? _latestFusionRisk;
  DateTime? _latestFusionAt;
  final ValueNotifier<double?> combinedRisk = ValueNotifier(null);
  final Map<String, Timer> _followupTimers = {};

  Future<void> initializeForUser(String userId) async {
    if (_userId == userId && _readingSubscription != null) return;
    await stop();
    _userId = userId;
    _readingSubscription = ChestStrapService().readingsStream.listen(
      _observeReading,
      onError: (error) => debugPrint('Anxiety alert monitor error: $error'),
    );
    unawaited(retryPendingUploads());
    unawaited(refreshForecast());
    _forecastTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      unawaited(refreshForecast());
    });
    await _restorePendingFollowups();
  }

  Future<void> stop() async {
    await _readingSubscription?.cancel();
    _readingSubscription = null;
    _forecastTimer?.cancel();
    _forecastTimer = null;
    _forecastRequestInFlight = false;
    _latestForecastAt = null;
    _userId = null;
    _lastReadingAt = null;
    _forecastGate.reset();
    _latestFusionRisk = null;
    _latestFusionAt = null;
    combinedRisk.value = null;
    for (final timer in _followupTimers.values) {
      timer.cancel();
    }
    _followupTimers.clear();
  }

  void _observeReading(ChestStrapReading reading) {
    final userId = _userId;
    if (userId == null || !reading.isWorn) {
      _forecastGate.reset();
      return;
    }

    final now = DateTime.now();
    if (_lastReadingAt != null &&
        now.difference(_lastReadingAt!) > const Duration(seconds: 10)) {
      _forecastGate.reset();
    }
    _lastReadingAt = now;
  }

  void updateFusionRisk(double riskScore) {
    _latestFusionRisk = riskScore.clamp(0.0, 100.0).toDouble();
    _latestFusionAt = DateTime.now();
    combinedRisk.value = _latestFusionRisk;
  }

  double? get latestFusionRisk {
    final observedAt = _latestFusionAt;
    if (_latestFusionRisk == null || observedAt == null) return null;
    if (DateTime.now().difference(observedAt) >= const Duration(seconds: 90)) {
      return null;
    }
    return _latestFusionRisk;
  }

  Future<void> refreshForecast() async {
    final userId = _userId;
    if (userId == null || _forecastRequestInFlight) return;
    final latestAt = _latestForecastAt;
    if (latestAt != null &&
        DateTime.now().difference(latestAt) < const Duration(seconds: 20)) {
      return;
    }

    _forecastRequestInFlight = true;
    try {
      final response = await ApiService.getEscalationForecast(userId);
      observeForecastResponse(response);
    } finally {
      _forecastRequestInFlight = false;
    }
  }

  void observeForecastResponse(
    Map<String, dynamic> response, {
    DateTime? observedAt,
  }) {
    if (response['status'] != 'success') return;
    final rawForecast = response['risk_forecast'];
    final rawHorizons = response['forecast_horizons_minutes'];
    final directForecast =
        rawForecast is List &&
        rawForecast.length == 2 &&
        rawForecast.every((value) => value is num) &&
        rawHorizons is List &&
        rawHorizons.length == 2 &&
        rawHorizons.every((value) => value is num) &&
        (rawHorizons[0] as num).toInt() == 5 &&
        (rawHorizons[1] as num).toInt() == 10;
    final legacyForecast =
        rawForecast is List &&
        rawForecast.length >= 10 &&
        rawForecast.every((value) => value is num);
    if (!directForecast && !legacyForecast) {
      // Incomplete or malformed forecasts must never trigger participant alerts.
      return;
    }

    final forecast = rawForecast
        .cast<num>()
        .take(directForecast ? 2 : 10)
        .map((value) => value.toDouble())
        .toList();
    final forecastHorizons = directForecast
        ? rawHorizons.cast<num>().map((value) => value.toInt()).toList()
        : null;
    final currentFromApi = response['current_risk_index'];
    final liveReading = ChestStrapService().hasLiveWornReading
        ? ChestStrapService().lastReading
        : null;
    if (liveReading == null || !liveReading.isWorn) return;

    final currentRisk = currentFromApi is num
        ? currentFromApi.toDouble()
        : liveReading.riskScore;
    final now = observedAt ?? DateTime.now();
    _latestForecastAt = now;
    final escalation = _forecastGate.evaluate(
      currentRisk: currentRisk,
      riskForecast: forecast,
      forecastHorizonsMinutes: forecastHorizons,
      observedAt: now,
    );
    if (escalation == null) return;
    final userId = _userId;
    if (userId == null) return;
    unawaited(_createPredictiveAlert(userId, liveReading, now, escalation));
  }

  Future<void> _createPredictiveAlert(
    String userId,
    ChestStrapReading reading,
    DateTime detectedAt,
    PredictedEscalation escalation,
  ) async {
    final event = AnxietyAlertEvent(
      eventId: 'anx:${detectedAt.toUtc().millisecondsSinceEpoch}',
      userId: userId,
      detectedAt: detectedAt,
      initialRiskScore: escalation.currentRisk,
      initialHr: reading.meanHR,
      initialBr: reading.meanBR,
      initialMotion: reading.stdAccMag,
      riskSource: 'physiological_forecast',
      predictedRiskScore: escalation.predictedPeakRisk,
      predictedLeadMinutes: escalation.leadMinutes,
      forecastIncrease: escalation.increase,
    );
    await _upsertEvent(event);
    final shown = await NotificationHelper.showAnxietyAlert(
      eventId: event.eventId,
      leadMinutes: escalation.leadMinutes,
    );
    if (!shown) {
      await _removeEvent(event.eventId);
      _forecastGate.allowRetry();
      return;
    }
    unawaited(_upload(event));
  }

  /// Exercises the complete Android notification and check-in route without
  /// sending a synthetic event to the research backend.
  Future<bool> showLocalTestAlert() async {
    final userId = _userId;
    final reading = ChestStrapService().hasLiveWornReading
        ? ChestStrapService().lastReading
        : null;
    if (userId == null || reading == null || !reading.isWorn) return false;

    final now = DateTime.now();
    final currentRisk = reading.riskScore.clamp(0.0, 100.0).toDouble();
    final predictedRisk = (currentRisk + 30.0).clamp(0.0, 100.0).toDouble();
    final event = AnxietyAlertEvent(
      eventId: 'anx:test:${now.toUtc().millisecondsSinceEpoch}',
      userId: userId,
      detectedAt: now,
      initialRiskScore: currentRisk,
      initialHr: reading.meanHR,
      initialBr: reading.meanBR,
      initialMotion: reading.stdAccMag,
      riskSource: 'notification_test',
      predictedRiskScore: predictedRisk,
      predictedLeadMinutes: 5,
      forecastIncrease: predictedRisk - currentRisk,
    );
    await _upsertEvent(event);
    final shown = await NotificationHelper.showAnxietyAlert(
      eventId: event.eventId,
      leadMinutes: 5,
    );
    if (!shown) await _removeEvent(event.eventId);
    return shown;
  }

  static Future<List<AnxietyAlertEvent>> _loadEvents() async {
    final prefs = await SharedPreferences.getInstance();
    // Notification action callbacks may run in a background isolate. Reload so
    // both isolates see the newest event before recording Yes/No feedback.
    await prefs.reload();
    final encoded = prefs.getString(_eventsKey);
    if (encoded == null || encoded.isEmpty) return [];
    try {
      final decoded = jsonDecode(encoded) as List<dynamic>;
      return decoded
          .map(
            (item) => AnxietyAlertEvent.fromJson(
              Map<String, dynamic>.from(item as Map),
            ),
          )
          .toList();
    } catch (error) {
      debugPrint('Could not read saved anxiety events: $error');
      return [];
    }
  }

  static Future<void> _saveEvents(List<AnxietyAlertEvent> events) async {
    final prefs = await SharedPreferences.getInstance();
    events.sort((a, b) => b.detectedAt.compareTo(a.detectedAt));
    final retained = events.take(100).map((event) => event.toJson()).toList();
    await prefs.setString(_eventsKey, jsonEncode(retained));
  }

  static Future<void> _upsertEvent(AnxietyAlertEvent event) async {
    final events = await _loadEvents();
    final index = events.indexWhere((saved) => saved.eventId == event.eventId);
    if (index == -1) {
      events.add(event);
    } else {
      events[index] = event;
    }
    await _saveEvents(events);
  }

  static Future<void> _removeEvent(String eventId) async {
    final events = await _loadEvents();
    events.removeWhere((event) => event.eventId == eventId);
    await _saveEvents(events);
  }

  static Future<AnxietyAlertEvent?> getEvent(String eventId) async {
    final events = await _loadEvents();
    for (final event in events) {
      if (event.eventId == eventId) return event;
    }
    return null;
  }

  static Future<Set<String>> _loadPendingUploadIds() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    return (prefs.getStringList(_pendingUploadsKey) ?? []).toSet();
  }

  static Future<void> _markUploadPending(String eventId) async {
    final prefs = await SharedPreferences.getInstance();
    final pending = await _loadPendingUploadIds();
    pending.add(eventId);
    await prefs.setStringList(_pendingUploadsKey, pending.toList());
  }

  static Future<void> _clearPendingUpload(String eventId) async {
    final prefs = await SharedPreferences.getInstance();
    final pending = await _loadPendingUploadIds();
    if (!pending.remove(eventId)) return;
    await prefs.setStringList(_pendingUploadsKey, pending.toList());
  }

  static Future<void> _upload(AnxietyAlertEvent event) async {
    if (event.riskSource == 'notification_test') return;
    final uploaded = await ApiService.sendAnxietyFeedback(event.toJson());
    if (uploaded) {
      await _clearPendingUpload(event.eventId);
    } else {
      await _markUploadPending(event.eventId);
    }
  }

  static Future<void> retryPendingUploads() async {
    final pending = await _loadPendingUploadIds();
    if (pending.isEmpty) return;
    final events = await _loadEvents();
    for (final event in events) {
      if (pending.contains(event.eventId)) {
        await _upload(event);
      }
    }
  }

  static Future<void> clearLocalEvents() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_eventsKey);
    await prefs.remove(_pendingUploadsKey);
  }

  static Future<Map<String, dynamic>> getLocalWeeklySummary(
    String userId,
  ) async {
    final cutoff = DateTime.now().subtract(const Duration(days: 7));
    final events = (await _loadEvents())
        .where(
          (event) =>
              event.userId == userId &&
              event.detectedAt.isAfter(cutoff) &&
              event.riskSource != 'notification_test',
        )
        .toList();
    final answered = events
        .where((event) => event.confirmedAnxious != null)
        .toList();
    final confirmed = answered
        .where((event) => event.confirmedAnxious == true)
        .length;

    String? mostCommon(Iterable<String?> values) {
      final counts = <String, int>{};
      for (final value in values) {
        final cleaned = value?.trim();
        if (cleaned == null || cleaned.isEmpty) continue;
        counts[cleaned] = (counts[cleaned] ?? 0) + 1;
      }
      if (counts.isEmpty) return null;
      return counts.entries.reduce((a, b) => a.value >= b.value ? a : b).key;
    }

    final helpfulActions = events.where((event) => event.feltBetter == true);
    return {
      'status': 'success',
      'alerts': events.length,
      'answered_alerts': answered.length,
      'confirmation_rate': answered.isEmpty ? null : confirmed / answered.length,
      'common_activity': mostCommon(events.map((event) => event.activity)),
      'most_effective_action': mostCommon(
        helpfulActions.map(
          (event) => event.alternativeAction ?? event.intervention,
        ),
      ),
    };
  }

  static Future<void> recordConfirmation(String eventId, bool confirmed) async {
    final event = await getEvent(eventId);
    if (event == null) return;
    event.confirmedAnxious = confirmed;
    await _upsertEvent(event);
    await _upload(event);
  }

  static Future<void> recordContext(String eventId, String activity) async {
    final event = await getEvent(eventId);
    if (event == null) return;
    event.activity = activity;
    await _upsertEvent(event);
    await _upload(event);
  }

  Future<void> recordIntervention({
    required String eventId,
    required bool completedGuidance,
    String? alternativeAction,
  }) async {
    final event = await getEvent(eventId);
    if (event == null) return;
    event.intervention = completedGuidance ? '2-minute paced breathing' : null;
    event.interventionAt = DateTime.now();
    event.interventionCompleted = completedGuidance;
    event.alternativeAction = (alternativeAction?.trim().isEmpty ?? true)
        ? null
        : alternativeAction!.trim();
    await _upsertEvent(event);
    await _upload(event);
    _scheduleFollowup(eventId, const Duration(minutes: 5));
  }

  Future<void> _restorePendingFollowups() async {
    final events = await _loadEvents();
    final now = DateTime.now();
    for (final event in events) {
      if (event.userId != _userId ||
          event.interventionAt == null ||
          event.followupAt != null) {
        continue;
      }
      final dueAt = event.interventionAt!.add(const Duration(minutes: 5));
      final remaining = dueAt.difference(now);
      _scheduleFollowup(
        event.eventId,
        remaining.isNegative ? Duration.zero : remaining,
      );
    }
  }

  void _scheduleFollowup(String eventId, Duration delay) {
    _followupTimers[eventId]?.cancel();
    _followupTimers[eventId] = Timer(delay, () => _captureFollowup(eventId));
  }

  Future<void> _captureFollowup(String eventId) async {
    _followupTimers.remove(eventId);
    final event = await getEvent(eventId);
    if (event == null) return;
    final reading = ChestStrapService().hasLiveWornReading
        ? ChestStrapService().lastReading
        : null;
    event.followupAt = DateTime.now();
    if (reading != null && reading.isWorn) {
      event.followupRiskScore = reading.riskScore;
      event.followupHr = reading.meanHR;
      event.followupBr = reading.meanBR;
      event.followupMotion = reading.stdAccMag;
    }
    await _upsertEvent(event);
    await _upload(event);
    await NotificationHelper.showAnxietyFollowup(
      eventId: eventId,
      signalsImproved:
          event.followupRiskScore != null &&
          (event.predictedRiskScore == null
              ? event.followupRiskScore! <= event.initialRiskScore - 10.0
              : event.followupRiskScore! <=
                    event.predictedRiskScore! - 10.0),
    );
  }

  static Future<void> recordFeltBetter(String eventId, bool feltBetter) async {
    final event = await getEvent(eventId);
    if (event == null) return;
    event.feltBetter = feltBetter;
    await _upsertEvent(event);
    await _upload(event);
  }

  static Future<void> handleNotificationAction({
    required String? actionId,
    required String? payload,
  }) async {
    if (payload == null || !payload.startsWith('anxiety_checkin:')) return;
    final eventId = payload.substring('anxiety_checkin:'.length);
    if (actionId == NotificationHelper.anxietyYesAction) {
      await recordConfirmation(eventId, true);
    } else if (actionId == NotificationHelper.anxietyNoAction) {
      await recordConfirmation(eventId, false);
    }
  }
}
