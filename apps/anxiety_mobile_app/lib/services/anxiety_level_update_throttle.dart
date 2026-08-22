class AnxietyLevelUpdate {
  final String fromLevel;
  final String toLevel;

  const AnxietyLevelUpdate({
    required this.fromLevel,
    required this.toLevel,
  });

  String get message =>
      'Your anxiety level changed from $fromLevel to $toLevel.';
}

/// Limits in-app level-change alerts while preserving the newest change.
///
/// The first real change is delivered immediately. Further changes within the
/// cooldown are combined into one pending update, which can be flushed when
/// the cooldown ends. Transitions to or from an unavailable reading are not
/// presented as anxiety changes.
class AnxietyLevelUpdateThrottle {
  static const Set<String> _validLevels = {
    'Low',
    'Moderate',
    'Elevated',
    'High',
  };

  final Duration minimumInterval;
  String? _observedLevel;
  String? _lastDeliveredLevel;
  String? _pendingLevel;
  DateTime? _lastDeliveredAt;

  AnxietyLevelUpdateThrottle({
    this.minimumInterval = const Duration(minutes: 1),
  });

  void seed(String level) {
    if (!_validLevels.contains(level)) return;
    _observedLevel = level;
    _lastDeliveredLevel = level;
  }

  AnxietyLevelUpdate? observe(String level, DateTime observedAt) {
    if (!_validLevels.contains(level)) {
      _observedLevel = null;
      _pendingLevel = null;
      return null;
    }

    final previous = _observedLevel;
    _observedLevel = level;
    if (previous == null) {
      _lastDeliveredLevel ??= level;
      return null;
    }
    if (previous == level) return null;

    final lastAt = _lastDeliveredAt;
    if (lastAt == null || observedAt.difference(lastAt) >= minimumInterval) {
      return _deliver(previous, level, observedAt);
    }

    _pendingLevel = level;
    return null;
  }

  AnxietyLevelUpdate? flush(DateTime now) {
    final pending = _pendingLevel;
    if (pending == null) return null;

    final lastAt = _lastDeliveredAt;
    if (lastAt != null && now.difference(lastAt) < minimumInterval) {
      return null;
    }

    _pendingLevel = null;
    final from = _lastDeliveredLevel;
    final to = _observedLevel;
    if (from == null || to == null || from == to) return null;
    return _deliver(from, to, now);
  }

  Duration? delayUntilFlush(DateTime now) {
    if (_pendingLevel == null) return null;
    final lastAt = _lastDeliveredAt;
    if (lastAt == null) return Duration.zero;
    final remaining = minimumInterval - now.difference(lastAt);
    return remaining.isNegative ? Duration.zero : remaining;
  }

  AnxietyLevelUpdate _deliver(
    String from,
    String to,
    DateTime deliveredAt,
  ) {
    _lastDeliveredAt = deliveredAt;
    _lastDeliveredLevel = to;
    _pendingLevel = null;
    return AnxietyLevelUpdate(fromLevel: from, toLevel: to);
  }
}
