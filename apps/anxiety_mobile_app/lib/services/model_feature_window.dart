/// A completed non-overlapping firmware minute, separate from live UI values.
class ModelFeatureWindow {
  static const int currentVersion = 2;
  final int version;
  final int startMilliseconds;
  final int endMilliseconds;
  final int status;
  final List<double> features;

  const ModelFeatureWindow({
    required this.version,
    required this.startMilliseconds,
    required this.endMilliseconds,
    required this.status,
    required this.features,
  });

  bool get isReady =>
      version == currentVersion &&
      status == 1 &&
      ((endMilliseconds - startMilliseconds) & 0xffffffff) == 60000 &&
      features.length == 10 &&
      features.every((value) => value.isFinite);

  int ageAt(int packetMilliseconds) =>
      (packetMilliseconds - endMilliseconds) & 0xffffffff;

  factory ModelFeatureWindow.fromCsvFields(List<String> parts) {
    final values = parts.sublist(16, 26).map(double.parse).toList();
    // Preserve the existing sensor temperature correction for both streams.
    values[6] -= 4.0;
    return ModelFeatureWindow(
      version: int.parse(parts[12]),
      startMilliseconds: int.parse(parts[13]),
      endMilliseconds: int.parse(parts[14]),
      status: int.parse(parts[15]),
      features: values,
    );
  }

  Map<String, dynamic> toJson() => {
    'version': version,
    'start_ms': startMilliseconds,
    'end_ms': endMilliseconds,
    'status': status,
    'features': features,
  };

  factory ModelFeatureWindow.fromJson(Map<String, dynamic> json) =>
      ModelFeatureWindow(
        version: json['version'] as int,
        startMilliseconds: json['start_ms'] as int,
        endMilliseconds: json['end_ms'] as int,
        status: json['status'] as int,
        features: (json['features'] as List)
            .map((value) => (value as num).toDouble())
            .toList(),
      );
}

/// Counts only full consecutive firmware minutes acquired after setup starts.
class CalibrationMinuteSequence {
  final int startedAtMilliseconds;
  int? _lastEnd;

  CalibrationMinuteSequence(this.startedAtMilliseconds);

  bool accept(ModelFeatureWindow? window, int packetTimestamp, bool isWorn) {
    if (!isWorn || packetTimestamp < startedAtMilliseconds) {
      throw StateError(
        'The strap was removed or restarted during setup. Please try again.',
      );
    }
    if (window == null || window.version != ModelFeatureWindow.currentVersion) {
      throw StateError(
        'The chest strap is using an incompatible feature format.',
      );
    }
    if (window.startMilliseconds < startedAtMilliseconds ||
        window.endMilliseconds == _lastEnd) {
      return false;
    }
    if (!window.isReady || window.ageAt(packetTimestamp) > 90000) {
      throw StateError(
        'A complete valid sensor minute was not received. Check the strap and retry.',
      );
    }
    if (_lastEnd != null && window.startMilliseconds != _lastEnd) {
      throw StateError(
        'A sensor minute was missed during setup. Check the connection and retry.',
      );
    }
    _lastEnd = window.endMilliseconds;
    return true;
  }
}
