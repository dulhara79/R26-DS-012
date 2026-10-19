enum RiskTier {
  low,
  medium,
  high,
  unknown;

  static RiskTier fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'low' => low,
        'medium' => medium,
        'high' => high,
        _ => unknown,
      };
}

enum AssessmentStatus {
  complete,
  partial,
  unavailable,
  unknown;

  static AssessmentStatus fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'complete' => complete,
        'partial' || 'provisional' => partial,
        'unavailable' || 'insufficient' => unavailable,
        _ => unknown,
      };
}

enum ForecastScope {
  physiological,
  multimodal,
  unknown;

  static ForecastScope fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'physiological' => physiological,
        'multimodal' => multimodal,
        _ => unknown,
      };
}

enum ModalityState {
  ok,
  stale,
  notValidated,
  unavailable,
  error,
  unknown;

  static ModalityState fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'ok' => ok,
        // The frozen backend contract represents stale evidence as poor_signal.
        'stale' || 'poor_signal' => stale,
        'not_validated' => notValidated,
        'absent' ||
        'unavailable' ||
        'warming_up' ||
        'insufficient_data' ||
        'no_support_set' => unavailable,
        'error' => error,
        _ => unknown,
      };
}

enum AttentionEventStatus {
  open,
  acknowledged,
  resolved,
  unknown;

  static AttentionEventStatus fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toUpperCase()) {
        'OPEN' => open,
        'ACKNOWLEDGED' => acknowledged,
        'RESOLVED' => resolved,
        _ => unknown,
      };
}

enum AttentionSeverity {
  elevated,
  high,
  unknown;

  static AttentionSeverity fromWire(Object? value) =>
      switch ((value ?? '').toString().trim().toLowerCase()) {
        'elevated' => elevated,
        'high' => high,
        _ => unknown,
      };
}
