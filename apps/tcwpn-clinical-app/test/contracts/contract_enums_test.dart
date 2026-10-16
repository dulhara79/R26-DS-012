import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

void main() {
  group('RiskTier', () {
    test('parses the documented Low/Medium/High values', () {
      expect(RiskTier.fromWire('Low'), RiskTier.low);
      expect(RiskTier.fromWire('Medium'), RiskTier.medium);
      expect(RiskTier.fromWire('High'), RiskTier.high);
    });

    test('unknown risk tiers do not become low', () {
      expect(RiskTier.fromWire('SEVERE_V2'), RiskTier.unknown);
      expect(RiskTier.fromWire(null), RiskTier.unknown);
    });
  });

  group('AssessmentStatus', () {
    test('parses the target contract', () {
      expect(AssessmentStatus.fromWire('complete'), AssessmentStatus.complete);
      expect(AssessmentStatus.fromWire('partial'), AssessmentStatus.partial);
      expect(
        AssessmentStatus.fromWire('unavailable'),
        AssessmentStatus.unavailable,
      );
    });

    test('maps current backend compatibility terms without changing meaning', () {
      expect(
        AssessmentStatus.fromWire('provisional'),
        AssessmentStatus.partial,
      );
      expect(
        AssessmentStatus.fromWire('insufficient'),
        AssessmentStatus.unavailable,
      );
    });

    test('unknown assessment state stays unknown', () {
      expect(
        AssessmentStatus.fromWire('future_state'),
        AssessmentStatus.unknown,
      );
    });
  });

  test('forecast scope is explicit', () {
    expect(
      ForecastScope.fromWire('physiological'),
      ForecastScope.physiological,
    );
    expect(ForecastScope.fromWire('multimodal'), ForecastScope.multimodal);
    expect(ForecastScope.fromWire('sensor_fusion_v2'), ForecastScope.unknown);
  });

  test('modality state preserves unavailable/experimental/error semantics', () {
    expect(ModalityState.fromWire('ok'), ModalityState.ok);
    expect(ModalityState.fromWire('stale'), ModalityState.stale);
    expect(ModalityState.fromWire('poor_signal'), ModalityState.stale);
    expect(ModalityState.fromWire('warming_up'), ModalityState.unavailable);
    expect(ModalityState.fromWire('insufficient_data'), ModalityState.unavailable);
    expect(
      ModalityState.fromWire('not_validated'),
      ModalityState.notValidated,
    );
    expect(ModalityState.fromWire('absent'), ModalityState.unavailable);
    expect(ModalityState.fromWire('error'), ModalityState.error);
    expect(ModalityState.fromWire('degraded_v2'), ModalityState.unknown);
  });

  test('attention lifecycle only accepts documented states', () {
    expect(AttentionEventStatus.fromWire('OPEN'), AttentionEventStatus.open);
    expect(
      AttentionEventStatus.fromWire('ACKNOWLEDGED'),
      AttentionEventStatus.acknowledged,
    );
    expect(
      AttentionEventStatus.fromWire('RESOLVED'),
      AttentionEventStatus.resolved,
    );
    expect(
      AttentionEventStatus.fromWire('SUPPRESSED'),
      AttentionEventStatus.unknown,
    );
  });

  test('attention severity parses verified Phase 4 policy values', () {
    expect(AttentionSeverity.fromWire('elevated'), AttentionSeverity.elevated);
    expect(AttentionSeverity.fromWire('high'), AttentionSeverity.high);
    expect(AttentionSeverity.fromWire('critical'), AttentionSeverity.unknown);
  });
}
