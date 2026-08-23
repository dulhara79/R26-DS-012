// test/widget_test.dart
//
// The generated test referenced `R26App`, which no longer exists — the root
// widget is `AnxietyConsoleApp`. These two tests cover the boot path and the
// one piece of logic most likely to be quietly wrong: fusion renormalisation
// when a modality has no reading.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/core/design/tokens.dart';
import 'package:r26_ds012_app/main.dart';

void main() {
  testWidgets('boots to sign-in when there is no session', (tester) async {
    await tester.pumpWidget(const ClinAnxApp(consented: false, signedIn: false));
    await tester.pump();
    expect(find.text('Sign in'), findsOneWidget);
  });

  test('fusion renormalises across reporting modalities only', () {
    final result = FusionResult.local(
      mrn: 'TEST-001',
      readings: {
        'c1_physiological': const ModalityReading(key: 'c1_physiological'),
        'c2_behavioral':
            const ModalityReading(key: 'c2_behavioral', score: 0.50),
        'c3_intervention':
            const ModalityReading(key: 'c3_intervention', score: 0.50),
        'c4_clinical_nlp':
            const ModalityReading(key: 'c4_clinical_nlp', score: 0.50),
      },
    );

    // Wearable is silent, so 0.20 + 0.15 + 0.40 = 0.75 rescales to 1.0.
    // Every reporting modality scored 0.50, so the composite must be 0.50 —
    // not 0.375, which is what an un-renormalised sum would give.
    expect(result.renormalised, isTrue);
    expect(result.modalitiesAvailable, 3);
    expect(result.compositeScore, closeTo(0.50, 1e-9));
    expect(result.band, AlertBand.red);
    expect(result.computedLocally, isTrue);
  });

  test('an unweighted phrase is not presented as attention', () {
    final withWeights = AttentionSpan.fromJson(
        {'text': 'difficulty controlling the worry', 'weight': 0.19});
    final withoutWeights = AttentionSpan.fromJson({'text': 'persistent worry'});

    expect(withWeights.hasWeight, isTrue);
    expect(withoutWeights.hasWeight, isFalse);
  });
}
