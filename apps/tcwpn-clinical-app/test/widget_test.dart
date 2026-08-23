// test/widget_test.dart
//
// The old `fusion renormalises across reporting modalities only` test exercised
// FusionResult.local(...), the client-side fusion fallback. That path was
// deleted when the app moved to the Central Backend: the server derives its
// weights from each component's validation AUROC above chance and renormalises
// per assessment, so a second weight vector living in the client could only ever
// disagree with it.
//
// What replaces it is the logic that is now most likely to be quietly wrong:
// PARSING. Every number the clinician sees arrives as JSON, and two of the
// parser's decisions are safety properties rather than conveniences.
//
// The fixtures below are REAL responses captured from the backend
// (GET /v1/doctor/patients/{subject_id}/timeline), not hand-written shapes. A
// hand-written fixture would have agreed with whatever the parser happened to
// do; these agree with what the server actually sends.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/core/design/tokens.dart';
import 'package:r26_ds012_app/main.dart';

/// Gate BLOCKED: one usable modality, so no composite was produced.
const String _blockedTimeline = '''
{
  "subject_id": "b40cd9ba-c1e6-4f82-9cfd-41a2548de884",
  "fusion_result_id": 1,
  "composite": null,
  "tier": null,
  "band": "GREY",
  "confidence": 0.0,
  "reason": "insufficient evidence: 1 usable modality, need 2",
  "renormalised": true,
  "modalities_used": 1,
  "weights": {},
  "contributions": {},
  "gate": {
    "passed": false,
    "usable_modalities": ["c3_clinical_nlp"],
    "rejected": {
      "c2_behavioral": "excluded by pre-registered rule (did not clear permutation null)"
    },
    "reason": "insufficient evidence: 1 usable modality, need 2"
  },
  "conformal": null,
  "modalities": {
    "c1_physiological": {"status": "absent", "score": null},
    "c2_behavioral": {"status": "absent", "score": null},
    "c3_clinical_nlp": {
      "status": "ok", "score": 0.71, "confidence": 0.087, "coverage": 1.0,
      "captured_at": "2026-08-23T08:41:50.550863+00:00",
      "age_minutes": 0.0, "fresh": true,
      "model_version": "TC-WPN-v1.0", "excluded": false
    },
    "c4_demographic": {"status": "absent", "score": null}
  },
  "updated_at": "2026-08-23T08:41:50.560000+00:00",
  "trend": []
}
''';

/// Gate PASSED: notes + demographic prior fused.
const String _scoredTimeline = '''
{
  "subject_id": "b40cd9ba-c1e6-4f82-9cfd-41a2548de884",
  "fusion_result_id": 2,
  "composite": 0.8878,
  "tier": "High",
  "band": "RED",
  "confidence": 0.5235,
  "reason": null,
  "renormalised": true,
  "modalities_used": 2,
  "weights": {
    "c1_physiological": 0.0, "c2_behavioral": 0.0,
    "c3_clinical_nlp": 0.65, "c4_demographic": 0.35
  },
  "contributions": {
    "c1_physiological": 0.0, "c2_behavioral": 0.0,
    "c3_clinical_nlp": 0.5428, "c4_demographic": 0.345
  },
  "gate": {
    "passed": true,
    "usable_modalities": ["c3_clinical_nlp", "c4_demographic"],
    "rejected": {}
  },
  "conformal": {"prediction_set": ["Medium", "High"], "calibrated": false},
  "modalities": {
    "c1_physiological": {"status": "absent", "score": null},
    "c2_behavioral": {"status": "absent", "score": null},
    "c3_clinical_nlp": {
      "status": "ok", "score": 0.71, "confidence": 0.087, "coverage": 1.0,
      "captured_at": "2026-08-23T08:41:50.550863+00:00",
      "age_minutes": 0.0, "fresh": true,
      "model_version": "TC-WPN-v1.0", "excluded": false
    },
    "c4_demographic": {
      "status": "ok", "score": 0.34, "confidence": 0.7, "coverage": 1.0,
      "captured_at": "2026-08-23T08:41:50.539281+00:00",
      "age_minutes": 0.0, "fresh": true,
      "model_version": "dcar", "excluded": false
    }
  },
  "updated_at": "2026-08-23T08:41:50.603659+00:00",
  "trend": [
    {"composite": null, "tier": null, "band": "GREY",
     "computed_at": "2026-08-23T08:41:50.546003+00:00", "trigger": "note-ingest"},
    {"composite": 0.8878, "tier": "High", "band": "RED",
     "computed_at": "2026-08-23T08:41:50.603659+00:00", "trigger": "contextual-ingest"}
  ]
}
''';

FusionResult _parse(String raw) =>
    FusionResult.fromJson(jsonDecode(raw) as Map<String, dynamic>, 'TEST-001');

void main() {
  testWidgets('boots to sign-in when there is no session', (tester) async {
    await tester
        .pumpWidget(const ClinAnxApp(consented: false, signedIn: false));
    await tester.pump();
    expect(find.text('Sign in'), findsOneWidget);
  });

  group('a blocked assessment is never presented as low risk', () {
    // The two defaults this group pins were, before the backend integration,
    // `composite_score` defaulting to 0 and `fromWire` falling through to GREEN.
    // Together they rendered an assessment the server REFUSED TO MAKE as
    // "Stable · 0.000". That is the single most dangerous thing this parser
    // could do, so it gets its own group.

    test('a null composite stays null rather than becoming 0.0', () {
      final r = _parse(_blockedTimeline);
      expect(r.compositeScore, isNull);
      expect(r.hasComposite, isFalse);
      expect(r.blocked, isTrue);
    });

    test('GREY parses as grey, not green', () {
      final r = _parse(_blockedTimeline);
      expect(r.band, AlertBand.grey);
      expect(r.band.isScored, isFalse);
    });

    test('an unrecognised band falls back to grey, not green', () {
      // If we cannot tell what the server meant, "no assessment" is honest and
      // "stable" is a guess with a clinical consequence.
      expect(AlertBandX.fromWire('SOMETHING_NEW'), AlertBand.grey);
      expect(AlertBandX.fromWire(null), AlertBand.grey);
      expect(AlertBandX.fromWire(''), AlertBand.grey);
    });

    test('the composite renders as an em dash, never as a number', () {
      expect(_parse(_blockedTimeline).compositeLabel, '\u2014');
    });

    test('grey is excluded from the severity scale', () {
      expect(AlertBandX.scored, isNot(contains(AlertBand.grey)));
      expect(AlertBandX.scored.length, 4);
    });

    test('the gate reason survives so the clinician learns WHY', () {
      final r = _parse(_blockedTimeline);
      expect(r.reason, contains('insufficient evidence'));
      expect(r.usableModalities, ['c3_clinical_nlp']);
      expect(r.rejected.keys, contains('c2_behavioral'));
    });

    test('a blocked row still carries an id, so a verdict can be attached', () {
      expect(_parse(_blockedTimeline).fusionResultId, 1);
    });
  });

  group('modality keys match the backend, not the paper', () {
    // The app previously used `c4_clinical_nlp` and `c3_intervention` — 3 and 4
    // swapped relative to the backend, plus a modality the backend does not
    // have. Against a real response that produced four contributions with
    // weight 0 and score null: the composite rendered, the breakdown was empty.
    // This group fails loudly if that regresses.

    test('TC-WPN is keyed c3_clinical_nlp and carries real weight', () {
      final r = _parse(_scoredTimeline);
      final tcwpn =
          r.contributions.firstWhere((c) => c.key == Modality.c3ClinicalNlp);
      expect(tcwpn.weight, closeTo(0.65, 1e-9));
      expect(tcwpn.score, closeTo(0.71, 1e-9));
      expect(tcwpn.available, isTrue);
    });

    test('no contribution is silently zeroed by a key mismatch', () {
      final r = _parse(_scoredTimeline);
      final weighted = r.contributions.where((c) => c.weight > 0).length;
      expect(weighted, greaterThanOrEqualTo(2),
          reason: 'every weight zero means the wire keys no longer match');
    });

    test('all four backend modalities are represented', () {
      final r = _parse(_scoredTimeline);
      expect(r.contributions.map((c) => c.key).toList(), Modality.all);
    });

    test('the server contribution is used, not weight x score', () {
      // 0.71 * 0.65 = 0.4615, but the server says 0.5428 because it harmonises
      // the raw score before weighting. Recomputing locally would silently
      // disagree with the composite printed beside it.
      final r = _parse(_scoredTimeline);
      final tcwpn =
          r.contributions.firstWhere((c) => c.key == Modality.c3ClinicalNlp);
      expect(tcwpn.contribution, closeTo(0.5428, 1e-9));
    });
  });

  group('a scored assessment parses faithfully', () {
    test('composite, tier and band come from the server', () {
      final r = _parse(_scoredTimeline);
      expect(r.compositeScore, closeTo(0.8878, 1e-9));
      expect(r.tier, 'High');
      expect(r.band, AlertBand.red);
      expect(r.modalitiesUsed, 2);
      expect(r.renormalised, isTrue);
    });

    test('the band is NOT re-derived from the composite', () {
      // The fusion service bands into three tiers at 0.33/0.66; the local
      // helper splits four ways at 0.25/0.50/0.75. At 0.8878 the local helper
      // says DARK RED while the server says RED. The server wins.
      final r = _parse(_scoredTimeline);
      expect(AlertBandX.fromScore(0.8878), AlertBand.darkRed);
      expect(r.band, AlertBand.red);
    });

    test('the trend keeps blocked points as blocked', () {
      final r = _parse(_scoredTimeline);
      expect(r.trend.length, 2);
      expect(r.trend.first.composite, isNull);
      expect(r.trend.first.band, AlertBand.grey);
      expect(r.trend.last.band, AlertBand.red);
    });

    test('a round trip through the cache preserves the safety fields', () {
      // stores.dart caches via toJson and reads back via fromJson, so the cached
      // and network paths cannot diverge in how a field is read.
      final original = _parse(_blockedTimeline);
      final restored = FusionResult.fromJson(
          jsonDecode(jsonEncode(original.toJson())) as Map<String, dynamic>,
          'TEST-001');
      expect(restored.compositeScore, isNull);
      expect(restored.band, AlertBand.grey);
      expect(restored.fusionResultId, 1);
      expect(restored.reason, original.reason);
    });
  });

  group('modality status is reported, not collapsed to a boolean', () {
    test('an absent reading is not available and explains itself', () {
      final r = _parse(_scoredTimeline);
      final c1 =
          r.contributions.firstWhere((c) => c.key == Modality.c1Physiological);
      expect(c1.available, isFalse);
      expect(c1.score, isNull);
      expect(c1.note, isNotNull);
    });

    test('a non-ok status with a score is still not usable evidence', () {
      // gate.py rejects anything that is not exactly "ok". The client applies
      // the same test, so a warming_up reading that happens to carry a number
      // never enters the breakdown as though it counted.
      const warming = ModalityReading(
          key: Modality.c1Physiological, status: 'warming_up', score: 0.62);
      expect(warming.available, isFalse);
      expect(warming.unavailableReason, contains('baseline'));
    });

    test(
        'no_support_set is distinguishable, because the fix is clinician action',
        () {
      const none = ModalityReading(
          key: Modality.c3ClinicalNlp, status: 'no_support_set');
      expect(none.available, isFalse);
      expect(none.unavailableReason, contains('support notes'));
    });
  });

  test('a reading captured after the fusion row is flagged, not shown as zero',
      () {
    // Physiological ingests are debounced (AUTO_FUSION_DEBOUNCE_MIN, default 5
    // minutes) because the stream arrives every 60 seconds. So a wearable
    // reading can sit in `modalities` as ok and fresh while `weights` gives it
    // zero, purely because the composite predates it. Rendering that as
    // "weight 0.00" would say the wearable was considered and discounted; it
    // has not been considered at all yet.
    final j = jsonDecode(_scoredTimeline) as Map<String, dynamic>;
    (j['modalities'] as Map)['c1_physiological'] = {
      'status': 'ok',
      'score': 0.418,
      'confidence': 0.8,
      'coverage': 0.81,
      // ten minutes AFTER updated_at
      'captured_at': '2026-08-23T08:51:50.603659+00:00',
      'age_minutes': 0.5,
      'fresh': true,
      'model_version': 'c1-lstmae-v1.2.0',
      'excluded': false,
    };
    final r = FusionResult.fromJson(j, 'TEST-001');
    final c1 =
        r.contributions.firstWhere((c) => c.key == Modality.c1Physiological);

    expect(c1.capturedAfterFusion, isTrue);
    expect(c1.pendingNextFusion, isTrue);
    expect(c1.note, contains('after this assessment'));
    expect(r.pendingReadings.map((c) => c.key), [Modality.c1Physiological]);
  });

  test('an offset-less server timestamp is read as UTC, not local', () {
    // SQLite drops tzinfo on round-trip, so a backend older than the timestamp
    // fix — and any cache written by an earlier build of this app — returns
    // `updated_at` without an offset. Dart's DateTime.parse reads that as LOCAL
    // time, which on a device at UTC+5:30 puts the composite 5.5 hours away
    // from the readings in the same payload.
    final naive = jsonDecode(_scoredTimeline) as Map<String, dynamic>;
    naive['updated_at'] = '2026-08-23T08:41:50.603659'; // no offset

    final aware = _parse(_scoredTimeline);
    final parsed = FusionResult.fromJson(naive, 'TEST-001');

    expect(parsed.updatedAt!.toUtc(), aware.updatedAt!.toUtc());
  });

  test('an unweighted phrase is not presented as attention', () {
    final withWeights = AttentionSpan.fromJson(
        {'text': 'difficulty controlling the worry', 'weight': 0.19});
    final withoutWeights = AttentionSpan.fromJson({'text': 'persistent worry'});

    expect(withWeights.hasWeight, isTrue);
    expect(withoutWeights.hasWeight, isFalse);
  });
}
