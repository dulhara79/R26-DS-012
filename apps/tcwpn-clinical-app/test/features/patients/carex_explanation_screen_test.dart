import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/data/repositories/carex_explanation_repository.dart';
import 'package:r26_ds012_app/features/patients/carex_explanation_screen.dart';

class _CarexRepository implements CarexExplanationRepository {
  final Map<String, dynamic> response;
  String? queriedSubject;

  _CarexRepository(this.response);

  @override
  Future<Map<String, dynamic>> explanation(String subjectId) async {
    queriedSubject = subjectId;
    return response;
  }
}

Map<String, dynamic> _example() => {
      'subject_id': 'subject-001',
      'fusion_result_id': 123,
      'explainer_version': 'care-x-v1.1',
      'assessed': true,
      'narrative': 'Stored clinical fusion explanation.',
      'attribution': {
        'ranked': [
          {
            'label': 'clinical notes',
            'modality': 'c3_clinical_nlp',
            'contribution': 0.24,
            'share_pct': 48.0,
          },
        ],
      },
      'weight_provenance': {
        'per_modality': [
          {
            'label': 'clinical notes',
            'final_weight': 0.4,
            'reading_confidence': 0.5,
            'coverage': 1.0,
            'drivers': ['confidence is estimated'],
          },
        ],
      },
      'counterfactuals': {
        'drop_one': [
          {
            'label': 'clinical notes',
            'composite_without': 0.48,
            'delta': -0.08,
            'tier_without': 'Medium',
          },
        ],
        'thresholds_authoritative': true,
      },
      'reliability_audit': {
        'findings': [
          {
            'label': 'clinical notes',
            'verdict': 'unquantified',
            'interpretation': 'Insufficient information for weight inflation.',
          },
        ],
      },
      'sufficiency': {
        'minimal_sufficient_size': 2,
        'scope_caveat': 'Only fixed-set score counterfactuals are evaluated.',
      },
      'honesty_ledger': {
        'items': [
          {
            'severity': 'high',
            'issue': 'Placeholder reference distribution',
            'detail': 'Clinical validation remains pending.',
            'affects': 'risk interpretation',
            'resolved_by': 'Use real held-out references',
          },
        ],
      },
    };

Future<void> _render(
  WidgetTester tester,
  _CarexRepository repository, {
  int? fusionId = 123,
}) async {
  await tester.pumpWidget(
    MaterialApp(
      home: CarexExplanationScreen(
        subjectId: 'subject-001',
        fusionResultId: fusionId,
        displayId: 'Patient A',
        repository: repository,
      ),
    ),
  );
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('shows the backend CARE-X explanation and research boundary',
      (tester) async {
    final repo = _CarexRepository(_example());
    await _render(tester, repo);

    expect(repo.queriedSubject, 'subject-001');
    expect(find.text('CARE-X · Why this assessment?'), findsOneWidget);
    expect(find.textContaining('Fusion result #123'), findsOneWidget);
    expect(find.text('Stored clinical fusion explanation.'), findsOneWidget);
    expect(find.textContaining('not CARE-AnxRAG evidence retrieval'),
        findsOneWidget);
    expect(find.textContaining('Contribution: 0.24'), findsOneWidget);
  });

  testWidgets('rejects stale CARE-X explanation from another fusion record',
      (tester) async {
    final updated = _example()..['fusion_result_id'] = 124;
    final repo = _CarexRepository(updated);
    await _render(tester, repo);

    expect(find.text('CARE-X explanation unavailable'), findsOneWidget);
    expect(find.textContaining('assessment changed on the server'),
        findsOneWidget);
    expect(find.text('Stored clinical fusion explanation.'), findsNothing);
  });

  testWidgets('rejects cross-subject CARE-X response', (tester) async {
    final changed = _example()..['subject_id'] = 'other-subject';
    await _render(tester, _CarexRepository(changed));

    expect(find.text('CARE-X explanation unavailable'), findsOneWidget);
    expect(find.text('Stored clinical fusion explanation.'), findsNothing);
  });

  testWidgets('missing fusion identifier cannot trigger evidence lookup',
      (tester) async {
    final repo = _CarexRepository(_example());
    await _render(tester, repo, fusionId: null);

    expect(repo.queriedSubject, isNull);
    expect(find.text('CARE-X explanation unavailable'), findsOneWidget);
    expect(find.textContaining('No fusion result is recorded'), findsOneWidget);
  });

  testWidgets('gate-refused assessment never invents explanations',
      (tester) async {
    final refused = _example()
      ..['assessed'] = false
      ..['refusal_reason'] = 'insufficient usable evidence'
      ..['attribution'] = null;
    await _render(tester, _CarexRepository(refused));

    expect(find.text('No scored assessment'), findsOneWidget);
    expect(find.text('insufficient usable evidence'), findsOneWidget);
    expect(find.textContaining('Contribution: 0.24'), findsNothing);
  });
}
