import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/p5c_repositories.dart';
import 'package:r26_ds012_app/domain/contracts/clinician_assessment.dart';
import 'package:r26_ds012_app/domain/evidence.dart';

const _base = 'https://backend.test';

void main() {
  test('patient evidence posts only the question to the verified backend route', () async {
    late http.Request seen;
    final api = ApiClient(
      _base,
      client: MockClient((request) async {
        seen = request;
        return http.Response(
          jsonEncode({
            'available': true,
            'answer': 'Evidence-grounded response',
            'citations': [
              {
                'citation_id': 'c1',
                'title': 'Guideline',
                'source_name': 'Source',
                'excerpt': 'Excerpt',
                'url': 'https://example.test/source',
                'evidence_level': 'guideline',
              }
            ],
            'confidence': 0.81,
            'conflict_score': 0.1,
            'abstained': false,
          }),
          200,
        );
      }),
      bearer: () => 'session-token',
    );
    final repo = CentralBackendEvidenceRepository(api);

    final result = await repo.ask(subjectId: 'subject-1', question: '  What helps?  ');

    expect(seen.method, 'POST');
    expect(seen.url.toString(), '$_base/v1/doctor/patients/subject-1/evidence');
    expect(jsonDecode(seen.body), {'question': 'What helps?'});
    expect(result.state, EvidenceState.answered);
    expect(result.answer, 'Evidence-grounded response');
    expect(result.citations, hasLength(1));
    expect(result.confidence, 0.81);
    expect(result.conflictScore, 0.1);
  });

  test('evidence abstention remains an abstention', () async {
    final api = ApiClient(
      _base,
      client: MockClient((_) async => http.Response(
            jsonEncode({
              'available': true,
              'answer': null,
              'citations': [],
              'abstained': true,
              'abstention_reason': 'insufficient evidence',
            }),
            200,
          )),
    );
    final result = await CentralBackendEvidenceRepository(api)
        .ask(subjectId: 'S1', question: 'Question');

    expect(result.state, EvidenceState.abstained);
    expect(result.answer, isNull);
    expect(result.abstentionReason, 'insufficient evidence');
  });

  test('blank evidence question is rejected before transport', () async {
    var calls = 0;
    final api = ApiClient(
      _base,
      client: MockClient((_) async {
        calls++;
        return http.Response('{}', 200);
      }),
    );

    await expectLater(
      CentralBackendEvidenceRepository(api).ask(subjectId: 'S1', question: '   '),
      throwsArgumentError,
    );
    expect(calls, 0);
  });

  test('clinician assessment binds exact fusion id and verified fields', () async {
    late http.Request seen;
    final api = ApiClient(
      _base,
      client: MockClient((request) async {
        seen = request;
        return http.Response(
          jsonEncode({
            'verdict_id': 8,
            'subject_id': 'subject-1',
            'agrees_with_model': false,
            'calibration_labels_total': 21,
            'conformal_calibrated': false,
          }),
          200,
        );
      }),
    );
    final repo = CentralBackendClinicianAssessmentRepository(api);
    const draft = ClinicianAssessmentDraft(
      fusionResultId: 37,
      tierLabel: 'High',
      author: 'dr-1',
      note: 'Observed during review.',
    );

    final receipt = await repo.submit(draft);

    expect(seen.url.toString(), '$_base/v1/verdict');
    expect(jsonDecode(seen.body), {
      'fusion_result_id': 37,
      'tier_label': 'High',
      'author': 'dr-1',
      'note': 'Observed during review.',
    });
    expect(receipt.verdictId, 8);
    expect(receipt.subjectId, 'subject-1');
    expect(receipt.fusionResultId, 37);
    expect(receipt.tierLabel, 'High');
    expect(receipt.agreesWithModel, isFalse);
  });

  test('verdict omits blank optional fields rather than inventing values', () async {
    late http.Request seen;
    final api = ApiClient(
      _base,
      client: MockClient((request) async {
        seen = request;
        return http.Response('{"verdict_id":9,"subject_id":"S1"}', 200);
      }),
    );

    await CentralBackendClinicianAssessmentRepository(api).submit(
      const ClinicianAssessmentDraft(
        fusionResultId: 38,
        tierLabel: 'Low',
        author: '',
        note: '',
      ),
    );

    final body = jsonDecode(seen.body) as Map<String, dynamic>;
    expect(body.keys, containsAll(<String>['fusion_result_id', 'tier_label']));
    expect(body.containsKey('author'), isFalse);
    expect(body.containsKey('note'), isFalse);
  });
}
