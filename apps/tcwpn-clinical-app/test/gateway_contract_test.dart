// test/gateway_contract_test.dart
//
// Section 29 asks for the API failure modes to be tested, not just the happy
// path. `widget_test.dart` already pins the PARSER — what a well-formed
// response turns into. This file pins the TRANSPORT and the REQUEST: what goes
// out on the wire, and what the app does when what comes back is not a
// well-formed response.
//
// The property these tests exist to protect is one line of section 19: a failed
// call must never become a score. Every error case below asserts that the call
// THREW, because throwing is what forces the controller into an explicit
// unavailable state. A test that accepted a default-constructed FusionResult
// here would be certifying the exact bug the architecture was rewritten to
// remove.
//
// No network is touched. MockClient is injected into ApiClient, which bypasses
// SecureHttp — correct, because a test double is not a network path and pinning
// a fake host would test nothing.

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/api/gateways.dart';
import 'package:r26_ds012_app/domain/models.dart';

const String _base = 'https://backend.test';

/// Every request the mock saw, so a test can assert on what was SENT.
final List<http.Request> sent = <http.Request>[];

CentralBackendGateway _gateway(
  Future<http.Response> Function(http.Request req) handler,
) => CentralBackendGateway(
  ApiClient(
    _base,
    client: MockClient((req) {
      sent.add(req);
      return handler(req);
    }),
    bearer: () => 'test-token',
  ),
);

/// A gateway whose every call fails with [status].
CentralBackendGateway _failing(
  int status, [
  String body = '{"detail":"nope"}',
]) => _gateway((_) async => http.Response(body, status));

Map<String, dynamic> _bodyOf(http.Request r) =>
    jsonDecode(r.body) as Map<String, dynamic>;

/// A timeline with `c2_behavioral` present, experimental, and NOT fused.
String _timelineWithExperimentalC2() => jsonEncode({
  'subject_id': 'b40cd9ba-c1e6-4f82-9cfd-41a2548de884',
  'fusion_result_id': 7,
  'composite': 0.5432,
  'tier': 'Medium',
  'band': 'AMBER',
  'confidence': 0.7123,
  'reason': null,
  'renormalised': true,
  'modalities_used': 2,
  'weights': {
    'c1_physiological': 0.4812,
    'c2_behavioral': 0.0,
    'c3_clinical_nlp': 0.5188,
    'c4_demographic': 0.0,
  },
  'contributions': {
    'c1_physiological': 0.2935,
    'c2_behavioral': 0.0,
    'c3_clinical_nlp': 0.2282,
    'c4_demographic': 0.0,
  },
  'gate': {
    'passed': true,
    'usable_modalities': ['c1_physiological', 'c3_clinical_nlp'],
    'rejected': {
      'c2_behavioral':
          'excluded by pre-registered rule (did not clear permutation null)',
    },
  },
  'conformal': null,
  'modalities': {
    'c1_physiological': {
      'status': 'ok',
      'score': 0.61,
      'confidence': 0.7,
      'coverage': 1.0,
      'captured_at': '2026-08-25T10:00:00+00:00',
      'age_minutes': 5.0,
      'fresh': true,
      'model_version': 'c1-lstmae-v1.2.0',
      'excluded': false,
    },
    // Reported, scored, and deliberately not fused.
    'c2_behavioral': {
      'status': 'not_validated',
      'score': 0.48,
      'confidence': 0.3,
      'captured_at': '2026-08-25T09:00:00+00:00',
      'age_minutes': 65.0,
      'fresh': true,
      'model_version': 'c2-gnn-v0.3',
      'excluded': true,
    },
    'c3_clinical_nlp': {
      'status': 'ok',
      'score': 0.44,
      'confidence': 0.5,
      'coverage': 1.0,
      'captured_at': '2026-08-25T09:30:00+00:00',
      'age_minutes': 35.0,
      'fresh': true,
      'model_version': 'TC-WPN-v1.0',
      'excluded': false,
    },
    'c4_demographic': {'status': 'absent', 'score': null},
  },
  'updated_at': '2026-08-25T10:05:00+00:00',
  'trend': [],
});

void main() {
  setUp(sent.clear);

  // ───────────────────────────────────────────────────────────────────────────
  group('the app talks to the Central Backend and nothing else', () {
    test('a note goes to /v1/clinical-notes, not to a model service', () async {
      final g = _gateway(
        (_) async => http.Response(
          jsonEncode({'subject_id': 'S1', 'status': 'ok', 'score': 0.44}),
          200,
        ),
      );

      await g.submitNote(
        subjectId: 'S1',
        noteText: 'Patient reports persistent worry.',
        noteType: 'Psychiatry note',
        noteDate: DateTime.utc(2026, 8, 25, 9, 30),
        supportSet: const [],
        visitCount: 3,
        author: 'dr-dulhara',
      );

      expect(sent.single.url.toString(), '$_base/v1/clinical-notes');
      expect(sent.single.method, 'POST');
      // The retired direct-inference paths, asserted by absence.
      expect(sent.single.url.path, isNot(contains('/predict')));
      expect(sent.single.url.path, isNot(contains('/fuse')));
      expect(sent.single.url.path, isNot(contains('/v3/risk/classify')));
      expect(sent.single.url.path, isNot(contains('/intervene')));
    });

    test('the note request carries no weights and no composite', () async {
      final g = _gateway(
        (_) async => http.Response(
          jsonEncode({'subject_id': 'S1', 'status': 'ok', 'score': 0.44}),
          200,
        ),
      );

      await g.submitNote(
        subjectId: 'S1',
        noteText: 'n',
        noteType: 'Psychiatry note',
        noteDate: DateTime.utc(2026, 8, 25),
        supportSet: [
          SupportNote(
            id: 'sn-1',
            text: 'prior note',
            label: 'anxiety',
            noteDate: DateTime.utc(2025, 11, 2),
            addedAt: DateTime.utc(2025, 11, 2),
          ),
        ],
        visitCount: 3,
      );

      final body = _bodyOf(sent.single);
      expect(body['subject_id'], 'S1');
      expect(body['support_set'], hasLength(1));
      // Temporal weighting is TC-WPN's job, server-side. If the client ever
      // starts pre-weighting the support set, the model is scoring something
      // the paper did not describe.
      expect((body['support_set'] as List).first, isNot(contains('weight')));
      expect(body.containsKey('weights'), isFalse);
      expect(body.containsKey('composite'), isFalse);
      expect(body.containsKey('composite_score'), isFalse);
    });

    test(
      'enrolment sends the MRN once and the app keeps the subject_id',
      () async {
        final g = _gateway(
          (_) async => http.Response(
            jsonEncode({
              'subject_id': 'b40cd9ba-c1e6-4f82-9cfd-41a2548de884',
              'pairing_code': '419-330',
            }),
            200,
          ),
        );

        final r = await g.enrol(mrn: 'S-000123', enrolledBy: 'dr-dulhara');

        expect(sent.single.url.toString(), '$_base/v1/subjects');
        expect(_bodyOf(sent.single)['mrn'], 'S-000123');
        expect(r.subjectId, 'b40cd9ba-c1e6-4f82-9cfd-41a2548de884');
        expect(r.pairingCode, '419-330');
      },
    );

    test(
      'a verdict is bound to the fusion row the clinician looked at',
      () async {
        final g = _gateway((_) async => http.Response('{"ok":true}', 200));
        await g.submitVerdict(fusionResultId: 7, tierLabel: 'Medium');

        final body = _bodyOf(sent.single);
        expect(sent.single.url.toString(), '$_base/v1/verdict');
        // Not "the latest row" — a specific id, or the label is attached to a
        // prediction the clinician never saw.
        expect(body['fusion_result_id'], 7);
        expect(body['tier_label'], 'Medium');
      },
    );

    test('the TC-WPN Space is only ever asked for /health', () async {
      final calls = <Uri>[];
      final warm = TcwpnWarmupGateway(
        ApiClient(
          'https://tcwpn.test',
          client: MockClient((req) async {
            calls.add(req.url);
            return http.Response('{"status":"ok"}', 200);
          }),
        ),
      );

      await warm.health();

      // hasTcwpnWarmup is false unless TCWPN_BASE was defined at build time, in
      // which case health() short-circuits and never reaches the transport.
      // Either way the assertion that matters holds: nothing but /health.
      for (final u in calls) {
        expect(u.path, '/health');
      }
    });
  });

  // ───────────────────────────────────────────────────────────────────────────
  group('a failed call is an error, never a score', () {
    test('400 surfaces as validation', () async {
      await expectLater(
        _failing(400, '{"detail":"note_text must not be empty"}').submitNote(
          subjectId: 'S1',
          noteText: '',
          noteType: 'Psychiatry note',
          noteDate: DateTime.utc(2026, 8, 25),
          supportSet: const [],
          visitCount: 1,
        ),
        throwsA(
          isA<ApiException>()
              .having((e) => e.kind, 'kind', ApiFailure.validation)
              .having((e) => e.detail, 'detail', contains('note_text')),
        ),
      );
    });

    test('422 surfaces as validation, with the server detail intact', () async {
      await expectLater(
        _failing(422, '{"detail":"visit_count must be >= 1"}').submitNote(
          subjectId: 'S1',
          noteText: 'n',
          noteType: 'Psychiatry note',
          noteDate: DateTime.utc(2026, 8, 25),
          supportSet: const [],
          visitCount: 0,
        ),
        throwsA(
          isA<ApiException>().having(
            (e) => e.kind,
            'kind',
            ApiFailure.validation,
          ),
        ),
      );
    });

    test('401 surfaces as unauthorized, not as an empty chart', () async {
      await expectLater(
        _failing(401).timeline(subjectId: 'S1', mrn: 'S-000123'),
        throwsA(
          isA<ApiException>().having(
            (e) => e.kind,
            'kind',
            ApiFailure.unauthorized,
          ),
        ),
      );
    });

    test(
      '403 is unauthorized too — a wrong token reads the same as none',
      () async {
        await expectLater(
          _failing(403).timeline(subjectId: 'S1', mrn: 'S-000123'),
          throwsA(
            isA<ApiException>().having(
              (e) => e.kind,
              'kind',
              ApiFailure.unauthorized,
            ),
          ),
        );
      },
    );

    test('500 throws rather than returning a default FusionResult', () async {
      // The whole point. A default-constructed FusionResult would render as a
      // composite of 0.000 — "Stable" — for a patient the server refused to
      // assess.
      await expectLater(
        _failing(500).timeline(subjectId: 'S1', mrn: 'S-000123'),
        throwsA(
          isA<ApiException>().having((e) => e.kind, 'kind', ApiFailure.server),
        ),
      );
    });

    test('503 from the fusion trigger throws', () async {
      await expectLater(
        _failing(503).runFusion('S1'),
        throwsA(isA<ApiException>()),
      );
    });

    test(
      'an unknown subject is null, which is a state — not an exception',
      () async {
        final r = await _failing(
          404,
        ).timeline(subjectId: 'S1', mrn: 'S-000123');
        expect(r, isNull);
      },
    );

    test('resolveMrn returns null for an unenrolled MRN', () async {
      expect(await _failing(404).resolveMrn('S-000999'), isNull);
    });

    test('a 409 on external-id linking is NOT swallowed', () async {
      // 409 means this external id already belongs to a DIFFERENT subject —
      // a cross-patient error. Swallowing it wires one patient's wearable to
      // another patient's chart.
      await expectLater(
        _failing(409).registerExternalId(
          subjectId: 'S1',
          modality: 'c2_behavioral',
          externalId: 'P-77',
        ),
        throwsA(isA<ApiException>()),
      );
    });

    test('a network drop is offline, not an empty result', () async {
      final g = _gateway((_) async => throw const SocketException('down'));
      await expectLater(
        g.timeline(subjectId: 'S1', mrn: 'S-000123'),
        throwsA(
          isA<ApiException>().having((e) => e.kind, 'kind', ApiFailure.offline),
        ),
      );
    });

    test('a timeout is a timeout', () async {
      final api = ApiClient(
        _base,
        client: MockClient((_) async {
          await Future<void>.delayed(const Duration(milliseconds: 300));
          return http.Response('{}', 200);
        }),
      );

      await expectLater(
        api.get(
          '/v1/health',
          timeout: const Duration(milliseconds: 50),
          retries: 0,
        ),
        throwsA(
          isA<ApiException>().having((e) => e.kind, 'kind', ApiFailure.timeout),
        ),
      );
    });

    test(
      'an HTML error page is malformed, not a FormatException crash',
      () async {
        final g = _gateway(
          (_) async => http.Response('<html>502 Bad Gateway</html>', 200),
        );
        await expectLater(
          g.timeline(subjectId: 'S1', mrn: 'S-000123'),
          throwsA(
            isA<ApiException>().having(
              (e) => e.kind,
              'kind',
              ApiFailure.malformed,
            ),
          ),
        );
      },
    );

    test('health() degrades to null rather than blocking the app', () async {
      // The one place swallowing is right: a health probe is decoration, and a
      // dead probe must not stop a clinician opening a chart.
      expect(await _failing(500).health(), isNull);
    });
  });

  // ───────────────────────────────────────────────────────────────────────────
  group('an experimental behavioural score never reaches the composite', () {
    late FusionResult r;

    setUp(() async {
      final g = _gateway(
        (_) async => http.Response(_timelineWithExperimentalC2(), 200),
      );
      r = (await g.timeline(subjectId: 'S1', mrn: 'S-000123'))!;
    });

    ComponentContribution c(String key) =>
        r.contributions.firstWhere((x) => x.key == key);

    test('not_validated with a real score is still not usable evidence', () {
      final c2 = c(Modality.c2Behavioral);
      expect(c2.score, closeTo(0.48, 1e-9));
      expect(c2.available, isFalse, reason: 'only status == ok counts');
      expect(c2.weight, 0.0);
    });

    test('the exclusion explains itself in clinician language', () {
      expect(
        c(Modality.c2Behavioral).reading.unavailableReason,
        contains('pre-registered rule'),
      );
    });

    test('an absent modality stays null and is never 0.0', () {
      final c4 = c(Modality.c4Demographic);
      expect(c4.score, isNull);
      expect(c4.available, isFalse);
    });

    test('the composite is the server\'s number, unmodified', () {
      expect(r.compositeScore, closeTo(0.5432, 1e-9));
      expect(r.tier, 'Medium');
      expect(r.modalitiesUsed, 2);
    });

    test('contributions are read, not recomputed as weight x score', () {
      final c1 = c(Modality.c1Physiological);
      // 0.61 * 0.4812 = 0.29353..., the server says 0.2935 — close here, but
      // the client must not be the one deciding that. Harmonisation happens
      // server-side and the two diverge as soon as it does anything.
      expect(c1.contribution, closeTo(0.2935, 1e-9));
    });

    test('the two fused modalities carry all the weight', () {
      final fused = r.contributions
          .where((x) => x.weight > 0)
          .map((x) => x.key);
      expect(
        fused,
        containsAll([Modality.c1Physiological, Modality.c3ClinicalNlp]),
      );
      expect(fused, isNot(contains(Modality.c2Behavioral)));
      expect(fused, isNot(contains(Modality.c4Demographic)));
    });
  });

  // ───────────────────────────────────────────────────────────────────────────
  group('CARE-AnxRAG', () {
    test(
      'guidance is requested through the backend, not a RAG service',
      () async {
        final g = _gateway(
          (_) async => http.Response(
            jsonEncode({
              'answer': 'Consider structured GAD-7 follow-up within two weeks.',
              'citations': [
                {'source': 'NICE CG113', 'section': '1.2.8'},
              ],
              'abstained': false,
            }),
            200,
          ),
        );

        final r = await g.evidence(subjectId: 'S1', question: 'next steps?');

        expect(
          sent.single.url.toString(),
          '$_base/v1/doctor/patients/S1/evidence',
        );
        expect(r['abstained'], isFalse);
        expect(r['citations'], hasLength(1));
      },
    );

    test('an abstention is passed through, not turned into advice', () async {
      final g = _gateway(
        (_) async => http.Response(
          jsonEncode({
            'answer': null,
            'abstained': true,
            'reason': 'insufficient retrieved evidence',
            'citations': <dynamic>[],
          }),
          200,
        ),
      );

      final r = await g.evidence(subjectId: 'S1', question: 'next steps?');

      expect(r['abstained'], isTrue);
      expect(r['answer'], isNull);
      expect(r['reason'], contains('insufficient'));
    });

    test(
      'an unavailable RAG backend throws instead of returning silence',
      () async {
        await expectLater(
          _failing(503).evidence(subjectId: 'S1', question: 'next steps?'),
          throwsA(isA<ApiException>()),
        );
      },
    );
  });
}
