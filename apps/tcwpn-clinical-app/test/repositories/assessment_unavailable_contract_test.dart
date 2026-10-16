import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/central_backend_repositories.dart';
import 'package:r26_ds012_app/domain/contracts/contract_enums.dart';

/// Synthetic, patient-free examples matching the backend's _assessment_wire.
Map<String, dynamic> _unavailablePayload({
  Object? fusionResultId = 125,
  Object? currentAssessment,
  String subjectId = 'subject-test',
}) => {
  'subject_id': subjectId,
  'fusion_result_id': fusionResultId,
  'current_assessment': currentAssessment,
  'forecast': null,
  'confidence': null,
  'assessment_status': 'unavailable',
  'modalities': <Map<String, dynamic>>[],
  'computed_at': fusionResultId == null ? null : '2026-09-16T08:30:00Z',
  'model_version': fusionResultId == null ? null : 'ragf-v0.4',
};

CentralBackendAssessmentRepository _repository(Map<String, dynamic> response) =>
    CentralBackendAssessmentRepository(
      ApiClient(
        'https://backend.example',
        bearer: () => 'synthetic-test-token',
        client: MockClient((request) async {
          expect(request.method, 'GET');
          expect(request.url.path, '/v1/patients/subject-test/assessment/latest');
          expect(request.headers['Authorization'], 'Bearer synthetic-test-token');
          return http.Response(
            jsonEncode(response),
            200,
            headers: {'content-type': 'application/json'},
          );
        }),
      ),
    );

void main() {
  test('historical fusion ID plus unavailable current score is valid, never risk', () async {
    final result = await _repository(_unavailablePayload()).latestAssessment('subject-test');
    expect(result, isNotNull);
    expect(result!.assessmentStatus, AssessmentStatus.unavailable);
    expect(result.fusionResultId, 125); // historical provenance, not live risk
    expect(result.currentAssessment.score, isNull);
    expect(result.currentAssessment.tier, RiskTier.unknown);
    expect(result.currentAssessment.band, isNull);
  });

  test('newly paired subject with no fusion result is unavailable, not malformed', () async {
    final result = await _repository(_unavailablePayload(fusionResultId: null))
        .latestAssessment('subject-test');
    expect(result, isNotNull);
    expect(result!.assessmentStatus, AssessmentStatus.unavailable);
    expect(result.fusionResultId, isNull);
    expect(result.currentAssessment.score, isNull);
  });

  test('unavailable result with an active score is still rejected', () async {
    final source = _unavailablePayload(
      currentAssessment: {'score': 0.6, 'tier': 'Medium', 'band': 'AMBER'},
    );
    await expectLater(
      _repository(source).latestAssessment('subject-test'),
      throwsA(isA<ApiException>()
          .having((error) => error.kind, 'kind', ApiFailure.malformed)),
    );
  });

  test('unavailable result with a positive-looking band but no score is rejected', () async {
    final source = _unavailablePayload(
      currentAssessment: {'score': null, 'tier': null, 'band': 'GREY'},
    );
    await expectLater(
      _repository(source).latestAssessment('subject-test'),
      throwsA(isA<ApiException>()
          .having((error) => error.kind, 'kind', ApiFailure.malformed)),
    );
  });

  test('invalid historical fusion identifier is rejected', () async {
    await expectLater(
      _repository(_unavailablePayload(fusionResultId: -1))
          .latestAssessment('subject-test'),
      throwsA(isA<ApiException>()
          .having((error) => error.kind, 'kind', ApiFailure.malformed)),
    );
  });

  test('response cannot silently change the requested patient identity', () async {
    await expectLater(
      _repository(_unavailablePayload(subjectId: 'different-subject'))
          .latestAssessment('subject-test'),
      throwsA(isA<ApiException>()
          .having((error) => error.kind, 'kind', ApiFailure.malformed)),
    );
  });
}
