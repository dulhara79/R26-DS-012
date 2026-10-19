import '../../domain/contracts/clinician_assessment.dart';
import '../../domain/evidence.dart';
import '../../domain/repositories/clinician_assessment_repository.dart';
import '../../domain/repositories/evidence_repository.dart';
import '../api/api_client.dart';
import '../api/gateways.dart';

class CentralBackendEvidenceRepository implements EvidenceRepository {
  final CentralBackendGateway _gateway;

  CentralBackendEvidenceRepository([ApiClient? api])
      : _gateway = CentralBackendGateway(api);

  @override
  Future<EvidenceResult> ask({
    required String subjectId,
    required String question,
  }) async {
    final json = await _gateway.evidence(
      subjectId: subjectId,
      question: question,
    );
    return EvidenceResult.fromJson(json);
  }

  void dispose() => _gateway.dispose();
}

class CentralBackendClinicianAssessmentRepository
    implements ClinicianAssessmentRepository {
  final CentralBackendGateway _gateway;

  CentralBackendClinicianAssessmentRepository([ApiClient? api])
      : _gateway = CentralBackendGateway(api);

  @override
  Future<ClinicianAssessmentReceipt> submit(
    ClinicianAssessmentDraft draft,
  ) async {
    final json = await _gateway.submitVerdict(
      fusionResultId: draft.fusionResultId,
      tierLabel: draft.tierLabel,
      author: draft.author,
      note: draft.note,
    );
    return ClinicianAssessmentReceipt.fromJson(json, request: draft);
  }

  void dispose() => _gateway.dispose();
}
