import '../../core/config/env.dart';
import '../../domain/repositories/assignment_invite_repository.dart';
import '../api/api_client.dart';
import '../api/session.dart';

class CentralBackendAssignmentInviteRepository
    implements AssignmentInviteRepository {
  CentralBackendAssignmentInviteRepository([ApiClient? api])
    : _api = api ?? ApiClient(Env.backendBase);

  final ApiClient _api;

  @override
  Future<AssignedSubject> redeem(String inviteCode) async {
    final code = inviteCode.trim();
    if (code.isEmpty || code.length > 128) {
      throw ArgumentError.value(
        inviteCode,
        'inviteCode',
        'must be 1–128 characters',
      );
    }
    const path = '/v1/clinicians/me/assignments';
    final payload = await _api.post(path, {
      'invite_code': code,
    }, timeout: Env.quickTimeout);
    final clinicianId = payload['clinician_id'];
    final subjectId = payload['subject_id'];
    if (clinicianId is! String ||
        clinicianId.trim().isEmpty ||
        (Session.clinicianId != null && clinicianId != Session.clinicianId) ||
        subjectId is! String ||
        subjectId.trim().isEmpty ||
        payload['active'] != true) {
      throw const ApiException(
        kind: ApiFailure.malformed,
        endpoint: path,
        detail: 'Assignment response did not confirm the clinician, subject and active state.',
      );
    }
    return AssignedSubject(subjectId.trim());
  }
}
