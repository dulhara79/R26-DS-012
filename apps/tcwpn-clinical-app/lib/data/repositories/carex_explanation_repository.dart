import '../../core/config/env.dart';
import '../api/api_client.dart';

/// CARE-X is an explanation of a stored fusion assessment, not a RAG answer.
/// Keep the Central Backend as the sole authority for weights and reasoning.
abstract interface class CarexExplanationRepository {
  Future<Map<String, dynamic>> explanation(String subjectId);
}

class CentralBackendCarexExplanationRepository
    implements CarexExplanationRepository {
  final ApiClient _api;
  final bool _ownsClient;

  CentralBackendCarexExplanationRepository([ApiClient? api])
      : _api = api ?? ApiClient(Env.backendBase),
        _ownsClient = api == null;

  @override
  Future<Map<String, dynamic>> explanation(String subjectId) {
    final id = subjectId.trim();
    if (id.isEmpty) {
      throw ArgumentError.value(subjectId, 'subjectId', 'must not be empty');
    }
    return _api.get(
      '/v1/doctor/patients/${Uri.encodeComponent(id)}/explanation',
      retries: 0,
      timeout: Env.quickTimeout,
    );
  }

  void dispose() {
    if (_ownsClient) _api.close();
  }
}
