import '../evidence.dart';

abstract interface class EvidenceRepository {
  Future<EvidenceResult> ask({
    required String subjectId,
    required String question,
  });
}
