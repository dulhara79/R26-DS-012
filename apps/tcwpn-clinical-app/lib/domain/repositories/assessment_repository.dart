import '../contracts/assessment_summary.dart';

abstract interface class AssessmentRepository {
  Future<AssessmentSummary?> latestAssessment(String subjectId);
}
