import '../contracts/clinician_assessment.dart';

abstract interface class ClinicianAssessmentRepository {
  Future<ClinicianAssessmentReceipt> submit(
    ClinicianAssessmentDraft draft,
  );
}
