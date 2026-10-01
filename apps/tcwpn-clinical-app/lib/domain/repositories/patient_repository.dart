import '../contracts/patient_summary.dart';

abstract interface class PatientRepository {
  Future<List<PatientSummary>> assignedPatients();
}
