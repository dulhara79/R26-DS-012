import '../../domain/contracts/dashboard_snapshot.dart';
import '../../domain/repositories/attention_event_repository.dart';
import '../../domain/repositories/dashboard_repository.dart';
import '../../domain/repositories/patient_repository.dart';

class CompositeDashboardRepository implements DashboardRepository {
  final PatientRepository patients;
  final AttentionEventRepository attentionEvents;
  final DateTime Function() now;

  CompositeDashboardRepository({
    required this.patients,
    required this.attentionEvents,
    DateTime Function()? now,
  }) : now = now ?? DateTime.now;

  @override
  Future<DashboardSnapshot> loadDashboard() async {
    final assigned = await patients.assignedPatients();
    final open = await attentionEvents.openEvents();
    return DashboardSnapshot(
      openEvents: open,
      assignedPatients: assigned,
      fetchedAt: now(),
    );
  }
}
