import '../contracts/dashboard_snapshot.dart';

abstract interface class DashboardRepository {
  Future<DashboardSnapshot> loadDashboard();
}
