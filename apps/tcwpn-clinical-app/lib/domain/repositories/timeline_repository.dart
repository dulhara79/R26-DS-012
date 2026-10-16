import '../contracts/timeline_entry.dart';

abstract interface class TimelineRepository {
  Future<List<TimelineEntry>> history(
    String subjectId, {
    int limit = 200,
  });
}
