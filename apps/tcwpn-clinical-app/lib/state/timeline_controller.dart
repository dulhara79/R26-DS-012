import 'package:flutter/foundation.dart';

import '../data/api/api_client.dart';
import '../domain/contracts/timeline_entry.dart';
import '../domain/repositories/timeline_repository.dart';
import 'async_data_state.dart';

enum TimelineWindow { hours24, days7 }

class TimelineController extends ChangeNotifier {
  final String subjectId;
  final TimelineRepository repository;
  final DateTime Function() now;

  AsyncDataState<List<TimelineEntry>> _state = const AsyncDataState.loading();
  TimelineWindow _window = TimelineWindow.hours24;

  TimelineController({
    required this.subjectId,
    required this.repository,
    DateTime Function()? now,
  }) : now = now ?? DateTime.now;

  AsyncDataState<List<TimelineEntry>> get state => _state;
  TimelineWindow get window => _window;

  List<TimelineEntry> get visibleEntries {
    final rows = _state.data ?? const <TimelineEntry>[];
    final cutoff = now().subtract(
      _window == TimelineWindow.hours24
          ? const Duration(hours: 24)
          : const Duration(days: 7),
    );
    return rows
        .where((entry) =>
            entry.computedAt == null || !entry.computedAt!.isBefore(cutoff))
        .toList(growable: false);
  }

  void setWindow(TimelineWindow value) {
    if (_window == value) return;
    _window = value;
    notifyListeners();
  }

  Future<void> load({bool showLoading = true}) async {
    if (showLoading) {
      _state = const AsyncDataState.loading();
      notifyListeners();
    }
    try {
      final rows = await repository.history(subjectId);
      _state = rows.isEmpty
          ? const AsyncDataState.empty()
          : AsyncDataState.data(List<TimelineEntry>.unmodifiable(rows));
    } on ApiException catch (error) {
      _state = switch (error.kind) {
        ApiFailure.offline || ApiFailure.timeout =>
          AsyncDataState.offline(message: error.message),
        ApiFailure.unauthorized =>
          AsyncDataState.sessionExpired(message: error.message),
        ApiFailure.forbidden => AsyncDataState.forbidden(message: error.message),
        ApiFailure.conflict => AsyncDataState.conflict(message: error.message),
        ApiFailure.notConfigured || ApiFailure.notFound =>
          AsyncDataState.unavailable(message: error.message),
        _ => AsyncDataState.error(message: error.message),
      };
    } catch (_) {
      _state = const AsyncDataState.error(
        message: 'Timeline history could not be loaded.',
      );
    }
    notifyListeners();
  }
}
