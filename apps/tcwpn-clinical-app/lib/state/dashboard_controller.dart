import 'package:flutter/foundation.dart';

import '../data/api/api_client.dart';
import '../data/local/dashboard_cache.dart';
import '../domain/contracts/dashboard_snapshot.dart';
import '../domain/repositories/auth_repository.dart';
import '../domain/repositories/dashboard_repository.dart';
import 'async_data_state.dart';

class DashboardController extends ChangeNotifier {
  final DashboardRepository repository;
  final AuthRepository? authRepository;
  final DashboardCacheStore cache;

  DashboardController({
    required this.repository,
    this.authRepository,
    DashboardCacheStore? cache,
  }) : cache = cache ?? const DashboardCacheStore();

  AsyncDataState<DashboardSnapshot> _state =
      const AsyncDataState<DashboardSnapshot>.loading();
  DashboardSnapshot? _lastSuccessful;

  AsyncDataState<DashboardSnapshot> get state => _state;

  Future<void> load({bool showLoading = true}) async {
    if (showLoading || _lastSuccessful == null) {
      _state = const AsyncDataState<DashboardSnapshot>.loading();
      notifyListeners();
    }

    try {
      await authRepository?.validateCurrentSession();
      final snapshot = await repository.loadDashboard();
      _lastSuccessful = snapshot;
      await cache.save(snapshot);
      _state = snapshot.isEmpty
          ? const AsyncDataState<DashboardSnapshot>.empty()
          : AsyncDataState<DashboardSnapshot>.data(snapshot);
    } on ApiException catch (e) {
      await _handleApiFailure(e);
    } on DashboardSessionExpiredException catch (e) {
      await authRepository?.expireCurrentSession();
      _state = AsyncDataState<DashboardSnapshot>.sessionExpired(
        message: e.message,
      );
    } on DashboardForbiddenException catch (e) {
      _state = AsyncDataState<DashboardSnapshot>.forbidden(message: e.message);
    } on DashboardConflictException catch (e) {
      _state = AsyncDataState<DashboardSnapshot>.conflict(message: e.message);
    } on DashboardOfflineException catch (e) {
      await _publishOffline(e.message);
    } on DashboardUnavailableException catch (e) {
      final cached = await _cachedSnapshot();
      _state = cached == null
          ? AsyncDataState<DashboardSnapshot>.unavailable(message: e.message)
          : AsyncDataState<DashboardSnapshot>.partial(
              cached,
              message:
                  'Assessment service unavailable. Showing the last server-provided dashboard snapshot.',
            );
    } catch (_) {
      _state = const AsyncDataState<DashboardSnapshot>.error(
        message: 'The clinician dashboard could not be loaded.',
      );
    }

    notifyListeners();
  }

  Future<void> _handleApiFailure(ApiException e) async {
    switch (e.kind) {
      case ApiFailure.unauthorized:
        await authRepository?.expireCurrentSession();
        _state = AsyncDataState<DashboardSnapshot>.sessionExpired(
          message: e.message,
        );
      case ApiFailure.forbidden:
        _state = AsyncDataState<DashboardSnapshot>.forbidden(
          message: e.message,
        );
      case ApiFailure.conflict:
        _state = AsyncDataState<DashboardSnapshot>.conflict(
          message: e.message,
        );
      case ApiFailure.offline:
      case ApiFailure.timeout:
      case ApiFailure.server:
        await _publishOffline(e.message);
      default:
        _state = AsyncDataState<DashboardSnapshot>.error(message: e.message);
    }
  }

  Future<void> _publishOffline(String message) async {
    final cached = await _cachedSnapshot();
    _state = AsyncDataState<DashboardSnapshot>.offline(
      cached: cached,
      message: cached == null
          ? message
          : 'Offline. Showing the last server-provided dashboard snapshot.',
    );
  }

  Future<DashboardSnapshot?> _cachedSnapshot() async {
    final inMemory = _lastSuccessful;
    if (inMemory != null) return inMemory.asCached();
    return cache.load();
  }
}

class DashboardSessionExpiredException implements Exception {
  final String message;
  const DashboardSessionExpiredException([
    this.message = 'Your clinician session has expired. Please sign in again.',
  ]);
}

class DashboardForbiddenException implements Exception {
  final String message;
  const DashboardForbiddenException([
    this.message =
        'You are signed in, but you do not have permission to access this dashboard data.',
  ]);
}

class DashboardConflictException implements Exception {
  final String message;
  const DashboardConflictException([
    this.message =
        'The server state changed. Refresh to load the current dashboard state.',
  ]);
}

class DashboardOfflineException implements Exception {
  final String message;
  const DashboardOfflineException([
    this.message = 'Dashboard data is unavailable while offline.',
  ]);
}

class DashboardUnavailableException implements Exception {
  final String message;
  const DashboardUnavailableException([
    this.message = 'Assessment service unavailable.',
  ]);
}
