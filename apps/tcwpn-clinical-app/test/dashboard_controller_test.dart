import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/dashboard_snapshot.dart';
import 'package:r26_ds012_app/domain/repositories/auth_repository.dart';
import 'package:r26_ds012_app/domain/repositories/dashboard_repository.dart';
import 'package:r26_ds012_app/state/async_data_state.dart';
import 'package:r26_ds012_app/state/dashboard_controller.dart';

class _Repository implements DashboardRepository {
  DashboardSnapshot? value;
  Object? failure;

  @override
  Future<DashboardSnapshot> loadDashboard() async {
    if (failure != null) throw failure!;
    return value!;
  }
}

class _AuthRepository implements AuthRepository {
  Object? failure;
  int validations = 0;
  int expirations = 0;

  @override
  Future<void> validateCurrentSession() async {
    validations++;
    if (failure != null) throw failure!;
  }

  @override
  Future<void> expireCurrentSession() async {
    expirations++;
  }
}

DashboardSnapshot _empty() => DashboardSnapshot(
      openEvents: const [],
      assignedPatients: const [],
      fetchedAt: DateTime.utc(2026, 9, 16),
    );

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  test('empty server snapshot becomes explicit empty state', () async {
    final repo = _Repository()..value = _empty();
    final controller = DashboardController(repository: repo);

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.empty);
  });

  test('offline refresh preserves cached provenance', () async {
    final repo = _Repository()..value = _empty();
    final controller = DashboardController(repository: repo);
    await controller.load();

    repo.failure = const DashboardOfflineException();
    await controller.load(showLoading: false);

    expect(controller.state.status, AsyncDataStatus.offline);
    expect(controller.state.data, isNotNull);
    expect(controller.state.data!.isFromCache, isTrue);
  });

  test('offline after controller recreation reloads persisted snapshot', () async {
    final onlineRepo = _Repository()..value = _empty();
    await DashboardController(repository: onlineRepo).load();

    final offlineRepo = _Repository()
      ..failure = const DashboardOfflineException();
    final recreated = DashboardController(repository: offlineRepo);
    await recreated.load();

    expect(recreated.state.status, AsyncDataStatus.offline);
    expect(recreated.state.data, isNotNull);
    expect(recreated.state.data!.isFromCache, isTrue);
  });

  test('unavailable source without prior snapshot stays unavailable', () async {
    final repo = _Repository()
      ..failure = const DashboardUnavailableException();
    final controller = DashboardController(repository: repo);

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.unavailable);
    expect(controller.state.data, isNull);
  });

  test('expired clinician session is explicit and never shown from cache', () async {
    final repo = _Repository()
      ..failure = const DashboardSessionExpiredException();
    final controller = DashboardController(repository: repo);

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.sessionExpired);
    expect(controller.state.data, isNull);
  });

  test('forbidden clinician access is distinct from expired session', () async {
    final repo = _Repository()..failure = const DashboardForbiddenException();
    final controller = DashboardController(repository: repo);

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.forbidden);
    expect(controller.state.data, isNull);
  });

  test('server state conflict is explicit and does not use stale cache', () async {
    final repo = _Repository()..failure = const DashboardConflictException();
    final controller = DashboardController(repository: repo);

    await controller.load();

    expect(controller.state.status, AsyncDataStatus.conflict);
    expect(controller.state.data, isNull);
  });

  test('401 transport failure expires the clinician session', () async {
    final auth = _AuthRepository();
    final repo = _Repository()
      ..failure = const ApiException(kind: ApiFailure.unauthorized);
    final controller = DashboardController(
      repository: repo,
      authRepository: auth,
    );

    await controller.load();

    expect(auth.validations, 1);
    expect(auth.expirations, 1);
    expect(controller.state.status, AsyncDataStatus.sessionExpired);
    expect(controller.state.data, isNull);
  });

  test('403 transport failure keeps the valid clinician session', () async {
    final auth = _AuthRepository();
    final repo = _Repository()
      ..failure = const ApiException(kind: ApiFailure.forbidden);
    final controller = DashboardController(
      repository: repo,
      authRepository: auth,
    );

    await controller.load();

    expect(auth.validations, 1);
    expect(auth.expirations, 0);
    expect(controller.state.status, AsyncDataStatus.forbidden);
  });

  test('409 transport failure requests canonical server refresh', () async {
    final auth = _AuthRepository();
    final repo = _Repository()
      ..failure = const ApiException(kind: ApiFailure.conflict);
    final controller = DashboardController(
      repository: repo,
      authRepository: auth,
    );

    await controller.load();

    expect(auth.expirations, 0);
    expect(controller.state.status, AsyncDataStatus.conflict);
  });
}
