import 'package:flutter/foundation.dart';

import '../data/api/api_client.dart';
import '../domain/contracts/attention_event.dart';
import '../domain/contracts/contract_enums.dart';
import '../domain/repositories/attention_event_repository.dart';
import '../domain/repositories/auth_repository.dart';
import 'async_data_state.dart';

class AttentionEventDetailController extends ChangeNotifier {
  final String eventId;
  final AttentionEventRepository repository;
  final AuthRepository? authRepository;

  AttentionEventDetailController({
    required this.eventId,
    required this.repository,
    this.authRepository,
  });

  AsyncDataState<AttentionEvent> _state =
      const AsyncDataState<AttentionEvent>.loading();
  bool _isMutating = false;
  String? _mutationMessage;

  AsyncDataState<AttentionEvent> get state => _state;
  bool get isMutating => _isMutating;
  String? get mutationMessage => _mutationMessage;

  Future<void> load({bool showLoading = true}) async {
    if (showLoading) {
      _state = const AsyncDataState<AttentionEvent>.loading();
      notifyListeners();
    }

    _mutationMessage = null;
    try {
      final event = await repository.eventById(eventId);
      _state = event == null
          ? const AsyncDataState<AttentionEvent>.unavailable(
              message: 'This attention event is no longer available on the server.',
            )
          : AsyncDataState<AttentionEvent>.data(event);
    } on ApiException catch (e) {
      await _handleLoadFailure(e);
    } catch (_) {
      _state = const AsyncDataState<AttentionEvent>.error(
        message: 'The attention event could not be loaded.',
      );
    }

    notifyListeners();
  }

  Future<void> acknowledge() async {
    if (_isMutating) return;
    final current = _state.data;
    if (current == null || current.status != AttentionEventStatus.open) return;
    await _mutate(current, () => repository.acknowledge(eventId));
  }

  Future<void> resolve() async {
    if (_isMutating) return;
    final current = _state.data;
    if (current == null ||
        current.status != AttentionEventStatus.acknowledged) {
      return;
    }
    await _mutate(current, () => repository.resolve(eventId));
  }

  Future<void> _mutate(
    AttentionEvent current,
    Future<AttentionEvent> Function() request,
  ) async {
    _isMutating = true;
    _mutationMessage = null;
    notifyListeners();

    try {
      final canonical = await request();
      _state = AsyncDataState<AttentionEvent>.data(canonical);
    } on ApiException catch (e) {
      if (e.kind == ApiFailure.conflict) {
        await _reconcileConflict();
      } else {
        await _handleMutationFailure(e, current);
      }
    } catch (_) {
      _state = const AsyncDataState<AttentionEvent>.error(
        message: 'The attention-event action could not be completed.',
      );
      _mutationMessage =
          'The action failed. The app did not change the server event state.';
    } finally {
      _isMutating = false;
      notifyListeners();
    }
  }

  Future<void> _reconcileConflict() async {
    try {
      final canonical = await repository.eventById(eventId);
      _state = canonical == null
          ? const AsyncDataState<AttentionEvent>.unavailable(
              message: 'This attention event is no longer available on the server.',
            )
          : AsyncDataState<AttentionEvent>.data(canonical);
      _mutationMessage =
          'This event changed on the server. The current server state is shown.';
    } on ApiException catch (e) {
      await _handleLoadFailure(e);
      _mutationMessage =
          'This event changed on the server, but the current server state could not be refreshed.';
    } catch (_) {
      _state = const AsyncDataState<AttentionEvent>.error(
        message: 'The current server event state could not be refreshed.',
      );
      _mutationMessage =
          'This event changed on the server, but the current server state could not be refreshed.';
    }
  }

  Future<void> _handleLoadFailure(ApiException e) async {
    switch (e.kind) {
      case ApiFailure.unauthorized:
        await authRepository?.expireCurrentSession();
        _state = AsyncDataState<AttentionEvent>.sessionExpired(
          message: e.message,
        );
      case ApiFailure.forbidden:
        _state = AsyncDataState<AttentionEvent>.forbidden(message: e.message);
      case ApiFailure.notFound:
      case ApiFailure.notConfigured:
      case ApiFailure.timeout:
        _state = AsyncDataState<AttentionEvent>.unavailable(message: e.message);
      case ApiFailure.offline:
        _state = AsyncDataState<AttentionEvent>.offline(message: e.message);
      case ApiFailure.conflict:
        _state = AsyncDataState<AttentionEvent>.conflict(message: e.message);
      case ApiFailure.server:
        _state = AsyncDataState<AttentionEvent>.error(message: e.message);
      default:
        _state = AsyncDataState<AttentionEvent>.error(message: e.message);
    }
  }

  Future<void> _handleMutationFailure(
    ApiException e,
    AttentionEvent current,
  ) async {
    switch (e.kind) {
      case ApiFailure.unauthorized:
        await authRepository?.expireCurrentSession();
        _state = AsyncDataState<AttentionEvent>.sessionExpired(
          message: e.message,
        );
      case ApiFailure.forbidden:
        _state = AsyncDataState<AttentionEvent>.forbidden(
          data: current,
          message: e.message,
        );
      case ApiFailure.offline:
        _state = AsyncDataState<AttentionEvent>.offline(
          cached: current,
          message: e.message,
        );
      case ApiFailure.notFound:
        _state = AsyncDataState<AttentionEvent>.unavailable(message: e.message);
      case ApiFailure.notConfigured:
      case ApiFailure.timeout:
        _state = AsyncDataState<AttentionEvent>.partial(
          current,
          message: e.message,
        );
      case ApiFailure.server:
      case ApiFailure.validation:
      case ApiFailure.malformed:
      case ApiFailure.insecureConnection:
      case ApiFailure.unknown:
        _state = AsyncDataState<AttentionEvent>.partial(
          current,
          message: e.message,
        );
      case ApiFailure.conflict:
        break;
    }
    _mutationMessage =
        'The action was not confirmed by the server. The last canonical event state is still shown.';
  }
}
