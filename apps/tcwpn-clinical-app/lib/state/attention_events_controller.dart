import 'package:flutter/foundation.dart';

import '../data/api/api_client.dart';
import '../domain/contracts/attention_event.dart';
import '../domain/repositories/attention_event_repository.dart';
import '../domain/repositories/auth_repository.dart';
import 'async_data_state.dart';

class AttentionEventsController extends ChangeNotifier {
  final AttentionEventRepository repository;
  final AuthRepository? authRepository;
  final String? subjectId;

  AttentionEventsController({
    required this.repository,
    this.authRepository,
    this.subjectId,
  });

  AsyncDataState<List<AttentionEvent>> _state =
      const AsyncDataState<List<AttentionEvent>>.loading();

  AsyncDataState<List<AttentionEvent>> get state => _state;

  Future<void> load({bool showLoading = true}) async {
    if (showLoading) {
      _state = const AsyncDataState<List<AttentionEvent>>.loading();
      notifyListeners();
    }

    try {
      final events = await repository.activity(subjectId: subjectId);
      _state = events.isEmpty
          ? const AsyncDataState<List<AttentionEvent>>.empty()
          : AsyncDataState<List<AttentionEvent>>.data(events);
    } on ApiException catch (e) {
      await _handleApiFailure(e);
    } catch (_) {
      _state = const AsyncDataState<List<AttentionEvent>>.error(
        message: 'Attention-event activity could not be loaded.',
      );
    }

    notifyListeners();
  }

  Future<void> _handleApiFailure(ApiException e) async {
    switch (e.kind) {
      case ApiFailure.unauthorized:
        await authRepository?.expireCurrentSession();
        _state = AsyncDataState<List<AttentionEvent>>.sessionExpired(
          message: e.message,
        );
      case ApiFailure.forbidden:
        _state = AsyncDataState<List<AttentionEvent>>.forbidden(
          message: e.message,
        );
      case ApiFailure.conflict:
        _state = AsyncDataState<List<AttentionEvent>>.conflict(
          message: e.message,
        );
      case ApiFailure.notConfigured:
      case ApiFailure.notFound:
      case ApiFailure.timeout:
        _state = AsyncDataState<List<AttentionEvent>>.unavailable(
          message: e.message,
        );
      case ApiFailure.offline:
        _state = AsyncDataState<List<AttentionEvent>>.offline(
          message: e.message,
        );
      case ApiFailure.server:
        _state = AsyncDataState<List<AttentionEvent>>.error(
          message: e.message,
        );
      default:
        _state = AsyncDataState<List<AttentionEvent>>.error(
          message: e.message,
        );
    }
  }
}
