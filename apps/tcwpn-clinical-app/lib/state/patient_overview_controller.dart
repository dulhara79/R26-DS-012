import 'package:flutter/foundation.dart';

import '../data/api/api_client.dart';
import '../domain/contracts/assessment_summary.dart';
import '../domain/repositories/assessment_repository.dart';
import '../domain/repositories/auth_repository.dart';
import 'async_data_state.dart';

class PatientOverviewController extends ChangeNotifier {
  final String subjectId;
  final AssessmentRepository repository;
  final AuthRepository? authRepository;

  PatientOverviewController({
    required this.subjectId,
    required this.repository,
    this.authRepository,
  });

  AsyncDataState<AssessmentSummary> _state =
      const AsyncDataState<AssessmentSummary>.loading();

  AsyncDataState<AssessmentSummary> get state => _state;

  Future<void> load({bool showLoading = true}) async {
    if (showLoading) {
      _state = const AsyncDataState<AssessmentSummary>.loading();
      notifyListeners();
    }

    try {
      final assessment = await repository.latestAssessment(subjectId);
      _state = assessment == null
          ? const AsyncDataState<AssessmentSummary>.unavailable(
              message: 'Assessment unavailable — no current assessment was returned.',
            )
          : AsyncDataState<AssessmentSummary>.data(assessment);
    } on ApiException catch (e) {
      await _handleApiFailure(e);
    } catch (_) {
      _state = const AsyncDataState<AssessmentSummary>.error(
        message: 'The patient assessment could not be loaded.',
      );
    }

    notifyListeners();
  }

  Future<void> _handleApiFailure(ApiException e) async {
    switch (e.kind) {
      case ApiFailure.unauthorized:
        await authRepository?.expireCurrentSession();
        _state = AsyncDataState<AssessmentSummary>.sessionExpired(
          message: e.message,
        );
      case ApiFailure.forbidden:
        _state = AsyncDataState<AssessmentSummary>.forbidden(
          message: e.message,
        );
      case ApiFailure.conflict:
        _state = AsyncDataState<AssessmentSummary>.conflict(
          message: e.message,
        );
      case ApiFailure.notConfigured:
        _state = AsyncDataState<AssessmentSummary>.unavailable(
          message: e.message,
        );
      case ApiFailure.offline:
      case ApiFailure.timeout:
      case ApiFailure.server:
        _state = AsyncDataState<AssessmentSummary>.offline(
          message: e.message,
        );
      default:
        _state = AsyncDataState<AssessmentSummary>.error(
          message: e.message,
        );
    }
  }
}
