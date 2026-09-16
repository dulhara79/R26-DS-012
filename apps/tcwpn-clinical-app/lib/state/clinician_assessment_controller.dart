import 'package:flutter/foundation.dart';

import '../domain/contracts/assessment_summary.dart';
import '../domain/contracts/clinician_assessment.dart';
import '../domain/repositories/clinician_assessment_repository.dart';

class ClinicianAssessmentController extends ChangeNotifier {
  final AssessmentSummary assessment;
  final String clinicianId;
  final ClinicianAssessmentRepository repository;

  String? _selectedTier;
  String _note = '';
  bool _submitting = false;
  ClinicianAssessmentReceipt? _receipt;
  String? _error;

  ClinicianAssessmentController({
    required this.assessment,
    required this.clinicianId,
    required this.repository,
  });

  String? get selectedTier => _selectedTier;
  String get note => _note;
  bool get submitting => _submitting;
  ClinicianAssessmentReceipt? get receipt => _receipt;
  String? get error => _error;
  bool get submitted => _receipt != null;

  bool get canSubmit =>
      !_submitting &&
      !submitted &&
      assessment.fusionResultId != null &&
      clinicianId.trim().isNotEmpty &&
      _selectedTier != null;

  void selectTier(String tier) {
    if (!clinicianAssessmentTierLabels.contains(tier)) {
      throw ArgumentError.value(
        tier,
        'tier',
        'must be one of Low, Medium, High',
      );
    }
    if (submitted) return;
    _selectedTier = tier;
    _error = null;
    notifyListeners();
  }

  void setNote(String value) {
    if (submitted) return;
    _note = value;
    notifyListeners();
  }

  Future<void> submit() async {
    if (_submitting || submitted) return;

    final fusionResultId = assessment.fusionResultId;
    if (fusionResultId == null) {
      _error = 'No authoritative fusion result is available.';
      notifyListeners();
      return;
    }

    final author = clinicianId.trim();
    if (author.isEmpty) {
      _error = 'Clinician identity is required before submission.';
      notifyListeners();
      return;
    }

    final tier = _selectedTier;
    if (tier == null) {
      _error = 'Select a clinician assessment tier.';
      notifyListeners();
      return;
    }

    final trimmedNote = _note.trim();
    final draft = ClinicianAssessmentDraft.validated(
      fusionResultId: fusionResultId,
      tierLabel: tier,
      author: author,
      note: trimmedNote.isEmpty ? null : trimmedNote,
    );

    _submitting = true;
    _error = null;
    notifyListeners();

    try {
      _receipt = await repository.submit(draft);
    } catch (e) {
      _error = '$e';
    } finally {
      _submitting = false;
      notifyListeners();
    }
  }
}
