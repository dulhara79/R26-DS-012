import 'package:flutter/foundation.dart';

import '../domain/evidence.dart';
import '../domain/repositories/evidence_repository.dart';

class SupportingEvidenceController extends ChangeNotifier {
  final String subjectId;
  final EvidenceRepository repository;

  bool _loading = false;
  EvidenceResult? _result;
  String? _askedQuestion;
  String? _error;

  SupportingEvidenceController({
    required this.subjectId,
    required this.repository,
  });

  bool get loading => _loading;
  EvidenceResult? get result => _result;
  String? get askedQuestion => _askedQuestion;
  String? get error => _error;

  Future<void> ask(String question) async {
    final trimmed = question.trim();
    if (trimmed.isEmpty) {
      _result = null;
      _error = 'Enter a question first.';
      notifyListeners();
      return;
    }
    if (_loading) return;

    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final next = await repository.ask(
        subjectId: subjectId,
        question: trimmed,
      );
      _askedQuestion = trimmed;
      _result = next;
    } catch (e) {
      _result = null;
      _error = '$e';
    } finally {
      _loading = false;
      notifyListeners();
    }
  }
}
